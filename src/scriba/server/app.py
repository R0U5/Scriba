"""HTTP API + the single-page UI.

Design notes:

* Job state changes are pushed to the browser over Server-Sent Events, so the
  UI never polls and progress feels instant.
* The heavy work happens on worker threads owned by :class:`JobManager`; the
  event loop only shuffles JSON.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, get_settings
from ..engines import catalogue, detect_gpu
from ..errors import ScribaError
from ..jobs import JobManager
from ..logging_setup import get_logger, setup_logging
from ..media.ffmpeg import ffmpeg_available, ffmpeg_version
from ..models import Job, TranscriptionOptions

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class SubmitRequest(BaseModel):
    url: str = Field(..., description="Media URL, or a local file path")
    options: TranscriptionOptions = Field(default_factory=TranscriptionOptions)


class EventHub:
    """Fan-out of job updates from worker threads to SSE subscribers."""

    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[str]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self) -> asyncio.Queue[str]:
        client: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._clients.add(client)
        return client

    def unregister(self, client: asyncio.Queue[str]) -> None:
        self._clients.discard(client)

    def publish(self, job: Job) -> None:
        """Called from a worker thread — hop onto the event loop safely."""
        if self._loop is None or self._loop.is_closed():
            return
        payload = job.model_dump_json()
        self._loop.call_soon_threadsafe(self._dispatch, payload)

    def _dispatch(self, payload: str) -> None:
        for client in list(self._clients):
            try:
                client.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover - slow consumer
                self._clients.discard(client)


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    setup_logging(settings.log_level)

    hub = EventHub()
    jobs = manager or JobManager(settings)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        hub.bind(asyncio.get_running_loop())
        unsubscribe = jobs.subscribe(hub.publish)
        jobs.start()
        log.info("Scriba %s ready — transcripts go to %s", __version__, settings.output_dir)
        try:
            yield
        finally:
            unsubscribe()
            jobs.stop()

    app = FastAPI(
        title="Scriba",
        version=__version__,
        description="Local audio transcription with state-of-the-art open models.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.hub = hub

    # ---------------------------------------------------------------- info
    @app.get("/api/health")
    async def health() -> dict:
        gpu = detect_gpu()
        return {
            "version": __version__,
            "ok": True,
            "gpu": {
                "available": gpu.available,
                "name": gpu.name,
                "vram_gb": gpu.total_vram_gb,
                "label": gpu.label,
            },
            "ffmpeg": {
                "available": ffmpeg_available(),
                "version": ffmpeg_version() if ffmpeg_available() else None,
            },
            "output_dir": str(settings.output_dir),
            "queue_depth": jobs.queue_depth(),
            "active_job": jobs.active_job_id,
            "stats": jobs.store.stats(),
        }

    @app.get("/api/engines")
    async def engines() -> dict:
        return {"engines": catalogue(), "default": settings.engine}

    # ---------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50) -> dict:
        return {"jobs": [job.model_dump(mode="json") for job in jobs.list(limit=limit)]}

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: SubmitRequest) -> dict:
        try:
            job = jobs.submit(request.url, request.options)
        except ScribaError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
        return job.model_dump(mode="json")

    @app.get("/api/jobs/{job_id}")
    async def read_job(job_id: str) -> dict:
        return _require(jobs, job_id).model_dump(mode="json")

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        _require(jobs, job_id)
        return {"cancelled": jobs.cancel(job_id)}

    @app.post("/api/jobs/{job_id}/retry", status_code=201)
    async def retry_job(job_id: str) -> dict:
        job = jobs.retry(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        return job.model_dump(mode="json")

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict:
        return {"deleted": jobs.delete(job_id)}

    @app.delete("/api/history")
    async def clear_history() -> dict:
        return {"removed": jobs.clear_history()}

    @app.get("/api/jobs/{job_id}/text", response_class=PlainTextResponse)
    async def job_text(job_id: str) -> str:
        job = _require(jobs, job_id)
        if job.result is None:
            raise HTTPException(status_code=409, detail={"message": "Transcript is not ready yet"})
        return job.result.text

    @app.get("/api/jobs/{job_id}/download/{fmt}")
    async def download(job_id: str, fmt: str):
        job = _require(jobs, job_id)
        path_text = job.outputs.get(fmt)
        if not path_text:
            raise HTTPException(
                status_code=404, detail={"message": f"No .{fmt} file was written for this job"}
            )
        path = Path(path_text)
        if not path.exists():
            raise HTTPException(
                status_code=410,
                detail={"message": f"{path.name} is no longer on disk (moved or deleted)"},
            )
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.post("/api/jobs/{job_id}/reveal")
    async def reveal(job_id: str, request: Request) -> dict:
        """Open the transcript folder in the OS file manager (local use only)."""
        if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(status_code=403, detail={"message": "Only available locally"})
        job = _require(jobs, job_id)
        if not job.outputs:
            raise HTTPException(status_code=404, detail={"message": "Nothing written yet"})
        target = Path(next(iter(job.outputs.values())))
        return {"revealed": _reveal_in_file_manager(target)}

    # -------------------------------------------------------------- events
    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        client = hub.register()

        async def stream():
            try:
                snapshot = [job.model_dump(mode="json") for job in jobs.list(limit=50)]
                yield _sse("snapshot", {"jobs": snapshot})
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(client.get(), timeout=15.0)
                        yield f"event: job\ndata: {payload}\n\n"
                    except TimeoutError:
                        yield f": keepalive {int(time.time())}\n\n"
            finally:
                hub.unregister(client)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    async def read_settings() -> dict:
        return {
            "output_dir": str(settings.output_dir),
            "engine": settings.engine,
            "model": settings.model,
            "device": settings.device,
            "keep_audio": settings.keep_audio,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "allow_private_hosts": settings.allow_private_hosts,
        }

    @app.patch("/api/settings")
    async def update_settings(payload: dict = Body(...)) -> dict:
        """Change the output directory at runtime (the only hot-editable setting)."""
        if "output_dir" in payload:
            directory = Path(str(payload["output_dir"])).expanduser()
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise HTTPException(
                    status_code=400, detail={"message": f"Cannot use that folder: {exc}"}
                ) from exc
            settings.output_dir = directory.resolve()
        return {"output_dir": str(settings.output_dir)}

    # ------------------------------------------------------------------ UI
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    return app


def _require(jobs: JobManager, job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"message": f"Job {job_id} not found"})
    return job


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _reveal_in_file_manager(path: Path) -> bool:
    system = platform.system()
    try:
        if system == "Windows":
            # explorer wants "/select,<path>" as a single argument.
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
        return True
    except Exception as exc:  # pragma: no cover - desktop-dependent
        log.debug("Could not reveal %s: %s", path, exc)
        return False
