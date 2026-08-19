import contextlib
import json
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from rosetta.jobs import JobManager
from rosetta.models import TranscriptionOptions
from rosetta.server.app import create_app


@pytest.fixture
def client(settings):
    manager = JobManager(settings)
    app = create_app(settings, manager)
    with TestClient(app) as test_client:
        yield test_client
    manager.stop()


def _wait(client, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("completed", "failed", "cancelled"):
            return payload
        time.sleep(0.03)
    raise AssertionError("job did not finish")


def test_health_reports_the_environment(client):
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert "available" in payload["gpu"]
    assert payload["ffmpeg"]["available"] in (True, False)
    assert payload["output_dir"]


def test_engine_catalogue_is_served(client):
    payload = client.get("/api/engines").json()
    keys = {engine["key"] for engine in payload["engines"]}
    assert "faster-whisper" in keys


def test_full_job_lifecycle(client, tone_mp3):
    created = client.post(
        "/api/jobs",
        json={"url": str(tone_mp3), "options": {"engine": "stub", "formats": ["txt", "srt"]}},
    )
    assert created.status_code == 201
    job_id = created.json()["id"]

    finished = _wait(client, job_id)
    assert finished["status"] == "completed"
    assert set(finished["outputs"]) == {"txt", "srt"}

    text = client.get(f"/api/jobs/{job_id}/text")
    assert text.status_code == 200
    assert "Stub segment" in text.text

    download = client.get(f"/api/jobs/{job_id}/download/srt")
    assert download.status_code == 200
    assert "-->" in download.text

    assert client.get(f"/api/jobs/{job_id}/download/vtt").status_code == 404

    listing = client.get("/api/jobs").json()["jobs"]
    assert any(job["id"] == job_id for job in listing)

    assert client.delete(f"/api/jobs/{job_id}").json() == {"deleted": True}
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_invalid_url_returns_a_typed_error(client):
    created = client.post("/api/jobs", json={"url": "ftp://example.com/a.mp3",
                                             "options": {"engine": "stub"}})
    payload = _wait(client, created.json()["id"])
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "invalid_url"


def test_retry_creates_a_second_job(client, tone_mp3):
    first = client.post("/api/jobs", json={"url": str(tone_mp3), "options": {"engine": "stub"}}).json()
    _wait(client, first["id"])
    second = client.post(f"/api/jobs/{first['id']}/retry")
    assert second.status_code == 201
    assert second.json()["id"] != first["id"]
    assert _wait(client, second.json()["id"])["status"] == "completed"


def test_missing_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_settings_patch_changes_the_output_folder(client, tmp_path):
    target = tmp_path / "elsewhere"
    payload = client.patch("/api/settings", json={"output_dir": str(target)}).json()
    assert payload["output_dir"] == str(target.resolve())
    assert target.exists()


def test_event_stream_pushes_live_job_updates(settings, tone_mp3):
    """SSE is long-lived, so this runs against a real uvicorn server."""
    import httpx

    manager = JobManager(settings)
    app = create_app(settings, manager)
    with (
        _serve(app) as base_url,
        httpx.Client(timeout=30.0) as http,
        http.stream("GET", f"{base_url}/api/events") as stream,
    ):
        assert stream.headers["content-type"].startswith("text/event-stream")
        events, job = [], None
        for line in stream.iter_lines():
            if not line.startswith("data:"):
                continue
            events.append(json.loads(line[5:]))
            if job is None:  # subscribe first, then give it something to report
                job = manager.submit(str(tone_mp3), TranscriptionOptions(engine="stub"))
            elif len(events) >= 2:
                break
    manager.stop()

    assert "jobs" in events[0]           # opening snapshot
    assert events[1]["id"] == job.id     # live update for the job we submitted


@contextlib.contextmanager
def _serve(app):
    """Run ``app`` on a background uvicorn server bound to a free port."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "the test server did not come up"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "RosettaAI" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
