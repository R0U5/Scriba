"""The transcription pipeline: URL in, transcript files out.

Every stage reports fractional progress; the pipeline maps those onto a single
0-100% bar so the UI can show one honest number for the whole run.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .config import Settings
from .engines import create_engine
from .errors import JobCancelled, RosettaError
from .logging_setup import get_logger
from .media.audio import decode_to_wav
from .media.fetch import fetch_media
from .models import Job, JobStage, JobStatus, Segment, TranscriptResult
from .util.text import slugify
from .writers import write_outputs

log = get_logger(__name__)

Emit = Callable[[Job], None]
Cancelled = Callable[[], bool]

#: How many recent segments the live preview carries. The finished transcript
#: is sent once at the end; streaming every segment of a 3-hour file on every
#: update would be pointless traffic.
LIVE_TAIL = 40
#: Minimum seconds between live-preview publishes.
LIVE_INTERVAL = 0.8

#: Fraction of the overall progress bar owned by each stage.
STAGE_SPANS: dict[JobStage, tuple[float, float]] = {
    JobStage.RESOLVING: (0.00, 0.03),
    JobStage.DOWNLOADING: (0.03, 0.28),
    JobStage.DECODING: (0.28, 0.36),
    JobStage.LOADING_MODEL: (0.36, 0.44),
    JobStage.TRANSCRIBING: (0.44, 0.96),
    JobStage.WRITING: (0.96, 1.00),
}


class Pipeline:
    """Runs one job at a time, start to finish."""

    def __init__(
        self,
        settings: Settings,
        *,
        emit: Emit | None = None,
        cancelled: Cancelled | None = None,
    ) -> None:
        self.settings = settings
        self._emit = emit
        self._cancelled = cancelled

    # -- progress plumbing -------------------------------------------------
    def _publish(self, job: Job) -> None:
        if self._emit:
            self._emit(job)

    def _stage(self, job: Job, stage: JobStage, message: str) -> None:
        job.stage = stage
        job.message = message
        job.progress = STAGE_SPANS.get(stage, (job.progress, job.progress))[0]
        self._publish(job)

    def _progress_for(self, job: Job, stage: JobStage) -> Callable[[float, str], None]:
        low, high = STAGE_SPANS[stage]
        last = 0.0

        def report(fraction: float, message: str) -> None:
            nonlocal last
            value = low + (high - low) * max(0.0, min(fraction, 1.0))
            # Only publish meaningful movement: the UI animates the rest.
            if value - last >= 0.002 or message != job.message:
                last = value
                job.progress = round(value, 4)
                job.message = message
                job.stage = stage
                self._publish(job)

        return report

    def _check_cancelled(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise JobCancelled()


    def _live_preview(self, job: Job, engine, duration: float):
        """Publish a rolling tail of segments so the UI fills in as we go."""
        collected: list[Segment] = []
        last = 0.0

        def on_segment(segment: Segment) -> None:
            nonlocal last
            collected.append(segment)
            now = time.monotonic()
            if now - last < LIVE_INTERVAL:
                return
            last = now
            job.result = TranscriptResult(
                text=" ".join(s.text for s in collected[-LIVE_TAIL:]),
                segments=collected[-LIVE_TAIL:],
                duration=duration,
                engine=engine.spec.key,
                model=engine.model_id,
                device=engine.device,
                compute_type=engine.compute_type,
                word_count=sum(len(s.text.split()) for s in collected),
                partial=True,
            )
            self._publish(job)

        return on_segment

    # -- main --------------------------------------------------------------
    def run(self, job: Job) -> Job:
        settings = self.settings
        workdir = settings.work_dir / job.id
        workdir.mkdir(parents=True, exist_ok=True)

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        started = time.monotonic()

        try:
            # 1. Resolve + download ---------------------------------------
            self._stage(job, JobStage.RESOLVING, "Resolving source…")
            self._check_cancelled()

            self._stage(job, JobStage.DOWNLOADING, "Fetching media…")
            fetched = fetch_media(
                job.url, workdir, settings,
                progress=self._progress_for(job, JobStage.DOWNLOADING),
                cancelled=self._cancelled,
            )
            job.media = fetched.info
            self._publish(job)
            self._check_cancelled()

            # 2. Decode ----------------------------------------------------
            self._stage(job, JobStage.DECODING, "Decoding audio…")
            audio = decode_to_wav(
                fetched.path,
                workdir / "audio.wav",
                hint_duration=fetched.info.duration or None,
                progress=self._progress_for(job, JobStage.DECODING),
                cancelled=self._cancelled,
            )
            if job.media:
                job.media.duration = audio.duration
            self._publish(job)
            self._check_cancelled()

            # 3. Load the model -------------------------------------------
            self._stage(job, JobStage.LOADING_MODEL, "Preparing model…")
            engine = create_engine(
                job.options.engine,
                model=job.options.model,
                device=job.options.device or settings.device,
                compute_type=job.options.compute_type or settings.compute_type,
            )
            job.options.engine = engine.spec.key
            job.options.model = engine.model_id
            job.options.device = engine.device
            job.options.compute_type = engine.compute_type
            engine.load(progress=self._progress_for(job, JobStage.LOADING_MODEL))
            self._check_cancelled()

            # 4. Transcribe ------------------------------------------------
            self._stage(job, JobStage.TRANSCRIBING, "Transcribing…")
            result: TranscriptResult = engine.transcribe(
                audio,
                job.options,
                progress=self._progress_for(job, JobStage.TRANSCRIBING),
                cancelled=self._cancelled,
                on_segment=self._live_preview(job, engine, audio.duration),
            )
            job.result = result
            self._check_cancelled()

            # 5. Write files ----------------------------------------------
            self._stage(job, JobStage.WRITING, "Writing transcript files…")
            output_dir = Path(job.options.output_dir or settings.output_dir).expanduser()
            stem = slugify(
                (job.media.title if job.media else None) or Path(job.url).stem, fallback=job.id
            )
            job.outputs = write_outputs(result, job.media, job.options, output_dir, stem)

            if settings.keep_audio:
                kept = output_dir / f"{Path(next(iter(job.outputs.values()))).stem}.wav"
                shutil.move(str(audio.path), kept)
                job.outputs["wav"] = str(kept)

            job.status = JobStatus.COMPLETED
            job.stage = JobStage.DONE
            job.progress = 1.0
            job.message = (
                f"Done in {time.monotonic() - started:.0f}s · "
                f"{result.word_count:,} words · {result.speed_factor:.0f}x realtime"
            )
            log.info("%s -> %s", job.summary(), job.outputs.get("txt", ""))

        except JobCancelled:
            job.status = JobStatus.CANCELLED
            job.message = "Cancelled"
            job.error = None
            log.info("Job %s cancelled", job.id)
        except RosettaError as exc:
            job.status = JobStatus.FAILED
            job.message = exc.message
            job.error = exc.to_dict()
            log.error("Job %s failed: %s", job.id, exc.message)
        except Exception as exc:  # pragma: no cover - unexpected bug
            job.status = JobStatus.FAILED
            job.message = f"Unexpected error: {exc}"
            job.error = {"code": "internal_error", "message": str(exc), "hint": None}
            log.exception("Job %s crashed", job.id)
        finally:
            job.finished_at = time.time()
            if not settings.keep_audio:
                shutil.rmtree(workdir, ignore_errors=True)
            self._publish(job)

        return job


def transcribe_url(
    url: str,
    settings: Settings,
    options=None,
    *,
    emit: Emit | None = None,
    cancelled: Cancelled | None = None,
) -> Job:
    """Convenience wrapper used by the CLI: run one URL synchronously."""
    from .models import TranscriptionOptions

    job = Job(url=url, options=options or TranscriptionOptions())
    return Pipeline(settings, emit=emit, cancelled=cancelled).run(job)
