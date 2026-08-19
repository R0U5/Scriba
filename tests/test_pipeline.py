import threading
import time

import pytest

from rosetta.errors import DownloadError
from rosetta.jobs import JobManager
from rosetta.models import Job, JobStage, JobStatus, OutputFormat, TranscriptionOptions
from rosetta.pipeline import Pipeline, transcribe_url
from rosetta.store import HistoryStore


def _options(**kwargs):
    base = {"engine": "stub", "formats": [OutputFormat.TXT, OutputFormat.SRT, OutputFormat.JSON]}
    base.update(kwargs)
    return TranscriptionOptions(**base)


def test_pipeline_writes_all_requested_formats(settings, tone_mp3):
    job = transcribe_url(str(tone_mp3), settings, _options())
    assert job.status is JobStatus.COMPLETED
    assert job.stage is JobStage.DONE
    assert set(job.outputs) == {"txt", "srt", "json"}
    for path in job.outputs.values():
        assert path.startswith(str(settings.output_dir))
    assert job.result.word_count > 0
    assert "Stub segment 0" in job.result.text


def test_pipeline_progress_is_monotonic_and_complete(settings, tone_mp3):
    seen: list[float] = []
    transcribe_url(str(tone_mp3), settings, _options(), emit=lambda job: seen.append(job.progress))
    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_pipeline_cleans_up_its_work_directory(settings, tone_mp3):
    job = transcribe_url(str(tone_mp3), settings, _options())
    assert not (settings.work_dir / job.id).exists()


def test_keep_audio_preserves_the_wav(settings, tone_mp3):
    settings.keep_audio = True
    job = transcribe_url(str(tone_mp3), settings, _options())
    assert "wav" in job.outputs
    assert job.outputs["wav"].endswith(".wav")


def test_failure_is_reported_with_a_hint(settings):
    job = transcribe_url("ftp://example.com/a.mp3", settings, _options())
    assert job.status is JobStatus.FAILED
    assert job.error["code"] == "invalid_url"
    assert job.error["hint"]


def test_cancellation_stops_the_run(settings, tone_mp3):
    job = Job(url=str(tone_mp3), options=_options())
    pipeline = Pipeline(settings, cancelled=lambda: True)
    pipeline.run(job)
    assert job.status is JobStatus.CANCELLED
    assert not job.outputs


def test_download_errors_are_typed(settings, monkeypatch):
    def explode(*args, **kwargs):
        raise DownloadError("nope", hint="try again")

    monkeypatch.setattr("rosetta.pipeline.fetch_media", explode)
    job = transcribe_url("https://example.com/a.mp3", settings, _options())
    assert job.status is JobStatus.FAILED
    assert job.error["message"] == "nope"


# ------------------------------------------------------------------- queue
def _wait(manager: JobManager, job_id: str, timeout: float = 30.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job and job.is_terminal:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


@pytest.fixture
def manager(settings):
    instance = JobManager(settings)
    instance.start()
    yield instance
    instance.stop()


def test_manager_runs_a_job_and_notifies_listeners(manager, tone_mp3):
    events: list[Job] = []
    done = threading.Event()

    def listen(job: Job) -> None:
        events.append(job)
        if job.is_terminal:
            done.set()

    manager.subscribe(listen)
    job = manager.submit(str(tone_mp3), _options())
    assert done.wait(timeout=30), "no terminal event was broadcast"
    assert _wait(manager, job.id).status is JobStatus.COMPLETED

    stages = [event.stage.value for event in events]
    assert "transcribing" in stages
    assert events[-1].stage.value == "done"
    assert events[-1].status is JobStatus.COMPLETED


def test_history_survives_a_new_manager(manager, tone_mp3, settings):
    job = manager.submit(str(tone_mp3), _options())
    _wait(manager, job.id)
    manager.stop()

    revived = JobManager(settings)
    stored = revived.get(job.id)
    assert stored is not None
    assert stored.status is JobStatus.COMPLETED
    assert stored.outputs == job.outputs


def test_cancel_while_queued_marks_the_job_cancelled(settings, tone_mp3, monkeypatch):
    instance = JobManager(settings)
    monkeypatch.setattr(instance, "start", lambda: None)  # keep the job queued
    job = instance.submit(str(tone_mp3), _options())

    assert instance.cancel(job.id)
    assert instance.get(job.id).status is JobStatus.CANCELLED

    # A worker that later picks the job up must skip it rather than run it.
    instance._run_job(job.id)
    assert instance.get(job.id).status is JobStatus.CANCELLED
    assert not instance.get(job.id).outputs


def test_retry_creates_a_new_job(manager, tone_mp3):
    job = manager.submit(str(tone_mp3), _options())
    _wait(manager, job.id)
    again = manager.retry(job.id)
    assert again.id != job.id
    assert _wait(manager, again.id).status is JobStatus.COMPLETED


def test_delete_removes_from_memory_and_history(manager, tone_mp3):
    job = manager.submit(str(tone_mp3), _options())
    _wait(manager, job.id)
    assert manager.delete(job.id)
    assert manager.get(job.id) is None
    # A worker finishing its bookkeeping must not resurrect a deleted job.
    time.sleep(0.3)
    assert manager.store.get(job.id) is None


def test_store_prunes_to_the_history_limit(settings):
    store = HistoryStore(settings.db_path)
    for index in range(15):
        store.save(Job(url=f"https://example.com/{index}.mp3"))
    store.prune(10)
    assert len(store.recent(50)) == 10
    assert store.stats()["jobs"] == 10


def test_live_preview_streams_partial_segments(settings, tone_mp3, monkeypatch):
    """The UI shows text arriving, so partial results must be published."""
    monkeypatch.setattr("rosetta.pipeline.LIVE_INTERVAL", 0.0)
    partials: list[int] = []

    def emit(job: Job) -> None:
        if job.result and job.result.partial:
            partials.append(len(job.result.segments))

    job = transcribe_url(str(tone_mp3), settings, _options(), emit=emit)
    assert partials, "no live preview was published"
    assert partials == sorted(partials)
    assert job.result.partial is False
    assert len(job.result.segments) >= max(partials)
