"""Job queue: submit URLs, run them one at a time, stream progress to listeners.

The GPU is the scarce resource, so jobs run on a small pool of worker threads
(one by default). Everything that touches shared state goes through a lock, and
progress updates are fanned out to any number of subscribers — the web UI's SSE
stream, the CLI's progress bar, or your own callback.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .config import Settings
from .errors import ScribaError
from .logging_setup import get_logger
from .models import Job, JobStage, JobStatus, TranscriptionOptions
from .pipeline import Pipeline
from .store import HistoryStore

log = get_logger(__name__)

Listener = Callable[[Job], None]


class JobManager:
    """Owns the queue, the workers, the in-memory job table and the history."""

    def __init__(self, settings: Settings, store: HistoryStore | None = None) -> None:
        self.settings = settings
        self.store = store or HistoryStore(settings.db_path)
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._cancelled: set[str] = set()
        self._listeners: list[Listener] = []
        self._lock = threading.RLock()
        self._workers: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._active: str | None = None

        for job in self.store.recent(settings.history_limit):
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                # A job that was in flight when the process died can never resume.
                job.status = JobStatus.FAILED
                job.message = "Interrupted by a restart"
                job.error = {"code": "interrupted", "message": job.message, "hint": None}
                self.store.save(job)
            self._jobs[job.id] = job

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._workers:
            return
        self._stopping.clear()
        for index in range(max(1, self.settings.max_concurrent_jobs)):
            worker = threading.Thread(
                target=self._work, name=f"scriba-worker-{index}", daemon=True
            )
            worker.start()
            self._workers.append(worker)
        log.debug("Started %d worker(s)", len(self._workers))

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        for _ in self._workers:
            self._queue.put("")  # sentinel to wake blocked workers
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers.clear()

    # -- subscriptions -----------------------------------------------------
    def subscribe(self, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _broadcast(self, job: Job) -> None:
        with self._lock:
            listeners = list(self._listeners)
        if not listeners:
            return
        # Listeners get a snapshot: the worker keeps mutating the live Job, and a
        # subscriber must never observe a state that moved on underneath it.
        snapshot = job.model_copy()
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:  # pragma: no cover - a bad listener must not kill a job
                log.exception("Job listener raised")

    # -- submission --------------------------------------------------------
    def submit(self, url: str, options: TranscriptionOptions | None = None) -> Job:
        options = options or TranscriptionOptions()
        if options.output_dir is None:
            options.output_dir = str(self.settings.output_dir)
        job = Job(url=url.strip(), options=options)
        position = self._queue.qsize()
        job.message = "Waiting in queue" if position else "Starting…"

        with self._lock:
            self._jobs[job.id] = job
        self.store.save(job)
        self._queue.put(job.id)
        self.start()
        self._broadcast(job)
        log.info("Queued job %s for %s", job.id, job.url)
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return False
            self._cancelled.add(job_id)
            if job.status is JobStatus.QUEUED:
                # Not started yet: mark it now, the worker will skip it.
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled"
                job.stage = JobStage.QUEUED
                self.store.save(job)
                self._broadcast(job)
        return True

    def retry(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        return self.submit(job.url, job.options.model_copy(deep=True))

    # -- queries -----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job or self.store.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def delete(self, job_id: str) -> bool:
        self.cancel(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
        return self.store.delete(job_id)

    def clear_history(self) -> int:
        with self._lock:
            keep = {jid: job for jid, job in self._jobs.items() if not job.is_terminal}
            self._jobs = keep
        return self.store.clear()

    @property
    def active_job_id(self) -> str | None:
        return self._active

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # -- worker ------------------------------------------------------------
    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not job_id:
                self._queue.task_done()
                continue
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            cancelled = job_id in self._cancelled
        if job is None:
            return
        if cancelled or job.status is JobStatus.CANCELLED:
            job.status = JobStatus.CANCELLED
            job.message = "Cancelled"
            self.store.save(job)
            self._broadcast(job)
            return

        self._active = job_id
        last_stage = job.stage

        def emit(updated: Job) -> None:
            nonlocal last_stage
            # SQLite writes are cheap but not free: persist on stage changes and
            # terminal states, broadcast everything.
            if updated.stage != last_stage or updated.is_terminal:
                last_stage = updated.stage
                self.store.save(updated)
            self._broadcast(updated)

        pipeline = Pipeline(
            self.settings,
            emit=emit,
            cancelled=lambda: job_id in self._cancelled or self._stopping.is_set(),
        )
        try:
            pipeline.run(job)
        except ScribaError as exc:  # pragma: no cover - pipeline handles its own
            job.status = JobStatus.FAILED
            job.error = exc.to_dict()
            job.message = exc.message
        finally:
            self._active = None
            self._cancelled.discard(job_id)
            with self._lock:
                # The user can delete a job during the moment it is finishing;
                # writing here anyway would resurrect it in the history.
                still_tracked = job_id in self._jobs
            if still_tracked:
                self.store.save(job)
                self.store.prune(self.settings.history_limit)
            self._broadcast(job)
