"""Job history, persisted in SQLite so past transcripts survive restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .logging_setup import get_logger
from .models import Job, JobStatus

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    finished_at REAL,
    status      TEXT NOT NULL,
    url         TEXT NOT NULL,
    title       TEXT,
    engine      TEXT,
    model       TEXT,
    duration    REAL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs (created_at DESC);
"""


class HistoryStore:
    """Thread-safe, dependency-free persistence for :class:`Job` records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- writes ------------------------------------------------------------
    def save(self, job: Job) -> None:
        payload = job.model_dump_json()
        row = (
            job.id,
            job.created_at,
            job.finished_at,
            job.status.value,
            job.url,
            job.media.title if job.media else None,
            job.options.engine,
            job.options.model,
            job.result.duration if job.result else (job.media.duration if job.media else None),
            payload,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (id, created_at, finished_at, status, url, title,
                                  engine, model, duration, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    finished_at=excluded.finished_at,
                    status=excluded.status,
                    title=excluded.title,
                    engine=excluded.engine,
                    model=excluded.model,
                    duration=excluded.duration,
                    payload=excluded.payload
                """,
                row,
            )
            self._conn.commit()

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM jobs")
            self._conn.commit()
            return cursor.rowcount

    def prune(self, keep: int) -> int:
        """Drop the oldest rows beyond ``keep``."""
        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM jobs WHERE id IN (
                    SELECT id FROM jobs ORDER BY created_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (keep,),
            )
            self._conn.commit()
            return cursor.rowcount

    # -- reads -------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _decode(row["payload"]) if row else None

    def recent(self, limit: int = 50, *, status: JobStatus | None = None) -> list[Job]:
        query = "SELECT payload FROM jobs"
        params: list = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [job for job in (_decode(row["payload"]) for row in rows) if job is not None]

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS jobs,
                       COALESCE(SUM(duration), 0) AS audio_seconds,
                       COALESCE(SUM(status = 'completed'), 0) AS completed
                FROM jobs
                """
            ).fetchone()
        return {
            "jobs": int(row["jobs"]),
            "completed": int(row["completed"]),
            "audio_seconds": float(row["audio_seconds"]),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _decode(payload: str) -> Job | None:
    try:
        return Job.model_validate(json.loads(payload))
    except Exception as exc:  # pragma: no cover - schema drift between versions
        log.warning("Skipping unreadable history row: %s", exc)
        return None
