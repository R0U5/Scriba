"""Domain models shared by the pipeline, the CLI and the HTTP API."""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


def new_id() -> str:
    """Short, sortable-ish, URL-safe job id."""
    return uuid.uuid4().hex[:12]


class Task(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"  # speech -> English text (Whisper family only)


class OutputFormat(str, Enum):
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"
    MD = "md"
    TSV = "tsv"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    """Ordered pipeline stages. The UI renders these as a timeline."""

    QUEUED = "queued"
    RESOLVING = "resolving"
    DOWNLOADING = "downloading"
    DECODING = "decoding"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    WRITING = "writing"
    DONE = "done"


STAGE_ORDER: list[JobStage] = [
    JobStage.QUEUED,
    JobStage.RESOLVING,
    JobStage.DOWNLOADING,
    JobStage.DECODING,
    JobStage.LOADING_MODEL,
    JobStage.TRANSCRIBING,
    JobStage.WRITING,
    JobStage.DONE,
]


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class Segment(BaseModel):
    """One utterance-sized chunk of transcript with timings in seconds."""

    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None = None
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class MediaInfo(BaseModel):
    """What we learned about the source media before transcribing."""

    source_url: str
    title: str
    duration: float = 0.0
    uploader: str | None = None
    webpage_url: str | None = None
    extractor: str | None = None
    filesize: int | None = None
    thumbnail: str | None = None


class TranscriptionOptions(BaseModel):
    """Everything the user can tune for a single job."""

    engine: str = "faster-whisper"
    model: str | None = None  # None -> engine default
    language: str | None = None  # None -> autodetect
    task: Task = Task.TRANSCRIBE
    formats: list[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.TXT, OutputFormat.SRT, OutputFormat.JSON]
    )
    output_dir: str | None = None  # None -> settings.output_dir
    device: str | None = None  # None -> auto (cuda if available)
    compute_type: str | None = None  # None -> engine default
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float = 0.0

    def merged_model(self, default: str) -> str:
        return self.model or default


class TranscriptResult(BaseModel):
    """The finished transcript plus everything worth showing about the run."""

    text: str
    segments: list[Segment]
    language: str | None = None
    language_probability: float | None = None
    duration: float = 0.0
    engine: str = ""
    model: str = ""
    device: str = ""
    compute_type: str | None = None
    processing_seconds: float = 0.0
    word_count: int = 0
    #: How trustworthy the segment timings are. ``approximate`` means the engine
    #: reported text without timings and we interpolated them across the window.
    timing_precision: str = "segment"
    #: True while the job is still running: ``segments`` is then a live tail of
    #: what has been recognised so far, not the finished transcript.
    partial: bool = False

    @property
    def speed_factor(self) -> float:
        """Audio seconds transcribed per wall-clock second (higher is better)."""
        if self.processing_seconds <= 0:
            return 0.0
        return self.duration / self.processing_seconds


class Job(BaseModel):
    """A unit of work: one URL in, one set of transcript files out."""

    id: str = Field(default_factory=new_id)
    url: str
    options: TranscriptionOptions = Field(default_factory=TranscriptionOptions)
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.QUEUED
    progress: float = 0.0  # 0..1 within the whole pipeline
    message: str = "Waiting in queue"
    media: MediaInfo | None = None
    result: TranscriptResult | None = None
    outputs: dict[str, str] = Field(default_factory=dict)  # format -> absolute path
    error: dict[str, str | None] | None = None
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def summary(self) -> str:
        title = self.media.title if self.media else self.url
        return f"[{self.id}] {title} ({self.status.value})"
