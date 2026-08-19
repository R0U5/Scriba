"""Typed errors.

Every error carries a human-readable ``message`` (shown verbatim in the UI) and
an optional ``hint`` with the concrete next step the user can take. The web UI
renders the hint underneath the message, so keep hints actionable.
"""

from __future__ import annotations


class RosettaError(Exception):
    """Base class for every error RosettaAI raises deliberately."""

    #: Short machine-readable code, surfaced in the API payload.
    code = "error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class ConfigError(RosettaError):
    code = "config_error"


class InvalidURLError(RosettaError):
    code = "invalid_url"


class DownloadError(RosettaError):
    code = "download_failed"


class MediaError(RosettaError):
    """ffmpeg/ffprobe missing, or the media could not be decoded."""

    code = "media_error"


class EngineNotAvailableError(RosettaError):
    """The requested ASR engine is not installed or its hardware is missing."""

    code = "engine_unavailable"


class TranscriptionError(RosettaError):
    code = "transcription_failed"


class JobCancelled(RosettaError):
    """Raised inside a worker when the user cancels a running job."""

    code = "cancelled"

    def __init__(self, message: str = "Job cancelled") -> None:
        super().__init__(message)
