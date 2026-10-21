"""Scriba — local‑first, production‑grade audio transcription.

Provides a CLI (`scriba`) and a programmatic API for downloading media, running
local ASR models, and saving transcripts in multiple formats.

Only the version is exported by default; the public API is deliberately minimal.
Consumers typically import ``scriba.cli`` or ``scriba.pipeline.transcribe_url``.
"""

from importlib import import_module

__version__: str = "1.0.0"

# Export the most useful entry points for ``import scriba`` users.
__all__ = [
    "__version__",
    "cli",
    "pipeline",
]


def __getattr__(name: str):  # pragma: no cover – exercised indirectly via tests
    if name == "cli":
        return import_module("scriba.cli")
    if name == "pipeline":
        return import_module("scriba.pipeline")
    raise AttributeError(name)
