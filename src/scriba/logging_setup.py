"""Console logging with Rich, shared by the CLI and the server."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str = "INFO", *, quiet: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger("scriba").setLevel(level.upper())
        return

    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=not quiet,
        markup=False,
        log_time_format="[%H:%M:%S]",
    )
    logging.basicConfig(level=level.upper(), format="%(message)s", handlers=[handler])

    # Third-party libraries are chatty; keep the console readable.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "numba", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str = "scriba") -> logging.Logger:
    return logging.getLogger(name)
