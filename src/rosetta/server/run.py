"""Launching uvicorn with the niceties expected of a desktop-grade tool."""

from __future__ import annotations

import threading
import webbrowser

import uvicorn

from ..config import Settings
from ..logging_setup import get_logger

log = get_logger(__name__)


def run_server(settings: Settings, *, reload: bool = False) -> None:
    url = f"http://{'localhost' if settings.host in ('0.0.0.0', '127.0.0.1') else settings.host}:{settings.port}"

    if settings.open_browser and not reload:
        threading.Timer(1.2, lambda: _open(url)).start()

    if reload:  # pragma: no cover - developer convenience
        uvicorn.run(
            "rosetta.server.app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
        return

    from .app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


def _open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - headless machines
        log.debug("Could not open a browser automatically")
