"""Launching uvicorn with the niceties expected of a desktop-grade tool."""

from __future__ import annotations

import socket
import threading
import time
import webbrowser

import uvicorn

from ..config import Settings
from ..logging_setup import get_logger

log = get_logger(__name__)

#: How many ports to try after the configured one before giving up.
PORT_SEARCH_RANGE = 20
#: How long to wait for the server to accept connections before opening a browser.
READY_TIMEOUT = 30.0


def port_is_free(host: str, port: int) -> bool:
    """True when nothing is listening on ``host:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host if host != "0.0.0.0" else "", port))
            return True
        except OSError:
            return False


def find_free_port(host: str, preferred: int) -> int:
    """Return ``preferred`` if it is free, else the next free port after it.

    Starting a second copy, or leaving a stale one running, should not end in a
    traceback about address reuse.
    """
    for candidate in range(preferred, preferred + PORT_SEARCH_RANGE):
        if port_is_free(host, candidate):
            return candidate
    raise OSError(
        f"Ports {preferred}-{preferred + PORT_SEARCH_RANGE - 1} are all in use on {host}."
    )


def wait_until_ready(host: str, port: int, timeout: float = READY_TIMEOUT) -> bool:
    """Block until the server accepts a connection, so the browser opens to a page."""
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((target, port)) == 0:
                return True
        time.sleep(0.15)
    return False


def open_when_ready(host: str, port: int, url: str) -> None:
    if not wait_until_ready(host, port):
        log.debug("Server was not ready in time; not opening a browser")
        return
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - headless machines
        log.debug("Could not open a browser automatically")


def run_server(settings: Settings, *, reload: bool = False) -> None:
    host_label = "localhost" if settings.host in ("0.0.0.0", "127.0.0.1") else settings.host
    url = f"http://{host_label}:{settings.port}"

    if settings.open_browser and not reload:
        # Waiting for readiness beats a fixed delay: a cold start on Windows can
        # take several seconds, and an early browser lands on a connection error.
        threading.Thread(
            target=open_when_ready, args=(settings.host, settings.port, url), daemon=True
        ).start()

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
