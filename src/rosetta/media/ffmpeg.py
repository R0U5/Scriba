"""Locating and driving ffmpeg.

We prefer a system ffmpeg (fast, feature-complete) and transparently fall back
to the static binary shipped by ``imageio-ffmpeg`` so a fresh ``pip install``
works even on a machine with nothing installed.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

from ..errors import MediaError
from ..logging_setup import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """Absolute path to an ffmpeg executable.

    Raises :class:`MediaError` with an install hint when nothing is available.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            log.debug("Using bundled ffmpeg at %s", bundled)
            return bundled
    except Exception:  # pragma: no cover - optional dependency
        pass
    raise MediaError(
        "ffmpeg was not found on this system.",
        hint=(
            "Install it with 'winget install Gyan.FFmpeg' (Windows), "
            "'brew install ffmpeg' (macOS) or 'sudo apt install ffmpeg' (Linux), "
            "or run 'pip install imageio-ffmpeg' for a bundled build."
        ),
    )


@functools.lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    """Path to ffprobe if present. Optional: duration is derived from PCM size."""
    return shutil.which("ffprobe")


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            [ffmpeg_path(), "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        first = (out.stdout or "").splitlines()[0] if out.stdout else ""
        return first.replace("ffmpeg version ", "").split(" Copyright")[0].strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def ffmpeg_available() -> bool:
    try:
        ffmpeg_path()
        return True
    except MediaError:
        return False
