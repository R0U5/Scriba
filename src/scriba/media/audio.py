"""Decoding to the one audio format every ASR model wants: 16 kHz mono PCM."""

from __future__ import annotations

import re
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import MediaError
from ..logging_setup import get_logger
from .ffmpeg import ffmpeg_path, ffprobe_path

log = get_logger(__name__)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit PCM
ProgressCB = Callable[[float, str], None]


@dataclass(slots=True)
class AudioFile:
    """A decoded 16 kHz mono WAV on disk."""

    path: Path
    duration: float
    sample_rate: int = SAMPLE_RATE

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


def probe_duration(path: Path) -> float | None:
    """Best-effort media duration in seconds (``None`` when unknown)."""
    probe = ffprobe_path()
    if not probe:
        return None
    try:
        out = subprocess.run(
            [
                probe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=120, check=False,
        )
        value = (out.stdout or "").strip()
        return float(value) if value and value != "N/A" else None
    except Exception:  # pragma: no cover - probing is advisory only
        return None


_TIME_RE = re.compile(r"out_time_ms=(\d+)")


def decode_to_wav(
    source: Path,
    destination: Path,
    *,
    hint_duration: float | None = None,
    progress: ProgressCB | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AudioFile:
    """Decode any media file to 16 kHz mono 16-bit WAV.

    Progress is reported as a 0..1 fraction using ffmpeg's ``-progress`` stream
    when the total duration is known.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = hint_duration or probe_duration(source) or 0.0

    cmd = [
        ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", "-f", "wav",
        "-progress", "pipe:1", "-nostats",
        str(destination),
    ]
    log.debug("decoding: %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if cancelled is not None and cancelled():
                process.kill()
                raise MediaError("Decoding cancelled")
            match = _TIME_RE.search(line)
            if match and progress:
                done = int(match.group(1)) / 1_000_000.0
                if total > 0:
                    progress(min(done / total, 1.0), f"Decoding audio · {done:.0f}s / {total:.0f}s")
                else:  # unknown length: report activity, not a fake percentage
                    progress(0.0, f"Decoding audio · {done:.0f}s")
        process.wait(timeout=60)
    finally:
        if process.poll() is None:  # pragma: no cover - defensive
            process.kill()

    if process.returncode != 0:
        stderr = (process.stderr.read() if process.stderr else "") or ""
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown ffmpeg error"
        raise MediaError(
            f"ffmpeg could not decode the audio: {detail}",
            hint="The source may be DRM-protected, video-only, or a corrupted download.",
        )

    duration = wav_duration(destination)
    if duration <= 0:
        raise MediaError(
            "The source contains no audio.",
            hint="Check that the URL points at a media file with an audio track.",
        )
    if progress:
        progress(1.0, f"Decoded {duration:.0f}s of audio")
    return AudioFile(path=destination, duration=duration)


def wav_duration(path: Path) -> float:
    """Exact duration of a PCM WAV, computed from its frame count."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or SAMPLE_RATE
            return frames / float(rate)
    except (wave.Error, OSError) as exc:
        raise MediaError(f"Not a readable WAV file: {path.name} ({exc})") from exc


def read_wav(path: Path, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Read a mono 16-bit WAV (optionally a time slice) as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if width != SAMPLE_WIDTH:
            raise MediaError(f"Expected 16-bit PCM, got {width * 8}-bit: {path.name}")
        total_frames = handle.getnframes()

        first = max(0, int(start * rate))
        last = total_frames if end is None else min(total_frames, int(end * rate))
        count = max(0, last - first)
        handle.setpos(min(first, total_frames))
        raw = handle.readframes(count)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:  # pragma: no cover - we always decode to mono
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write float32 samples in [-1, 1] as a 16-bit mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silence(
    path: Path, *, noise_db: float = -35.0, min_duration: float = 0.35
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` silence intervals via ffmpeg's ``silencedetect``."""
    cmd = [
        ffmpeg_path(), "-hide_banner", "-nostdin", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=1800)
    except Exception as exc:  # pragma: no cover - advisory
        log.debug("silencedetect failed: %s", exc)
        return []

    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for line in (out.stderr or "").splitlines():
        if (m := _SILENCE_START.search(line)) is not None:
            pending = float(m.group(1))
        elif (m := _SILENCE_END.search(line)) is not None and pending is not None:
            silences.append((pending, float(m.group(1))))
            pending = None
    return silences


def plan_chunks(
    duration: float,
    *,
    target: float = 600.0,
    tolerance: float = 45.0,
    silences: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Split ``duration`` into windows of about ``target`` seconds.

    When silence intervals are supplied, each cut is nudged (within
    ``tolerance``) onto the middle of the nearest pause, so chunk boundaries
    never land mid-word — the classic failure mode of naive fixed-size chunking.
    """
    if duration <= target:
        return [(0.0, duration)]

    silences = silences or []
    cuts: list[float] = []
    position = target
    while position < duration - 1.0:
        best = position
        best_delta = tolerance + 1.0
        for start, end in silences:
            middle = (start + end) / 2.0
            delta = abs(middle - position)
            if delta < best_delta:
                best, best_delta = middle, delta
        if cuts and best <= cuts[-1] + 1.0:
            best = position  # never go backwards
        cuts.append(round(best, 3))
        position = best + target

    bounds = [0.0, *cuts, duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
