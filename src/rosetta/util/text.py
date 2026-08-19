"""Formatting helpers for timestamps, filenames and readable transcripts."""

from __future__ import annotations

import re
import unicodedata

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?…][\"')\]]?$")


def format_timestamp(seconds: float, *, separator: str = ",", always_hours: bool = True) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    seconds = max(0.0, float(seconds))
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    if hours or always_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"
    return f"{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def format_duration(seconds: float) -> str:
    """Compact human duration: ``4m 12s`` / ``1h 03m``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def slugify(value: str, *, max_length: int = 80, fallback: str = "transcript") -> str:
    """Filesystem-safe slug that still reads like the original title."""
    value = unicodedata.normalize("NFKC", value or "").strip()
    value = _UNSAFE_CHARS.sub(" ", value)
    value = value.replace("'", "").replace("’", "")
    value = _WHITESPACE.sub(" ", value).strip(" .")
    value = value.replace(" ", "-")
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    if len(value) > max_length:
        value = value[:max_length].rstrip("-._")
    return value or fallback


def unique_path(directory, stem: str, suffix: str):
    """Return ``directory/stem.suffix``, adding ``-2``, ``-3``… if taken."""
    from pathlib import Path

    directory = Path(directory)
    candidate = directory / f"{stem}.{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}.{suffix}"
        counter += 1
    return candidate


def paragraphs(texts: list[str], *, max_chars: int = 700) -> list[str]:
    """Group segment texts into readable paragraphs.

    Segments are merged until a sentence ends and the paragraph is long enough,
    which turns ASR's choppy 5-second segments into prose a human wants to read.
    """
    out: list[str] = []
    buffer: list[str] = []
    length = 0
    for raw in texts:
        text = raw.strip()
        if not text:
            continue
        buffer.append(text)
        length += len(text) + 1
        if length >= max_chars and _SENTENCE_END.search(text):
            out.append(" ".join(buffer))
            buffer, length = [], 0
    if buffer:
        out.append(" ".join(buffer))
    return out


def wrap_caption(text: str, *, max_line: int = 42, max_lines: int = 2) -> str:
    """Wrap a caption cue to at most ``max_lines`` lines of ``max_line`` chars."""
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_line or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # Too long for the cue: rebalance across the allowed lines instead of cutting.
    target = max(1, len(text) // max_lines + 1)
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= target or not current or len(lines) == max_lines - 1:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len([w for w in _WHITESPACE.split(text.strip()) if w])
