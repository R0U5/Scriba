"""Rendering a transcript into the files people actually use."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .models import MediaInfo, OutputFormat, Segment, TranscriptionOptions, TranscriptResult
from .util.text import format_duration, format_timestamp, paragraphs, unique_path, wrap_caption

#: Subtitle cues longer than this get split at word boundaries when possible.
MAX_CUE_SECONDS = 7.0
MAX_CUE_CHARS = 84


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #
def render_txt(result: TranscriptResult, media: MediaInfo | None = None) -> str:
    body = "\n\n".join(paragraphs([s.text for s in result.segments])) or result.text
    return body.strip() + "\n"


def render_md(result: TranscriptResult, media: MediaInfo | None = None) -> str:
    title = (media.title if media else None) or "Transcript"
    lines = [f"# {title}", ""]

    facts: list[str] = []
    if media and media.uploader:
        facts.append(f"- **Source:** {media.uploader}")
    if media and (media.webpage_url or media.source_url):
        url = media.webpage_url or media.source_url
        facts.append(f"- **URL:** <{url}>")
    facts.append(f"- **Duration:** {format_duration(result.duration)}")
    if result.language:
        facts.append(f"- **Language:** {result.language}")
    facts.append(f"- **Model:** {result.model} ({result.engine})")
    facts.append(f"- **Words:** {result.word_count:,}")
    facts.append(f"- **Transcribed:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}")
    lines.extend(facts)
    lines.extend(["", "---", ""])

    # Timestamped paragraphs: each block is anchored to the segment it starts at.
    block: list[Segment] = []
    length = 0
    for segment in result.segments:
        block.append(segment)
        length += len(segment.text) + 1
        if length >= 700 and segment.text.rstrip()[-1:] in ".!?…":
            lines.append(_md_block(block))
            lines.append("")
            block, length = [], 0
    if block:
        lines.append(_md_block(block))
    return "\n".join(lines).strip() + "\n"


def _md_block(block: list[Segment]) -> str:
    stamp = format_timestamp(block[0].start, separator=".")[:-4]  # HH:MM:SS
    text = " ".join(s.text for s in block)
    return f"**[{stamp}]** {text}"


# --------------------------------------------------------------------------- #
# Subtitles
# --------------------------------------------------------------------------- #
def _cues(segments: list[Segment]) -> list[Segment]:
    """Split over-long segments into subtitle-sized cues."""
    cues: list[Segment] = []
    for segment in segments:
        too_long = segment.duration > MAX_CUE_SECONDS or len(segment.text) > MAX_CUE_CHARS
        if not too_long:
            # Copy: renumbering below must not touch the caller's segments,
            # which are also serialised into the JSON output.
            cues.append(segment.model_copy())
            continue
        if segment.words:
            cues.extend(_split_by_words(segment, len(cues)))
        else:
            cues.extend(_split_evenly(segment, len(cues)))
    for index, cue in enumerate(cues):
        cue.id = index
    return cues


def _split_by_words(segment: Segment, first_id: int) -> list[Segment]:
    parts: list[Segment] = []
    current: list = []
    for word in segment.words or []:
        current.append(word)
        text = "".join(w.word for w in current).strip()
        span = current[-1].end - current[0].start
        if len(text) >= MAX_CUE_CHARS or span >= MAX_CUE_SECONDS:
            parts.append(
                Segment(id=first_id + len(parts), start=current[0].start,
                        end=current[-1].end, text=text)
            )
            current = []
    if current:
        text = "".join(w.word for w in current).strip()
        parts.append(
            Segment(id=first_id + len(parts), start=current[0].start,
                    end=max(current[-1].end, segment.end), text=text)
        )
    return parts or [segment]


def _split_evenly(segment: Segment, first_id: int) -> list[Segment]:
    words = segment.text.split()
    if not words:
        return [segment]
    chunks: list[list[str]] = [[]]
    for word in words:
        candidate = " ".join([*chunks[-1], word])
        if len(candidate) > MAX_CUE_CHARS and chunks[-1]:
            chunks.append([word])
        else:
            chunks[-1].append(word)

    total_chars = sum(len(" ".join(c)) for c in chunks) or 1
    parts: list[Segment] = []
    cursor = segment.start
    for index, chunk in enumerate(chunks):
        text = " ".join(chunk)
        span = segment.duration * (len(text) / total_chars)
        end = segment.end if index == len(chunks) - 1 else min(cursor + span, segment.end)
        parts.append(Segment(id=first_id + index, start=cursor, end=end, text=text))
        cursor = end
    return parts


def render_srt(result: TranscriptResult, media: MediaInfo | None = None) -> str:
    lines: list[str] = []
    for index, cue in enumerate(_cues(result.segments), start=1):
        speaker = f"[{cue.speaker}] " if cue.speaker else ""
        lines.append(str(index))
        lines.append(
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}"
        )
        lines.append(wrap_caption(speaker + cue.text))
        lines.append("")
    return "\n".join(lines)


def render_vtt(result: TranscriptResult, media: MediaInfo | None = None) -> str:
    header = ["WEBVTT", ""]
    if media and media.title:
        header = ["WEBVTT", f"NOTE {media.title}", ""]
    lines = list(header)
    for cue in _cues(result.segments):
        speaker = f"<v {cue.speaker}>" if cue.speaker else ""
        lines.append(
            f"{format_timestamp(cue.start, separator='.')} --> "
            f"{format_timestamp(cue.end, separator='.')}"
        )
        lines.append(speaker + wrap_caption(cue.text))
        lines.append("")
    return "\n".join(lines)


def render_tsv(result: TranscriptResult, media: MediaInfo | None = None) -> str:
    rows = ["start\tend\ttext"]
    rows.extend(
        f"{int(s.start * 1000)}\t{int(s.end * 1000)}\t" + s.text.replace("\t", " ")
        for s in result.segments
    )
    return "\n".join(rows) + "\n"


def render_json(
    result: TranscriptResult,
    media: MediaInfo | None = None,
    options: TranscriptionOptions | None = None,
) -> str:
    payload = {
        "rosetta_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "media": media.model_dump() if media else None,
        "options": options.model_dump(mode="json") if options else None,
        "transcript": {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": round(result.duration, 3),
            "word_count": result.word_count,
            "timing_precision": result.timing_precision,
        },
        "run": {
            "engine": result.engine,
            "model": result.model,
            "device": result.device,
            "compute_type": result.compute_type,
            "processing_seconds": round(result.processing_seconds, 2),
            "speed_factor": round(result.speed_factor, 1),
        },
        "segments": [
            {
                "id": s.id,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                **({"speaker": s.speaker} if s.speaker else {}),
                **(
                    {
                        "words": [
                            {
                                "start": round(w.start, 3),
                                "end": round(w.end, 3),
                                "word": w.word,
                                **({"probability": round(w.probability, 4)}
                                   if w.probability is not None else {}),
                            }
                            for w in s.words
                        ]
                    }
                    if s.words
                    else {}
                ),
            }
            for s in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


RENDERERS = {
    OutputFormat.TXT: render_txt,
    OutputFormat.MD: render_md,
    OutputFormat.SRT: render_srt,
    OutputFormat.VTT: render_vtt,
    OutputFormat.TSV: render_tsv,
}


def render(
    fmt: OutputFormat,
    result: TranscriptResult,
    media: MediaInfo | None = None,
    options: TranscriptionOptions | None = None,
) -> str:
    if fmt is OutputFormat.JSON:
        return render_json(result, media, options)
    return RENDERERS[fmt](result, media)


def write_outputs(
    result: TranscriptResult,
    media: MediaInfo | None,
    options: TranscriptionOptions,
    output_dir: Path,
    stem: str,
) -> dict[str, str]:
    """Write every requested format next to each other, sharing one stem."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = list(dict.fromkeys(options.formats)) or [OutputFormat.TXT]

    # Reserve the stem once so all formats of one run stay grouped together.
    primary = unique_path(output_dir, stem, formats[0].value)
    final_stem = primary.stem

    written: dict[str, str] = {}
    for fmt in formats:
        path = output_dir / f"{final_stem}.{fmt.value}"
        path.write_text(render(fmt, result, media, options), encoding="utf-8")
        written[fmt.value] = str(path)
    return written
