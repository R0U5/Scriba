"""NVIDIA Parakeet TDT — the throughput champion of open ASR.

A 0.6B FastConformer/TDT model that transcribes with punctuation, casing and
word-level timestamps at thousands of times realtime on a consumer GPU, while
scoring better than Whisper large-v3 on the Open ASR Leaderboard. v3 adds
automatic language identification across 25 European languages.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..errors import JobCancelled, TranscriptionError
from ..logging_setup import get_logger
from ..media.audio import AudioFile
from ..models import Segment, TranscriptionOptions, TranscriptResult, Word
from ..util.text import word_count
from ._nemo import (
    chunked_wavs,
    free_cuda_memory,
    inference_mode,
    nemo_availability,
    torch_autocast,
    wrap_oom,
)
from .base import (
    ASREngine,
    Availability,
    CancelCB,
    EngineSpec,
    ModelSpec,
    ProgressCB,
    SegmentCB,
)

log = get_logger(__name__)

# Windows fed to the model. FastConformer attention is quadratic in time, so
# 5-minute windows keep peak VRAM comfortable on a 12 GB card.
CHUNK_SECONDS = 300.0

SPEC = EngineSpec(
    key="parakeet",
    label="NVIDIA Parakeet TDT",
    tagline="Beats Whisper large-v3 on accuracy and runs an order of magnitude faster.",
    vendor="NVIDIA",
    license="CC-BY-4.0 (commercial use allowed)",
    install_extra="nvidia",
    install_hint=(
        'pip install "rosetta-ai[nvidia]"  # NVIDIA NeMo toolkit + CUDA PyTorch'
    ),
    supports_translate=False,
    supports_word_timestamps=True,
    models=[
        ModelSpec(
            id="nvidia/parakeet-tdt-0.6b-v3",
            label="Parakeet TDT 0.6B v3",
            parameters="0.6B",
            vram_gb=3.0,
            languages="25 European languages + auto language ID",
            wer=6.32,
            speed="~1000x+ realtime on an RTX 4070",
            note="Multilingual successor to v2. Punctuation, casing, word timestamps.",
            default=True,
        ),
        ModelSpec(
            id="nvidia/parakeet-tdt-0.6b-v2",
            label="Parakeet TDT 0.6B v2",
            parameters="0.6B",
            vram_gb=3.0,
            languages="English only",
            wer=6.05,
            speed="~1000x+ realtime on an RTX 4070",
            note="English specialist; marginally more accurate than v3 on English.",
        ),
    ],
)


class ParakeetEngine(ASREngine):
    spec = SPEC

    def __init__(self, model=None, device=None, compute_type=None) -> None:
        super().__init__(model, device, compute_type)
        self._model = None

    @classmethod
    def availability(cls) -> Availability:
        return nemo_availability()

    def default_compute_type(self, device: str) -> str:
        return "bfloat16" if device == "cuda" else "float32"

    # -- lifecycle ---------------------------------------------------------
    def load(self, progress: ProgressCB | None = None) -> None:
        if self._loaded:
            return
        self.ensure_available()
        if progress:
            progress(0.0, f"Loading {self.model_id} on {self.device.upper()}…")

        started = time.monotonic()
        try:
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
            model = model.to(self.device)
            model.eval()
        except Exception as exc:
            raise wrap_oom(exc, self.model_id) from exc

        self._model = model
        self._loaded = True
        elapsed = time.monotonic() - started
        log.info("Loaded %s on %s in %.1fs", self.model_id, self.device, elapsed)
        if progress:
            progress(1.0, f"Model ready ({elapsed:.1f}s)")

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        free_cuda_memory()

    # -- transcription -----------------------------------------------------
    def transcribe(
        self,
        audio: AudioFile,
        options: TranscriptionOptions,
        *,
        progress: ProgressCB | None = None,
        cancelled: CancelCB | None = None,
        on_segment: SegmentCB | None = None,
    ) -> TranscriptResult:
        self.load()
        assert self._model is not None

        started = time.monotonic()
        segments: list[Segment] = []
        workdir = audio.path.parent / "chunks"

        with chunked_wavs(audio, target=CHUNK_SECONDS, workdir=workdir) as chunks:
            for chunk in chunks:
                if cancelled is not None and cancelled():
                    raise JobCancelled()
                hypothesis = self._run(chunk.path, options.word_timestamps)
                fresh = _to_segments(
                    hypothesis, offset=chunk.start, first_id=len(segments),
                    fallback=(chunk.start, chunk.end), want_words=options.word_timestamps,
                )
                segments.extend(fresh)
                if on_segment:
                    for segment in fresh:
                        on_segment(segment)
                if progress:
                    done = min(chunk.end / max(audio.duration, 1e-6), 0.999)
                    progress(done, f"Transcribing · chunk {chunk.index + 1}/{len(chunks)}")

        elapsed = time.monotonic() - started
        text = " ".join(s.text for s in segments).strip()
        if progress:
            progress(1.0, f"Transcribed {len(segments)} segments")

        return TranscriptResult(
            text=text,
            segments=segments,
            language=options.language,
            duration=audio.duration,
            engine=self.spec.key,
            model=self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            processing_seconds=elapsed,
            word_count=word_count(text),
        )

    def _run(self, path: Path, want_words: bool):
        try:
            with inference_mode(), torch_autocast(self.device, self.compute_type):
                output = self._model.transcribe(
                    [str(path)], timestamps=True, batch_size=1, verbose=False
                )
        except TypeError:  # older NeMo builds do not accept ``verbose``
            with inference_mode(), torch_autocast(self.device, self.compute_type):
                output = self._model.transcribe([str(path)], timestamps=True, batch_size=1)
        except Exception as exc:
            raise wrap_oom(exc, self.model_id) from exc

        if not output:
            raise TranscriptionError("Parakeet returned no output for this audio.")
        first = output[0]
        return first[0] if isinstance(first, list) and first else first


def _to_segments(
    hypothesis,
    *,
    offset: float,
    first_id: int,
    fallback: tuple[float, float],
    want_words: bool,
) -> list[Segment]:
    """Normalise a NeMo hypothesis into our :class:`Segment` list."""
    text = (getattr(hypothesis, "text", None) or str(hypothesis)).strip()
    stamps = getattr(hypothesis, "timestamp", None) or {}
    entries = stamps.get("segment") if isinstance(stamps, dict) else None

    if not entries:
        return [Segment(id=first_id, start=fallback[0], end=fallback[1], text=text)] if text else []

    words = stamps.get("word") if (want_words and isinstance(stamps, dict)) else None
    segments: list[Segment] = []
    for index, entry in enumerate(entries):
        body = (entry.get("segment") or entry.get("text") or "").strip()
        if not body:
            continue
        start = float(entry.get("start", 0.0)) + offset
        end = float(entry.get("end", start)) + offset
        segments.append(
            Segment(
                id=first_id + index,
                start=start,
                end=end,
                text=body,
                words=_words_between(words, entry, offset) if words else None,
            )
        )
    return segments


def _words_between(words, entry, offset: float) -> list[Word] | None:
    start = float(entry.get("start", 0.0))
    end = float(entry.get("end", start))
    collected = [
        Word(
            start=float(w.get("start", 0.0)) + offset,
            end=float(w.get("end", 0.0)) + offset,
            word=(w.get("word") or w.get("text") or "").strip(),
        )
        for w in words
        if start - 1e-3 <= float(w.get("start", 0.0)) <= end + 1e-3
    ]
    return collected or None
