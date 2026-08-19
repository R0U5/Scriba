"""NVIDIA Canary-Qwen 2.5B — the accuracy leader on the Open ASR Leaderboard.

A speech-augmented language model: a FastConformer encoder feeding an unmodified
Qwen3-1.7B decoder. It holds the top English WER (5.63%) among open models and
fits in roughly 6 GB of VRAM in bfloat16, so it runs on a 12 GB card.

The decoder is an LLM, so it emits text without timings. Long audio is split on
pauses and the resulting sentences are timed by interpolation within each
window — good enough to navigate a transcript, but use Whisper or Parakeet when
you need frame-accurate subtitles.
"""

from __future__ import annotations

import re
import time

from ..errors import JobCancelled, TranscriptionError
from ..logging_setup import get_logger
from ..media.audio import AudioFile
from ..models import Segment, TranscriptionOptions, TranscriptResult
from ..util.text import word_count
from ._nemo import (
    Chunk,
    chunked_wavs,
    free_cuda_memory,
    inference_mode,
    nemo_availability,
    suggested_batch,
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
    detect_gpu,
)

log = get_logger(__name__)

# Canary-Qwen was trained on utterances up to ~40s; stay just under that.
CHUNK_SECONDS = 35.0
PROMPT = "Transcribe the following:"

SPEC = EngineSpec(
    key="canary-qwen",
    label="NVIDIA Canary-Qwen 2.5B",
    tagline="Highest open-model accuracy on English (5.63% WER). Needs a CUDA GPU.",
    vendor="NVIDIA",
    license="CC-BY-4.0 (commercial use allowed)",
    install_extra="nvidia",
    install_hint='pip install "rosetta-ai[nvidia]"  # NVIDIA NeMo toolkit + CUDA PyTorch',
    supports_translate=False,
    supports_word_timestamps=False,
    requires_gpu=True,
    models=[
        ModelSpec(
            id="nvidia/canary-qwen-2.5b",
            label="Canary-Qwen 2.5B",
            parameters="2.5B",
            vram_gb=6.5,
            languages="English only",
            wer=5.63,
            speed="~50-100x realtime on an RTX 4070",
            note="Top of the Open ASR Leaderboard. Timings are interpolated, not exact.",
            default=True,
        )
    ],
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[]?[A-Z0-9])")


class CanaryQwenEngine(ASREngine):
    spec = SPEC

    def __init__(self, model=None, device=None, compute_type=None) -> None:
        super().__init__(model, device, compute_type)
        self._model = None

    @classmethod
    def availability(cls) -> Availability:
        return nemo_availability(needs_gpu=True)

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
            from nemo.collections.speechlm2.models import SALM

            model = SALM.from_pretrained(self.model_id)
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
        batch_size = suggested_batch(detect_gpu().total_vram_gb, small=1, medium=4, large=8)
        workdir = audio.path.parent / "chunks"

        with chunked_wavs(audio, target=CHUNK_SECONDS, workdir=workdir) as chunks:
            for offset in range(0, len(chunks), batch_size):
                if cancelled is not None and cancelled():
                    raise JobCancelled()
                batch = chunks[offset : offset + batch_size]
                for chunk, text in zip(batch, self._generate(batch), strict=False):
                    if not text:
                        continue
                    fresh = _interpolate(text, chunk, first_id=len(segments))
                    segments.extend(fresh)
                    if on_segment:
                        for segment in fresh:
                            on_segment(segment)
                if progress:
                    done = min(batch[-1].end / max(audio.duration, 1e-6), 0.999)
                    progress(
                        done,
                        f"Transcribing · chunk {min(offset + batch_size, len(chunks))}/{len(chunks)}",
                    )

        for index, segment in enumerate(segments):  # ids must be contiguous
            segment.id = index

        elapsed = time.monotonic() - started
        text = " ".join(s.text for s in segments).strip()
        if progress:
            progress(1.0, f"Transcribed {len(segments)} segments")

        return TranscriptResult(
            text=text,
            segments=segments,
            language="en",
            duration=audio.duration,
            engine=self.spec.key,
            model=self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            processing_seconds=elapsed,
            word_count=word_count(text),
            timing_precision="approximate",
        )

    def _generate(self, batch: list[Chunk]) -> list[str]:
        model = self._model
        prompts = [
            [
                {
                    "role": "user",
                    "content": f"{PROMPT} {model.audio_locator_tag}",
                    "audio": [str(chunk.path)],
                }
            ]
            for chunk in batch
        ]
        # ~2.5 tokens per spoken word, plus headroom for punctuation.
        max_new_tokens = int(max(chunk.duration for chunk in batch) * 12) + 32
        try:
            with inference_mode(), torch_autocast(self.device, self.compute_type):
                answers = model.generate(prompts=prompts, max_new_tokens=max_new_tokens)
        except Exception as exc:
            raise wrap_oom(exc, self.model_id) from exc

        texts: list[str] = []
        for row in answers:
            # NeMo returns CUDA tensors; tolerate plain sequences too.
            ids = row.cpu() if hasattr(row, "cpu") else row
            try:
                texts.append(model.tokenizer.ids_to_text(ids).strip())
            except Exception as exc:  # pragma: no cover - tokenizer contract change
                raise TranscriptionError(f"Could not decode Canary output: {exc}") from exc
        return texts


def _interpolate(text: str, chunk: Chunk, *, first_id: int) -> list[Segment]:
    """Split a chunk's text into sentences timed proportionally to their length."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    total_chars = sum(len(s) for s in sentences) or 1
    segments: list[Segment] = []
    cursor = chunk.start
    for index, sentence in enumerate(sentences):
        span = chunk.duration * (len(sentence) / total_chars)
        end = chunk.end if index == len(sentences) - 1 else min(cursor + span, chunk.end)
        segments.append(
            Segment(id=first_id + index, start=round(cursor, 3), end=round(end, 3), text=sentence)
        )
        cursor = end
    return segments
