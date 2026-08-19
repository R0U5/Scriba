"""Shared plumbing for the NVIDIA NeMo engines (Parakeet, Canary-Qwen).

NeMo models are trained on short-form audio, so long recordings are split into
silence-aligned windows before inference and stitched back together afterwards.
Cutting on pauses (rather than on a fixed grid) keeps words intact, which is
what separates a usable long-form transcript from a mangled one.
"""

from __future__ import annotations

import contextlib
import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..errors import TranscriptionError
from ..logging_setup import get_logger
from ..media.audio import AudioFile, detect_silence, plan_chunks, read_wav, write_wav
from .base import Availability, detect_gpu

log = get_logger(__name__)

NEMO_INSTALL_HINT = (
    'Install the NVIDIA engines with: pip install "rosetta-ai[nvidia]" '
    "(or: pip install nemo_toolkit[asr])"
)


def nemo_availability(*, needs_gpu: bool = False) -> Availability:
    """Is NeMo importable, and is the hardware adequate?"""
    if importlib.util.find_spec("nemo") is None:
        return Availability(False, "NVIDIA NeMo is not installed.", NEMO_INSTALL_HINT)
    if importlib.util.find_spec("torch") is None:  # pragma: no cover
        return Availability(False, "PyTorch is not installed.", NEMO_INSTALL_HINT)
    if needs_gpu and not detect_gpu().available:
        return Availability(
            False,
            "This model needs an NVIDIA GPU; no CUDA device was detected.",
            "Use the Whisper engine on CPU-only machines.",
        )
    return Availability(True)


@dataclass(slots=True)
class Chunk:
    """A silence-aligned window of audio written to a temporary WAV."""

    index: int
    start: float
    end: float
    path: Path

    @property
    def duration(self) -> float:
        return self.end - self.start


@contextlib.contextmanager
def chunked_wavs(
    audio: AudioFile, *, target: float, workdir: Path, use_silence: bool = True
) -> Iterator[list[Chunk]]:
    """Yield the audio as silence-aligned WAV chunks, cleaning up afterwards."""
    workdir.mkdir(parents=True, exist_ok=True)
    silences = detect_silence(audio.path) if (use_silence and audio.duration > target) else []
    windows = plan_chunks(
        audio.duration, target=target, tolerance=max(5.0, target * 0.12), silences=silences
    )
    log.debug("split %.0fs into %d chunk(s)", audio.duration, len(windows))

    chunks: list[Chunk] = []
    try:
        if len(windows) == 1:
            chunks.append(Chunk(0, 0.0, audio.duration, audio.path))
        else:
            for index, (start, end) in enumerate(windows):
                samples = read_wav(audio.path, start, end)
                path = workdir / f"chunk_{index:04d}.wav"
                write_wav(path, samples)
                chunks.append(Chunk(index, start, end, path))
        yield chunks
    finally:
        for chunk in chunks:
            if chunk.path != audio.path:
                chunk.path.unlink(missing_ok=True)


def torch_autocast(device: str, compute_type: str):
    """Half-precision inference context, or a no-op on CPU/float32."""
    if device != "cuda" or compute_type not in ("float16", "bfloat16"):
        return contextlib.nullcontext()
    try:
        import torch
    except ImportError:  # pragma: no cover - availability checks catch this first
        return contextlib.nullcontext()

    dtype = torch.bfloat16 if compute_type == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def inference_mode():
    """``torch.inference_mode()`` when torch is present, else a no-op."""
    try:
        import torch
    except ImportError:  # pragma: no cover - availability checks catch this first
        return contextlib.nullcontext()

    return torch.inference_mode()


def free_cuda_memory() -> None:
    try:  # pragma: no cover - requires torch + CUDA
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def wrap_oom(exc: Exception, model_id: str) -> TranscriptionError:
    text = str(exc).lower()
    if "out of memory" in text:
        return TranscriptionError(
            f"The GPU ran out of memory running {model_id}.",
            hint="Lower the chunk length, or switch to Whisper large-v3-turbo.",
        )
    if "no module named" in text or "importerror" in text:
        return TranscriptionError(f"NeMo is missing a dependency: {exc}", hint=NEMO_INSTALL_HINT)
    if "connection" in text or "resolve" in text or "not a valid" in text:
        return TranscriptionError(
            f"Could not download the {model_id} weights.",
            hint="The first run needs internet access to Hugging Face.",
        )
    return TranscriptionError(f"{model_id} failed: {exc}")


def suggested_batch(vram_gb: float | None, *, small: int, medium: int, large: int) -> int:
    vram = vram_gb or 0.0
    if vram >= 20:
        return large
    if vram >= 10:
        return medium
    return small
