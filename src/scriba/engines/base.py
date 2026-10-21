"""Engine contract, hardware detection and model catalogue metadata."""

from __future__ import annotations

import functools
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from ..errors import EngineNotAvailableError
from ..logging_setup import get_logger
from ..media.audio import AudioFile
from ..models import Segment, TranscriptionOptions, TranscriptResult

log = get_logger(__name__)

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]
#: Called with each segment as soon as it is recognised, for live preview.
SegmentCB = Callable[["Segment"], None]


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class GPUInfo:
    available: bool
    name: str | None = None
    total_vram_gb: float | None = None
    source: str = "none"

    @property
    def label(self) -> str:
        if not self.available:
            return "CPU only"
        if self.name and self.total_vram_gb:
            return f"{self.name} · {self.total_vram_gb:.0f} GB"
        return self.name or "CUDA GPU"


@functools.lru_cache(maxsize=1)
def detect_gpu() -> GPUInfo:
    """Detect an NVIDIA GPU without importing torch if we can avoid it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        line = (out.stdout or "").strip().splitlines()
        if out.returncode == 0 and line:
            name, _, memory = line[0].partition(",")
            return GPUInfo(True, name.strip(), float(memory.strip()) / 1024.0, "nvidia-smi")
    except Exception:  # pragma: no cover - no driver
        pass

    try:  # pragma: no cover - depends on optional torch
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(True, props.name, props.total_memory / 1e9, "torch")
    except Exception:
        pass

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:  # pragma: no cover
            return GPUInfo(True, "CUDA GPU", None, "ctranslate2")
    except Exception:
        pass

    return GPUInfo(False)


def resolve_device(requested: str | None) -> str:
    """``None``/``auto`` -> cuda when a GPU is present, else cpu."""
    if requested and requested not in ("auto", ""):
        return requested
    return "cuda" if detect_gpu().available else "cpu"


# --------------------------------------------------------------------------- #
# Catalogue metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One selectable checkpoint, with the numbers that drive the UI picker."""

    id: str
    label: str
    parameters: str
    vram_gb: float
    languages: str
    wer: float | None = None  # Open ASR Leaderboard average WER, %
    speed: str = ""
    note: str = ""
    default: bool = False


@dataclass(frozen=True, slots=True)
class Availability:
    available: bool
    reason: str | None = None
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class EngineSpec:
    key: str
    label: str
    tagline: str
    vendor: str
    license: str
    models: list[ModelSpec] = field(default_factory=list)
    install_extra: str | None = None
    install_hint: str | None = None
    supports_translate: bool = False
    supports_word_timestamps: bool = True
    requires_gpu: bool = False

    @property
    def default_model(self) -> str:
        for model in self.models:
            if model.default:
                return model.id
        return self.models[0].id if self.models else ""

    def model(self, model_id: str) -> ModelSpec | None:
        return next((m for m in self.models if m.id == model_id), None)


# --------------------------------------------------------------------------- #
# Engine contract
# --------------------------------------------------------------------------- #
class ASREngine(ABC):
    """A loadable speech-recognition backend.

    Instances are cached process-wide (see :mod:`scriba.engines.registry`) so a
    multi-gigabyte model is paid for once, not once per job.
    """

    spec: EngineSpec

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_id = model or self.spec.default_model
        self.device = resolve_device(device)
        self.compute_type = compute_type or self.default_compute_type(self.device)
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    @abstractmethod
    def availability(cls) -> Availability:
        """Can this engine run right now (dependencies + hardware)?"""

    @abstractmethod
    def load(self, progress: ProgressCB | None = None) -> None:
        """Load weights into memory. Idempotent."""

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def default_compute_type(self, device: str) -> str:
        return "float16" if device == "cuda" else "int8"

    # -- work --------------------------------------------------------------
    @abstractmethod
    def transcribe(
        self,
        audio: AudioFile,
        options: TranscriptionOptions,
        *,
        progress: ProgressCB | None = None,
        cancelled: CancelCB | None = None,
        on_segment: SegmentCB | None = None,
    ) -> TranscriptResult:
        """Transcribe a decoded 16 kHz mono WAV.

        ``on_segment`` is invoked as each segment is decoded so callers can show
        the transcript building up instead of a spinner.
        """

    # -- helpers -----------------------------------------------------------
    def ensure_available(self) -> None:
        status = self.availability()
        if not status.available:
            raise EngineNotAvailableError(
                status.reason or f"{self.spec.label} is not available.",
                hint=status.hint or self.spec.install_hint,
            )

    def describe(self) -> dict[str, str]:
        return {
            "engine": self.spec.key,
            "model": self.model_id,
            "device": self.device,
            "compute_type": self.compute_type,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<{type(self).__name__} model={self.model_id} device={self.device} "
            f"compute={self.compute_type} loaded={self._loaded}>"
        )
