"""ASR engine registry.

Engines are described by static specs (so the UI can show the full catalogue
even when an engine is not installed) and instantiated lazily. Loaded models are
cached process-wide: switching jobs never reloads the same weights twice.
"""

from __future__ import annotations

import threading

from ..errors import EngineNotAvailableError
from ..logging_setup import get_logger
from .base import (
    ASREngine,
    Availability,
    EngineSpec,
    GPUInfo,
    ModelSpec,
    detect_gpu,
    resolve_device,
)
from .canary_qwen import CanaryQwenEngine
from .faster_whisper_engine import FasterWhisperEngine
from .parakeet import ParakeetEngine

log = get_logger(__name__)

ENGINE_CLASSES: dict[str, type[ASREngine]] = {
    FasterWhisperEngine.spec.key: FasterWhisperEngine,
    ParakeetEngine.spec.key: ParakeetEngine,
    CanaryQwenEngine.spec.key: CanaryQwenEngine,
}

#: Aliases so "whisper", "large-v3" and friends do what people expect.
ALIASES = {
    "whisper": "faster-whisper",
    "faster_whisper": "faster-whisper",
    "fasterwhisper": "faster-whisper",
    "nemo": "parakeet",
    "parakeet-tdt": "parakeet",
    "canary": "canary-qwen",
    "canary_qwen": "canary-qwen",
}

_lock = threading.Lock()
_cache: dict[tuple[str, str, str, str], ASREngine] = {}


def normalise_key(key: str | None) -> str:
    key = (key or "faster-whisper").strip().lower()
    return ALIASES.get(key, key)


def engine_specs() -> list[EngineSpec]:
    return [cls.spec for cls in ENGINE_CLASSES.values()]


def get_spec(key: str) -> EngineSpec:
    key = normalise_key(key)
    if key not in ENGINE_CLASSES:
        raise EngineNotAvailableError(
            f"Unknown engine '{key}'.",
            hint=f"Available engines: {', '.join(ENGINE_CLASSES)}",
        )
    return ENGINE_CLASSES[key].spec


def availability(key: str) -> Availability:
    return ENGINE_CLASSES[normalise_key(key)].availability()


def catalogue() -> list[dict]:
    """Everything the UI needs to render the engine/model picker."""
    gpu = detect_gpu()
    out: list[dict] = []
    for cls in ENGINE_CLASSES.values():
        spec = cls.spec
        status = cls.availability()
        out.append(
            {
                "key": spec.key,
                "label": spec.label,
                "tagline": spec.tagline,
                "vendor": spec.vendor,
                "license": spec.license,
                "available": status.available,
                "reason": status.reason,
                "hint": status.hint or spec.install_hint,
                "install_extra": spec.install_extra,
                "supports_translate": spec.supports_translate,
                "supports_word_timestamps": spec.supports_word_timestamps,
                "requires_gpu": spec.requires_gpu,
                "default_model": spec.default_model,
                "models": [
                    {
                        "id": model.id,
                        "label": model.label,
                        "parameters": model.parameters,
                        "vram_gb": model.vram_gb,
                        "languages": model.languages,
                        "wer": model.wer,
                        "speed": model.speed,
                        "note": model.note,
                        "default": model.default,
                        "fits": gpu.total_vram_gb is None or model.vram_gb <= gpu.total_vram_gb,
                    }
                    for model in spec.models
                ],
            }
        )
    return out


def create_engine(
    key: str | None = None,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> ASREngine:
    """Return a (possibly cached) engine instance, without loading weights yet."""
    key = normalise_key(key)
    cls = ENGINE_CLASSES.get(key)
    if cls is None:
        raise EngineNotAvailableError(
            f"Unknown engine '{key}'.", hint=f"Available engines: {', '.join(ENGINE_CLASSES)}"
        )

    status = cls.availability()
    if not status.available:
        raise EngineNotAvailableError(
            status.reason or f"{cls.spec.label} is not available.",
            hint=status.hint or cls.spec.install_hint,
        )

    model = model or cls.spec.default_model
    resolved_device = resolve_device(device)
    cache_key = (key, model, resolved_device, compute_type or "auto")

    with _lock:
        engine = _cache.get(cache_key)
        if engine is None:
            # One model in VRAM at a time: free the previous one first.
            for stale_key, stale in list(_cache.items()):
                if stale_key[2] == resolved_device and stale.loaded:
                    log.info("Unloading %s to make room for %s", stale.model_id, model)
                    stale.unload()
                    _cache.pop(stale_key, None)
            engine = cls(model=model, device=device, compute_type=compute_type)
            _cache[cache_key] = engine
        return engine


def unload_all() -> None:
    with _lock:
        for engine in _cache.values():
            engine.unload()
        _cache.clear()


__all__ = [
    "ASREngine",
    "Availability",
    "EngineSpec",
    "GPUInfo",
    "ModelSpec",
    "availability",
    "catalogue",
    "create_engine",
    "detect_gpu",
    "engine_specs",
    "get_spec",
    "normalise_key",
    "resolve_device",
    "unload_all",
]
