"""OpenAI Whisper via CTranslate2 (faster-whisper).

The default engine: 99 languages, translation-to-English, word timestamps,
and a 4-5x speedup over the reference implementation at the same accuracy.
Whisper large-v3 in float16 needs about 5 GB of VRAM, which leaves plenty of
headroom on a 12 GB card such as an RTX 4070.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..errors import JobCancelled, TranscriptionError
from ..logging_setup import get_logger
from ..media.audio import AudioFile
from ..models import Segment, TranscriptionOptions, TranscriptResult, Word
from ..util.text import word_count
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

SPEC = EngineSpec(
    key="faster-whisper",
    label="Whisper (faster-whisper)",
    tagline="99 languages, battle-tested, translates to English. The safe default.",
    vendor="OpenAI · SYSTRAN",
    license="MIT",
    install_extra="whisper",
    install_hint="Install with: pip install faster-whisper",
    supports_translate=True,
    supports_word_timestamps=True,
    models=[
        ModelSpec(
            id="large-v3",
            label="Whisper large-v3",
            parameters="1.55B",
            vram_gb=5.0,
            languages="99 languages",
            wer=7.44,
            speed="~15-25x realtime on an RTX 4070",
            note="Best multilingual accuracy in the Whisper family.",
            default=True,
        ),
        ModelSpec(
            id="large-v3-turbo",
            label="Whisper large-v3-turbo",
            parameters="809M",
            vram_gb=3.0,
            languages="99 languages",
            wer=7.83,
            speed="~40-60x realtime on an RTX 4070",
            note="Pruned decoder: nearly large-v3 quality at roughly 3x the speed.",
        ),
        ModelSpec(
            id="distil-large-v3.5",
            label="Distil-Whisper large-v3.5",
            parameters="756M",
            vram_gb=2.5,
            languages="English only",
            wer=7.08,
            speed="~50x realtime on an RTX 4070",
            note="Distilled English-only model; very fast, low VRAM.",
        ),
        ModelSpec(
            id="medium",
            label="Whisper medium",
            parameters="769M",
            vram_gb=2.5,
            languages="99 languages",
            wer=9.5,
            speed="~40x realtime on an RTX 4070",
            note="Fallback for GPUs with little free memory.",
        ),
        ModelSpec(
            id="small",
            label="Whisper small",
            parameters="244M",
            vram_gb=1.2,
            languages="99 languages",
            wer=12.0,
            speed="Runs comfortably on CPU",
            note="Draft-quality; useful for quick checks and CPU-only machines.",
        ),
    ],
)


class FasterWhisperEngine(ASREngine):
    spec = SPEC

    def __init__(self, model=None, device=None, compute_type=None) -> None:
        super().__init__(model, device, compute_type)
        self._model = None
        self._batched = None

    # -- availability ------------------------------------------------------
    @classmethod
    def availability(cls) -> Availability:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return Availability(
                False,
                "faster-whisper is not installed.",
                "Run: pip install faster-whisper",
            )
        return Availability(True)

    def default_compute_type(self, device: str) -> str:
        # int8_float16 halves VRAM with a negligible accuracy cost, but float16
        # is faster and fits easily on a 12 GB card, so it stays the default.
        return "float16" if device == "cuda" else "int8"

    # -- lifecycle ---------------------------------------------------------
    def load(self, progress: ProgressCB | None = None) -> None:
        if self._loaded:
            return
        self.ensure_available()
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        if progress:
            progress(0.0, f"Loading {self.model_id} on {self.device.upper()}…")
        # Weights are fetched first, with progress: faster-whisper downloads them
        # silently, which makes a 3 GB first run look like a hang.
        source = self._prefetch_weights(progress)

        started = time.monotonic()
        try:
            self._model = WhisperModel(
                source,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=0,  # 0 -> use every physical core
            )
        except Exception as exc:
            raise _load_error(exc, self.model_id, self.device, self.compute_type) from exc

        self._batched = BatchedInferencePipeline(model=self._model) if self._use_batching else None
        self._loaded = True
        elapsed = time.monotonic() - started
        log.info("Loaded %s (%s/%s) in %.1fs", self.model_id, self.device, self.compute_type, elapsed)
        if progress:
            progress(1.0, f"Model ready ({elapsed:.1f}s)")

    @property
    def _use_batching(self) -> bool:
        """Batched decoding is a large win on GPU and a wash on CPU."""
        return self.device == "cuda"

    def unload(self) -> None:
        self._batched = None
        self._model = None
        self._loaded = False

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

        kwargs = {
            "language": options.language or None,
            "task": options.task.value,
            "beam_size": max(1, options.beam_size),
            "temperature": options.temperature,
            "initial_prompt": options.initial_prompt or None,
            "word_timestamps": options.word_timestamps,
            "vad_filter": options.vad_filter,
            # Whisper's autoregressive loop can spiral into repeated text when
            # it conditions on its own previous output; disabling that is the
            # single most effective anti-hallucination setting.
            "condition_on_previous_text": False,
        }

        started = time.monotonic()
        try:
            if self._batched is not None:
                iterator, info = self._batched.transcribe(
                    str(audio.path), batch_size=self._batch_size(), **kwargs
                )
            else:
                iterator, info = self._model.transcribe(str(audio.path), **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"Whisper failed to start: {exc}") from exc

        total = audio.duration or float(getattr(info, "duration", 0.0)) or 0.0
        segments: list[Segment] = []
        texts: list[str] = []

        try:
            for raw in iterator:
                if cancelled is not None and cancelled():
                    raise JobCancelled()
                segment = Segment(
                    id=len(segments),
                    start=float(raw.start),
                    end=float(raw.end),
                    text=raw.text.strip(),
                    avg_logprob=getattr(raw, "avg_logprob", None),
                    no_speech_prob=getattr(raw, "no_speech_prob", None),
                    words=[
                        Word(start=float(w.start), end=float(w.end), word=w.word,
                             probability=getattr(w, "probability", None))
                        for w in (raw.words or [])
                    ] if options.word_timestamps and raw.words else None,
                )
                if segment.text:
                    segments.append(segment)
                    texts.append(segment.text)
                    if on_segment:
                        on_segment(segment)
                if progress and total > 0:
                    progress(
                        min(segment.end / total, 0.999),
                        f"Transcribing · {_clock(segment.end)} / {_clock(total)}",
                    )
        except JobCancelled:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Whisper failed while decoding: {exc}") from exc

        elapsed = time.monotonic() - started
        text = " ".join(texts).strip()
        if progress:
            progress(1.0, f"Transcribed {len(segments)} segments")

        return TranscriptResult(
            text=text,
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=total,
            engine=self.spec.key,
            model=self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            processing_seconds=elapsed,
            word_count=word_count(text),
        )

    # -- weights -----------------------------------------------------------
    def _prefetch_weights(self, progress: ProgressCB | None) -> str:
        """Make sure the weights are on disk, reporting download progress.

        Returns a local directory when it can, otherwise the model id so that
        faster-whisper resolves it the usual way. Any failure here is
        non-fatal: the model loader will report a proper error if the weights
        really are unavailable.
        """
        if Path(self.model_id).is_dir():
            return self.model_id

        try:
            from faster_whisper.utils import download_model
        except ImportError:  # pragma: no cover - checked by availability()
            return self.model_id

        # Already downloaded? Then this returns instantly and touches no network.
        try:
            return download_model(self.model_id, local_files_only=True)
        except Exception:
            pass

        repo_id = _repo_id_for(self.model_id)
        if repo_id is None:
            return self.model_id

        total = _remote_size(repo_id)
        if progress:
            size = f" ({total / 1e9:.1f} GB)" if total else ""
            progress(0.0, f"Downloading {self.model_id}{size} — first run only")

        try:
            from huggingface_hub import snapshot_download

            downloaded = 0

            def report(chunk: int) -> None:
                nonlocal downloaded
                downloaded += chunk
                if not progress:
                    return
                if total:
                    progress(
                        min(downloaded / total, 0.99),
                        f"Downloading {self.model_id} · {downloaded / 1e9:.2f} / {total / 1e9:.1f} GB",
                    )
                else:
                    progress(0.0, f"Downloading {self.model_id} · {downloaded / 1e6:.0f} MB")

            path = snapshot_download(
                repo_id,
                allow_patterns=ALLOW_PATTERNS,
                tqdm_class=_reporting_tqdm(report),
            )
            if progress:
                progress(1.0, "Weights downloaded")
            return path
        except Exception as exc:
            log.debug("Prefetch of %s failed (%s); falling back to the loader", repo_id, exc)
            return self.model_id

    def _batch_size(self) -> int:
        """Pick a batch size that fits the detected card."""
        vram = detect_gpu().total_vram_gb or 0.0
        if vram >= 20:
            return 16
        if vram >= 10:  # RTX 4070 / 4070 Ti / 3080
            return 8
        if vram >= 6:
            return 4
        return 2


#: The files a CTranslate2 Whisper model actually needs, mirroring
#: faster_whisper.utils.download_model so no extra weights are fetched.
ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _repo_id_for(model_id: str) -> str | None:
    """Map a size name such as ``large-v3`` to its Hugging Face repository."""
    if "/" in model_id:
        return model_id
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(model_id)
    except Exception:  # pragma: no cover - private API moved
        return None


def _remote_size(repo_id: str) -> int | None:
    """Total download size in bytes, or ``None`` when it cannot be determined."""
    try:
        from fnmatch import fnmatch

        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        total = sum(
            sibling.size or 0
            for sibling in (info.siblings or [])
            if any(fnmatch(sibling.rfilename, pattern) for pattern in ALLOW_PATTERNS)
        )
        return total or None
    except Exception:  # pragma: no cover - offline or API change
        return None


def _reporting_tqdm(report):
    """A tqdm class that forwards byte counts instead of drawing a bar."""
    from tqdm.auto import tqdm as base_tqdm

    class ReportingTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # our own UI shows the progress
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if n:
                report(n)
            return super().update(n)

    return ReportingTqdm


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def _load_error(exc: Exception, model: str, device: str, compute_type: str) -> TranscriptionError:
    text = str(exc).lower()
    if "out of memory" in text or "cublas" in text:
        return TranscriptionError(
            f"The GPU ran out of memory loading {model}.",
            hint="Pick a smaller model (large-v3-turbo) or set compute_type to int8_float16.",
        )
    if "connection" in text or "resolve" in text or "offline" in text or "couldn't find" in text:
        return TranscriptionError(
            f"Could not download the {model} weights.",
            hint="The first run needs internet access to Hugging Face; check your connection or proxy.",
        )
    if "unsupported" in text and "compute" in text:
        return TranscriptionError(
            f"compute_type '{compute_type}' is not supported on this {device} device.",
            hint="Use float16 on CUDA, or int8 on CPU.",
        )
    return TranscriptionError(f"Could not load {model}: {exc}")
