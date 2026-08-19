"""The default engine's glue code, checked against the real faster-whisper API.

Model weights are never downloaded here: a fake model stands in, but every
keyword argument is validated against the installed library's real signature,
so a breaking API change is caught rather than surfacing at runtime.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from rosetta.engines.faster_whisper_engine import FasterWhisperEngine, _load_error
from rosetta.errors import TranscriptionError
from rosetta.media.audio import AudioFile
from rosetta.models import TranscriptionOptions

faster_whisper = pytest.importorskip("faster_whisper")


class FakeSegment(SimpleNamespace):
    pass


class FakeModel:
    """Stands in for ``faster_whisper.WhisperModel``."""

    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        words = [SimpleNamespace(start=0.0, end=0.4, word="Hello", probability=0.98)]
        segments = [
            FakeSegment(id=0, start=0.0, end=2.0, text=" Hello there. ",
                        avg_logprob=-0.2, no_speech_prob=0.01, words=words),
            FakeSegment(id=1, start=2.0, end=4.0, text="   ",  # blank: must be dropped
                        avg_logprob=-0.9, no_speech_prob=0.8, words=None),
            FakeSegment(id=2, start=4.0, end=6.0, text="Second line.",
                        avg_logprob=-0.3, no_speech_prob=0.02, words=None),
        ]
        info = SimpleNamespace(language="en", language_probability=0.99, duration=6.0)
        return iter(segments), info


def _engine(model: FakeModel) -> FasterWhisperEngine:
    engine = FasterWhisperEngine(model="large-v3", device="cpu")
    engine._model = model
    engine._batched = None
    engine._loaded = True
    return engine


def test_transcribe_maps_segments_words_and_metadata(tone_wav):
    model = FakeModel()
    engine = _engine(model)
    seen, streamed = [], []

    result = engine.transcribe(
        AudioFile(path=tone_wav, duration=6.0),
        TranscriptionOptions(word_timestamps=True, initial_prompt="Acme Corp"),
        progress=lambda fraction, message: seen.append(fraction),
        on_segment=streamed.append,
    )

    assert result.text == "Hello there. Second line."
    assert [s.id for s in result.segments] == [0, 1]      # blank segment dropped, ids compacted
    assert result.segments[0].words[0].word == "Hello"
    assert result.language == "en" and result.language_probability == 0.99
    assert result.duration == 6.0
    assert result.word_count == 4
    assert result.model == "large-v3" and result.device == "cpu"
    assert [s.text for s in streamed] == ["Hello there.", "Second line."]
    assert seen and seen[-1] == 1.0


def test_every_kwarg_exists_in_the_installed_library(tone_wav):
    model = FakeModel()
    engine = _engine(model)
    engine.transcribe(AudioFile(path=tone_wav, duration=6.0), TranscriptionOptions())

    supported = set(inspect.signature(faster_whisper.WhisperModel.transcribe).parameters)
    assert set(model.kwargs) <= supported, set(model.kwargs) - supported


def test_batched_pipeline_kwargs_are_supported():
    supported = set(inspect.signature(faster_whisper.BatchedInferencePipeline.transcribe).parameters)
    assert "batch_size" in supported


def test_hallucination_guard_is_enabled(tone_wav):
    model = FakeModel()
    _engine(model).transcribe(AudioFile(path=tone_wav, duration=6.0), TranscriptionOptions())
    assert model.kwargs["condition_on_previous_text"] is False


def test_cancellation_is_honoured(tone_wav):
    from rosetta.errors import JobCancelled

    engine = _engine(FakeModel())
    with pytest.raises(JobCancelled):
        engine.transcribe(
            AudioFile(path=tone_wav, duration=6.0), TranscriptionOptions(), cancelled=lambda: True
        )


def test_batch_size_scales_with_vram(monkeypatch):
    from rosetta.engines.base import GPUInfo

    engine = FasterWhisperEngine(device="cpu")
    for vram, expected in [(24.0, 16), (12.0, 8), (8.0, 4), (4.0, 2)]:
        monkeypatch.setattr(
            "rosetta.engines.faster_whisper_engine.detect_gpu",
            lambda vram=vram: GPUInfo(True, "test", vram),
        )
        assert engine._batch_size() == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA failed with error out of memory", "smaller model"),
        ("Could not resolve host: huggingface.co", "internet access"),
        ("Requested unsupported compute type", "float16 on CUDA"),
    ],
)
def test_load_errors_carry_actionable_hints(message, expected):
    error = _load_error(RuntimeError(message), "large-v3", "cuda", "float16")
    assert isinstance(error, TranscriptionError)
    assert expected in (error.hint or "")


def test_availability_reports_installed():
    assert FasterWhisperEngine.availability().available is True


# --------------------------------------------------------------- weights ----
def test_repo_id_resolution():
    from rosetta.engines.faster_whisper_engine import _repo_id_for

    assert _repo_id_for("large-v3") == "Systran/faster-whisper-large-v3"
    assert _repo_id_for("org/custom-ct2") == "org/custom-ct2"
    assert _repo_id_for("not-a-real-size") is None


def test_allow_patterns_match_faster_whisper(monkeypatch):
    """We must fetch exactly the files faster-whisper itself asks for."""
    import inspect

    from rosetta.engines.faster_whisper_engine import ALLOW_PATTERNS

    source = inspect.getsource(faster_whisper.utils.download_model)
    for pattern in ALLOW_PATTERNS:
        assert f'"{pattern}"' in source, f"{pattern} is no longer used by faster-whisper"


def test_cached_weights_skip_the_download(monkeypatch, tmp_path):
    """A second run must not touch the network."""
    calls = []
    monkeypatch.setattr(
        "faster_whisper.utils.download_model",
        lambda model, local_files_only=False, **kw: (calls.append(local_files_only), str(tmp_path))[1],
    )
    engine = FasterWhisperEngine(model="large-v3", device="cpu")
    assert engine._prefetch_weights(None) == str(tmp_path)
    assert calls == [True], "the cache must be consulted with local_files_only"


def test_download_reports_progress(monkeypatch, tmp_path):
    def missing_from_cache(model, local_files_only=False, **kw):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr("faster_whisper.utils.download_model", missing_from_cache)
    monkeypatch.setattr(
        "rosetta.engines.faster_whisper_engine._remote_size", lambda repo: 3_000_000_000
    )

    def fake_snapshot(repo_id, allow_patterns=None, tqdm_class=None, **kw):
        assert repo_id == "Systran/faster-whisper-large-v3"
        bar = tqdm_class(total=3_000_000_000, unit="B")
        bar.update(1_500_000_000)
        bar.update(1_500_000_000)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

    messages = []
    engine = FasterWhisperEngine(model="large-v3", device="cpu")
    path = engine._prefetch_weights(lambda fraction, message: messages.append((fraction, message)))

    assert path == str(tmp_path)
    assert "3.0 GB" in messages[0][1] and "first run only" in messages[0][1]
    assert any("1.50 / 3.0 GB" in message for _, message in messages)
    assert messages[-1] == (1.0, "Weights downloaded")


def test_download_failure_falls_back_to_the_loader(monkeypatch):
    monkeypatch.setattr(
        "faster_whisper.utils.download_model",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("not cached")),
    )
    monkeypatch.setattr("rosetta.engines.faster_whisper_engine._remote_size", lambda repo: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(OSError("network down")),
    )
    engine = FasterWhisperEngine(model="large-v3", device="cpu")
    assert engine._prefetch_weights(None) == "large-v3"
