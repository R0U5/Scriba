import pytest

from rosetta import engines
from rosetta.engines._nemo import Chunk, chunked_wavs
from rosetta.engines.canary_qwen import _interpolate
from rosetta.engines.parakeet import _to_segments
from rosetta.errors import EngineNotAvailableError
from rosetta.media.audio import AudioFile


def test_catalogue_shape():
    entries = engines.catalogue()
    keys = {entry["key"] for entry in entries}
    assert {"faster-whisper", "parakeet", "canary-qwen"} <= keys
    for entry in entries:
        assert entry["models"]
        assert entry["default_model"] in {m["id"] for m in entry["models"]}
        assert isinstance(entry["available"], bool)
        if not entry["available"]:
            assert entry["hint"], "an unavailable engine must tell the user how to install it"


def test_every_model_has_comparable_metadata():
    for entry in engines.catalogue():
        for model in entry["models"]:
            assert model["vram_gb"] > 0
            assert model["languages"]
            assert model["wer"] is None or 0 < model["wer"] < 50


def test_aliases_resolve():
    assert engines.normalise_key("whisper") == "faster-whisper"
    assert engines.normalise_key("CANARY") == "canary-qwen"
    assert engines.normalise_key(None) == "faster-whisper"


def test_unknown_engine_raises_with_hint():
    with pytest.raises(EngineNotAvailableError) as excinfo:
        engines.create_engine("does-not-exist")
    assert "Available engines" in (excinfo.value.hint or "")


def test_engine_instances_are_cached():
    first = engines.create_engine("stub")
    assert engines.create_engine("stub") is first


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(engines, "detect_gpu", lambda: engines.GPUInfo(False))
    monkeypatch.setattr("rosetta.engines.base.detect_gpu", lambda: engines.GPUInfo(False))
    assert engines.resolve_device(None) == "cpu"
    assert engines.resolve_device("cuda") == "cuda"


# ------------------------------------------------------------- NeMo helpers
def test_chunked_wavs_splits_and_cleans_up(tone_wav, tmp_path):
    audio = AudioFile(path=tone_wav, duration=6.0)
    with chunked_wavs(audio, target=2.0, workdir=tmp_path, use_silence=False) as chunks:
        assert len(chunks) == 3
        assert all(chunk.path.exists() for chunk in chunks)
        paths = [chunk.path for chunk in chunks]
    assert not any(path.exists() for path in paths)


def test_chunked_wavs_passes_short_audio_through(tone_wav, tmp_path):
    audio = AudioFile(path=tone_wav, duration=6.0)
    with chunked_wavs(audio, target=60.0, workdir=tmp_path) as chunks:
        assert len(chunks) == 1
        assert chunks[0].path == tone_wav
    assert tone_wav.exists()  # the source must not be deleted


def test_parakeet_segment_conversion_applies_offset():
    class Hypothesis:
        text = "hello world"
        timestamp = {
            "segment": [{"start": 0.0, "end": 1.0, "segment": "hello"},
                        {"start": 1.0, "end": 2.0, "segment": "world"}],
            "word": [{"start": 0.1, "end": 0.5, "word": "hello"},
                     {"start": 1.1, "end": 1.6, "word": "world"}],
        }

    segments = _to_segments(Hypothesis(), offset=10.0, first_id=5, fallback=(10.0, 12.0), want_words=True)
    assert [s.id for s in segments] == [5, 6]
    assert segments[0].start == 10.0 and segments[1].end == 12.0
    assert segments[0].words[0].word == "hello"


def test_parakeet_falls_back_to_chunk_bounds_without_timestamps():
    class Hypothesis:
        text = "no timings here"
        timestamp = {}

    segments = _to_segments(Hypothesis(), offset=0.0, first_id=0, fallback=(4.0, 9.0), want_words=False)
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (4.0, 9.0)


def test_canary_interpolation_covers_the_window():
    chunk = Chunk(index=0, start=30.0, end=60.0, path=None)
    segments = _interpolate("First sentence here. Second one is longer than the first.", chunk, first_id=0)
    assert len(segments) == 2
    assert segments[0].start == 30.0
    assert segments[-1].end == 60.0
    assert segments[0].end == segments[1].start
