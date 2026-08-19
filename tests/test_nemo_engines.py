"""Parakeet and Canary-Qwen orchestration, exercised with stand-in models.

NeMo itself is a multi-gigabyte optional dependency, so these tests replace the
loaded model with a fake and verify what RosettaAI is responsible for: chunking,
timestamp offsets, batching, streaming callbacks and result assembly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rosetta.engines.canary_qwen import CanaryQwenEngine
from rosetta.engines.parakeet import ParakeetEngine
from rosetta.media.audio import AudioFile
from rosetta.models import TranscriptionOptions


class FakeParakeet:
    """Returns one hypothesis per call, with chunk-relative timestamps."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, paths, timestamps=True, batch_size=1, verbose=False):
        self.calls += 1
        index = self.calls
        return [
            SimpleNamespace(
                text=f"chunk {index}",
                timestamp={
                    "segment": [{"start": 0.0, "end": 1.0, "segment": f"chunk {index} first"},
                                {"start": 1.0, "end": 2.0, "segment": f"chunk {index} second"}],
                    "word": [],
                },
            )
        ]


def test_parakeet_stitches_chunks_with_absolute_timestamps(tone_wav, monkeypatch):
    monkeypatch.setattr("rosetta.engines.parakeet.CHUNK_SECONDS", 2.0)
    engine = ParakeetEngine(device="cpu")
    engine._model = FakeParakeet()
    engine._loaded = True

    streamed = []
    result = engine.transcribe(
        AudioFile(path=tone_wav, duration=6.0),
        TranscriptionOptions(),
        on_segment=streamed.append,
    )

    assert engine._model.calls == 3                      # 6s split into 2s windows
    assert len(result.segments) == 6
    assert [s.id for s in result.segments] == list(range(6))
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert starts[2] == pytest.approx(2.0)               # second chunk is offset
    assert result.segments[-1].end == pytest.approx(6.0)
    assert result.duration == 6.0
    assert len(streamed) == 6
    assert result.timing_precision == "segment"


def test_parakeet_progress_reaches_completion(tone_wav, monkeypatch):
    monkeypatch.setattr("rosetta.engines.parakeet.CHUNK_SECONDS", 3.0)
    engine = ParakeetEngine(device="cpu")
    engine._model = FakeParakeet()
    engine._loaded = True

    fractions = []
    engine.transcribe(
        AudioFile(path=tone_wav, duration=6.0),
        TranscriptionOptions(),
        progress=lambda fraction, message: fractions.append(fraction),
    )
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


class FakeCanary:
    audio_locator_tag = "<|audio|>"

    def __init__(self):
        self.prompts = []
        self.tokenizer = SimpleNamespace(ids_to_text=lambda ids: ids)

    def generate(self, prompts, max_new_tokens):
        self.prompts.append((prompts, max_new_tokens))
        # NeMo hands back CUDA tensors; mimic the .cpu() hop the real one needs.
        return [
            SimpleNamespace(cpu=lambda i=i: f"Sentence one of {i}. Sentence two of {i}.")
            for i in range(len(prompts))
        ]


def test_canary_batches_chunks_and_interpolates_timings(tone_wav, monkeypatch):
    monkeypatch.setattr("rosetta.engines.canary_qwen.CHUNK_SECONDS", 2.0)
    monkeypatch.setattr("rosetta.engines.canary_qwen.suggested_batch", lambda *a, **k: 2)
    engine = CanaryQwenEngine(device="cpu")
    engine._model = FakeCanary()
    engine._loaded = True

    result = engine.transcribe(AudioFile(path=tone_wav, duration=6.0), TranscriptionOptions())

    assert len(engine._model.prompts) == 2               # 3 chunks, batch size 2
    first_batch = engine._model.prompts[0][0]
    assert len(first_batch) == 2
    assert "<|audio|>" in first_batch[0][0]["content"]
    assert first_batch[0][0]["audio"]

    assert result.timing_precision == "approximate"
    assert result.language == "en"
    assert [s.id for s in result.segments] == list(range(len(result.segments)))
    assert result.segments[0].start == 0.0
    assert result.segments[-1].end == pytest.approx(6.0)
    for earlier, later in zip(result.segments, result.segments[1:], strict=False):
        assert earlier.end <= later.start + 1e-6


def test_canary_requires_a_gpu():
    status = CanaryQwenEngine.availability()
    assert status.available is False  # no NeMo (and no CUDA) in the test environment
    assert status.hint
