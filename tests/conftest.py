"""Shared fixtures.

The stub engine lets the whole pipeline — download, decode, queue, writers,
HTTP API — be exercised without downloading multi-gigabyte model weights.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from rosetta.config import Settings
from rosetta.engines.base import ASREngine, Availability, EngineSpec, ModelSpec
from rosetta.media.audio import write_wav
from rosetta.models import Segment, TranscriptResult

STUB_SPEC = EngineSpec(
    key="stub",
    label="Stub engine",
    tagline="Deterministic output for tests.",
    vendor="tests",
    license="MIT",
    models=[
        ModelSpec(id="stub-1", label="Stub", parameters="0", vram_gb=0.1,
                  languages="English", wer=1.0, default=True)
    ],
)


class StubEngine(ASREngine):
    """Emits one segment per two seconds of audio."""

    spec = STUB_SPEC
    calls: list[dict] = []

    @classmethod
    def availability(cls) -> Availability:
        return Availability(True)

    def load(self, progress=None) -> None:
        if progress:
            progress(1.0, "Stub ready")
        self._loaded = True

    def transcribe(
        self, audio, options, *, progress=None, cancelled=None, on_segment=None
    ) -> TranscriptResult:
        StubEngine.calls.append({"path": str(audio.path), "options": options.model_dump()})
        count = max(1, int(audio.duration // 2))
        segments = [
            Segment(id=i, start=i * 2.0, end=min((i + 1) * 2.0, audio.duration),
                    text=f"Stub segment {i} of the transcript.")
            for i in range(count)
        ]
        for index, segment in enumerate(segments):
            if cancelled and cancelled():
                from rosetta.errors import JobCancelled

                raise JobCancelled()
            if on_segment:
                on_segment(segment)
            if progress:
                progress((index + 1) / count, f"Transcribing {index + 1}/{count}")
        text = " ".join(s.text for s in segments)
        return TranscriptResult(
            text=text, segments=segments, language="en", duration=audio.duration,
            engine="stub", model=self.model_id, device="cpu", compute_type="int8",
            processing_seconds=0.05, word_count=len(text.split()),
        )


@pytest.fixture(autouse=True)
def _register_stub():
    """Make the stub engine selectable for the duration of each test."""
    from rosetta import engines

    engines.ENGINE_CLASSES["stub"] = StubEngine
    StubEngine.calls.clear()
    yield
    engines.unload_all()
    engines.ENGINE_CLASSES.pop("stub", None)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = Settings(
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        data_dir=tmp_path / "data",
        engine="stub",
        history_limit=10,
        log_level="WARNING",
    )
    config.ensure_dirs()
    return config


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory) -> Path:
    """A 6-second 16 kHz mono WAV."""
    path = tmp_path_factory.mktemp("media") / "tone.wav"
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 6, 16_000 * 6, dtype=np.float32)) * 0.2
    write_wav(path, samples)
    return path


@pytest.fixture(scope="session")
def tone_mp3(tmp_path_factory) -> Path:
    """The same tone, encoded, so the ffmpeg decode path is exercised."""
    from rosetta.media.ffmpeg import ffmpeg_path

    path = tmp_path_factory.mktemp("media") / "tone.mp3"
    subprocess.run(
        [ffmpeg_path(), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=6", "-c:a", "libmp3lame", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def sample_result() -> TranscriptResult:
    segments = [
        Segment(id=0, start=0.0, end=3.2, text="The first thing to understand is the pipeline."),
        Segment(id=1, start=3.2, end=6.9, text="Audio is decoded to sixteen kilohertz mono."),
        Segment(id=2, start=6.9, end=12.4, text="Then the model turns it into text, and we write files."),
    ]
    text = " ".join(s.text for s in segments)
    return TranscriptResult(
        text=text, segments=segments, language="en", duration=12.4,
        engine="stub", model="stub-1", device="cpu", compute_type="int8",
        processing_seconds=1.0, word_count=len(text.split()),
    )
