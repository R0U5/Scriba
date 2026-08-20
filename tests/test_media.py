import os

import numpy as np
import pytest

from rosetta.config import Settings
from rosetta.errors import InvalidURLError, MediaError
from rosetta.media.audio import decode_to_wav, plan_chunks, read_wav, wav_duration, write_wav
from rosetta.media.fetch import (
    _as_local_path,
    _is_media_content_type,
    _looks_like_direct_media,
    fetch_media,
    validate_url,
)


# --------------------------------------------------------------------- audio
def test_wav_round_trip(tmp_path):
    samples = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    path = write_wav(tmp_path / "a.wav", samples)
    assert wav_duration(path) == pytest.approx(1.0)
    assert np.allclose(read_wav(path), samples, atol=1e-3)


def test_read_wav_slice(tone_wav):
    assert read_wav(tone_wav, 1.0, 2.0).shape[0] == 16_000


def test_read_wav_rejects_non_wav(tmp_path):
    bogus = tmp_path / "b.wav"
    bogus.write_bytes(b"not audio")
    with pytest.raises(MediaError):
        wav_duration(bogus)


def test_decode_mp3_to_pcm(tone_mp3, tmp_path):
    seen: list[str] = []
    audio = decode_to_wav(tone_mp3, tmp_path / "out.wav", progress=lambda f, m: seen.append(m))
    assert audio.duration == pytest.approx(6.0, abs=0.2)
    assert audio.path.exists()
    assert seen  # progress was reported


def test_decode_rejects_garbage(tmp_path):
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"\x00" * 4096)
    with pytest.raises(MediaError):
        decode_to_wav(junk, tmp_path / "out.wav")


# -------------------------------------------------------------------- chunks
def test_plan_chunks_short_audio_is_one_window():
    assert plan_chunks(120.0, target=600.0) == [(0.0, 120.0)]


def test_plan_chunks_snaps_to_silence():
    chunks = plan_chunks(1500.0, target=600.0, silences=[(580.0, 586.0), (1180.0, 1184.0)])
    assert [round(c[0]) for c in chunks] == [0, 583, 1182]
    assert chunks[-1][1] == 1500.0
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))  # no gaps


def test_plan_chunks_ignores_distant_silence():
    chunks = plan_chunks(1200.0, target=600.0, silences=[(10.0, 12.0)], tolerance=30.0)
    assert chunks[0][1] == 600.0


# --------------------------------------------------------------------- urls
def test_validate_url_adds_scheme():
    assert validate_url("example.com/a.mp3") == "https://example.com/a.mp3"


def test_validate_url_rejects_other_schemes():
    with pytest.raises(InvalidURLError):
        validate_url("ftp://example.com/a.mp3")


def test_validate_url_blocks_private_hosts_by_default():
    with pytest.raises(InvalidURLError):
        validate_url("http://127.0.0.1:8080/a.mp3")
    assert validate_url("http://127.0.0.1:8080/a.mp3", allow_private=True)


def test_validate_url_rejects_empty():
    with pytest.raises(InvalidURLError):
        validate_url("   ")


def test_direct_media_detection():
    assert _looks_like_direct_media("https://x.com/a/b.MP3?token=1")
    assert not _looks_like_direct_media("https://youtube.com/watch?v=abc")
    assert _is_media_content_type("audio/mpeg", "https://x.com/stream")
    assert _is_media_content_type("application/octet-stream", "https://x.com/a.m4a")
    assert not _is_media_content_type("text/html", "https://x.com/page")


def test_fetch_media_accepts_local_paths(tone_mp3, tmp_path):
    result = fetch_media(str(tone_mp3), tmp_path, Settings())
    assert result.path == tone_mp3
    assert result.info.extractor == "local file"
    assert result.from_cache


def test_fetch_media_accepts_windows_style_path(tone_mp3):
    # On Windows, backslash paths must be treated as local, not mangled into
    # https://C:\…
    windows = str(tone_mp3.resolve()).replace("/", "\\")
    if os.name == "nt":
        result = fetch_media(windows, tone_mp3.parent, Settings())
        assert result.path == tone_mp3.resolve()
    else:
        # POSIX: a Windows drive path (with a drive letter) cannot exist locally
        # and must fail cleanly, proving it never reaches URL validation.
        with pytest.raises(InvalidURLError):
            fetch_media(r"C:\Users\willi\Downloads\Jessica Starling", tone_mp3.parent, Settings())


def test_fetch_media_rejects_missing_windows_path(tmp_path):
    with pytest.raises(InvalidURLError):
        _as_local_path(r"C:\Users\willi\Downloads\Jessica Starling")


def test_fetch_media_rejects_directory_path(tmp_path):
    with pytest.raises(InvalidURLError, match="folder"):
        fetch_media(str(tmp_path), tmp_path, Settings())


def test_fetch_media_reports_missing_local_file(tmp_path):
    with pytest.raises(InvalidURLError):
        fetch_media(f"file://{tmp_path}/nope.mp3", tmp_path, Settings())
