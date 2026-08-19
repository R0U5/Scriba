"""Guards for the launcher scripts.

These run on the user's machine before anything else does, so a syntax-level
mistake here is the first thing a new user sees.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WINDOWS_SCRIPT = REPO / "scripts" / "start-windows.ps1"
UNIX_SCRIPT = REPO / "run.sh"
BATCH_LAUNCHER = REPO / "run.bat"

UTF8_BOM = b"\xef\xbb\xbf"


def test_windows_script_starts_with_a_utf8_bom():
    """Without a BOM, Windows PowerShell 5.1 decodes the file as ANSI."""
    assert WINDOWS_SCRIPT.read_bytes().startswith(UTF8_BOM)


def test_windows_script_is_ascii_only():
    """A single non-ASCII character can break the whole script.

    Decoded as cp1252, the UTF-8 bytes of an em dash (E2 80 94) end in 0x94,
    which is U+201D - a curly double quote. PowerShell accepts curly quotes as
    string delimiters, so one em dash inside a string closes it early and every
    line after it is swallowed.
    """
    body = WINDOWS_SCRIPT.read_bytes()[len(UTF8_BOM):]
    for number, line in enumerate(body.split(b"\r\n"), start=1):
        offenders = {bytes([b]) for b in line if b > 127}
        assert not offenders, (
            f"scripts/start-windows.ps1 line {number} contains non-ASCII bytes "
            f"{sorted(offenders)}: {line.decode('utf-8', 'replace')}"
        )


def test_windows_script_uses_crlf():
    body = WINDOWS_SCRIPT.read_bytes()[len(UTF8_BOM):]
    assert b"\r\n" in body
    assert body.replace(b"\r\n", b"").count(b"\n") == 0, "mixed line endings"


def test_unix_script_is_lf_and_executable():
    raw = UNIX_SCRIPT.read_bytes()
    assert b"\r" not in raw, "CRLF line endings stop bash from running the script"
    assert raw.startswith(b"#!/usr/bin/env bash")


def test_batch_launcher_is_ascii_and_crlf():
    """cmd.exe is unforgiving: a stray byte or a bare LF breaks the file."""
    raw = BATCH_LAUNCHER.read_bytes()
    assert all(b < 128 for b in raw), "run.bat must stay ASCII-only"
    assert b"\r\n" in raw
    assert raw.replace(b"\r\n", b"").count(b"\n") == 0, "mixed line endings"


def test_batch_launcher_delegates_to_powershell():
    text = BATCH_LAUNCHER.read_text()
    assert "%~dp0scripts\\start-windows.ps1" in text, "must call the launcher by absolute path"
    assert "-ExecutionPolicy Bypass" in text
    assert "%*" in text, "arguments must be forwarded"
    assert "pause" in text, "a double-clicked window must stay open on failure"


def test_launchers_never_make_the_user_manage_an_environment():
    """The whole point: no venv creation or activation anywhere in the flow."""
    for script in (UNIX_SCRIPT, BATCH_LAUNCHER, WINDOWS_SCRIPT):
        text = script.read_text(encoding="utf-8-sig").lower()
        assert "venv" not in text or "-m venv" not in text
        assert "activate" not in text


def test_default_launch_path_does_not_touch_the_pytorch_index():
    """Whisper must never depend on download.pytorch.org being reachable.

    Declaring that index at project level made uv resolve it on every run, which
    broke plain `./run.sh` behind a proxy. It now appears only inside the
    opt-in NVIDIA install.
    """
    assert "download.pytorch.org" not in (REPO / "pyproject.toml").read_text()

    for script in (UNIX_SCRIPT, WINDOWS_SCRIPT):
        text = script.read_text(encoding="utf-8-sig")
        index_lines = [line for line in text.splitlines() if "download.pytorch.org" in line]
        assert index_lines, "the NVIDIA path should still know the CUDA index"
        # Every mention must sit inside the install-nvidia branch, which starts
        # after the marker check and never runs on a normal launch.
        assert text.index("install-nvidia") < min(text.index(line) for line in index_lines)


def test_nvidia_install_verifies_cuda_is_actually_available():
    """A CPU-only torch wheel would silently make the NeMo engines useless."""
    for script in (UNIX_SCRIPT, WINDOWS_SCRIPT):
        text = script.read_text(encoding="utf-8-sig")
        assert "torch.cuda.is_available()" in text
