"""Guards for the setup scripts.

These run on the user's machine before anything else does, so a syntax-level
mistake here is the first thing a new user sees.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WINDOWS_SCRIPT = REPO / "scripts" / "setup-windows.ps1"
UNIX_SCRIPT = REPO / "scripts" / "setup.sh"

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
            f"scripts/setup-windows.ps1 line {number} contains non-ASCII bytes "
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
