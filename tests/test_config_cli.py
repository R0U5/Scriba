from pathlib import Path

import pytest
from typer.testing import CliRunner

from rosetta.cli import app
from rosetta.config import load_settings
from rosetta.errors import ConfigError

runner = CliRunner()


# ------------------------------------------------------------------ config
def test_defaults_are_usable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.port == 8420
    assert settings.engine == "faster-whisper"
    assert settings.output_dir.name == "transcripts"


def test_toml_file_is_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rosetta.toml").write_text(
        '[rosetta]\nengine = "parakeet"\nport = 9100\nkeep_audio = true\n'
    )
    settings = load_settings()
    assert settings.engine == "parakeet"
    assert settings.port == 9100
    assert settings.keep_audio is True


def test_environment_beats_file_and_kwargs_beat_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rosetta.toml").write_text('engine = "parakeet"\n')
    monkeypatch.setenv("ROSETTA_ENGINE", "canary-qwen")
    monkeypatch.setenv("ROSETTA_KEEP_AUDIO", "yes")
    assert load_settings().engine == "canary-qwen"
    assert load_settings().keep_audio is True
    assert load_settings(engine="stub").engine == "stub"


def test_invalid_boolean_environment_value_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROSETTA_KEEP_AUDIO", "perhaps")
    with pytest.raises(ConfigError):
        load_settings()


def test_ensure_dirs_creates_everything(tmp_path):
    settings = load_settings(
        output_dir=tmp_path / "o", work_dir=tmp_path / "w", data_dir=tmp_path / "d"
    )
    settings.ensure_dirs()
    assert all(Path(p).is_dir() for p in (settings.output_dir, settings.work_dir, settings.data_dir))


# --------------------------------------------------------------------- cli
def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "RosettaAI" in result.stdout


def test_models_lists_engines():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "faster-whisper" in result.stdout
    assert "Parakeet" in result.stdout


def test_doctor_reports_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffmpeg" in result.stdout
    assert "Output folder" in result.stdout


def test_transcribe_writes_files(tmp_path, monkeypatch, tone_mp3):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["transcribe", str(tone_mp3), "--engine", "stub", "--format", "txt,srt",
         "--output", str(tmp_path / "out"), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    written = [Path(line) for line in result.stdout.strip().splitlines()]
    assert {p.suffix for p in written} == {".txt", ".srt"}
    assert all(p.exists() for p in written)
    assert "Stub segment" in written[0].read_text()


def test_transcribe_rejects_unknown_format(tmp_path, tone_mp3):
    result = runner.invoke(app, ["transcribe", str(tone_mp3), "--format", "docx"])
    assert result.exit_code == 2
    assert "Unknown output format" in result.stdout


def test_transcribe_reports_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "ftp://example.com/a.mp3", "--engine", "stub"])
    assert result.exit_code == 1
    assert "Unsupported URL scheme" in result.stdout


def test_history_is_empty_at_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROSETTA_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "No transcriptions yet" in result.stdout


# ----------------------------------------------------- the default command ---
def test_bare_command_starts_the_server(monkeypatch, tmp_path):
    """`rosetta` with no arguments is what run.bat runs, so it must work.

    Invoking the Typer command object directly used to hand it OptionInfo
    sentinels instead of defaults, and every settings field failed validation.
    """
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr("rosetta.server.run.run_server", lambda settings, **kw: captured.update(
        host=settings.host, port=settings.port, output_dir=settings.output_dir
    ))

    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert isinstance(captured["host"], str)
    assert isinstance(captured["port"], int)
    assert isinstance(captured["output_dir"], Path)
    assert "RosettaAI" in result.stdout


def test_serve_command_and_bare_command_agree(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(
        "rosetta.server.run.run_server", lambda settings, **kw: seen.append(settings.port)
    )
    assert runner.invoke(app, []).exit_code == 0
    assert runner.invoke(app, ["serve"]).exit_code == 0
    assert seen[0] == seen[1]


def test_busy_port_falls_back_instead_of_crashing(monkeypatch, tmp_path):
    """A second copy, or a stale one, must not end in an address-in-use trace."""
    import socket

    monkeypatch.chdir(tmp_path)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        busy_port = taken.getsockname()[1]

        chosen = {}
        monkeypatch.setattr(
            "rosetta.server.run.run_server", lambda settings, **kw: chosen.update(port=settings.port)
        )
        result = runner.invoke(app, ["serve", "--port", str(busy_port)])

    assert result.exit_code == 0, result.output
    assert chosen["port"] != busy_port
    assert "already in use" in result.stdout
