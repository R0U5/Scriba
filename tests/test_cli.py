import subprocess
from pathlib import Path

def test_version_flag(tmp_path: Path) -> None:
    """Ensure ``scriba --version`` prints the version without error."""
    # Install the package in editable mode into a temporary venv
    # (skip installation here – we invoke the module directly).
    result = subprocess.run(
        ["python3", "-m", "scriba", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Scriba" in result.stdout

