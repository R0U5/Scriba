import subprocess
import sys
from pathlib import Path


def test_version_flag(tmp_path: Path) -> None:
    """Ensure ``scriba --version`` prints the version without error."""
    # Use the interpreter running the tests so the check works in any
    # environment (CI, venv, uvx) where the package is importable.
    result = subprocess.run(
        [sys.executable, "-m", "scriba", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Scriba" in result.stdout

