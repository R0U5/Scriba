#!/usr/bin/env bash
# Sets RosettaAI up on Linux or macOS.
#
#   ./scripts/setup.sh            # core app + Whisper engine
#   ./scripts/setup.sh --nvidia   # also install the NeMo engines (large)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python="${PYTHON:-python3}"
"$python" --version

[ -d .venv ] || { echo "Creating virtual environment (.venv)…"; "$python" -m venv .venv; }
venv_python="$root/.venv/bin/python"

"$venv_python" -m pip install --upgrade pip --quiet
echo "Installing RosettaAI + Whisper engine…"
"$venv_python" -m pip install -e ".[whisper]"

if [ "${1:-}" = "--nvidia" ]; then
  echo "Installing PyTorch (CUDA 12.4) + NeMo engines — this is a large download…"
  "$venv_python" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  "$venv_python" -m pip install -e ".[nvidia]"
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected: installing CUDA runtime libraries for CTranslate2…"
  "$venv_python" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
fi

echo
"$venv_python" -m rosetta doctor
echo
echo "Done. Start RosettaAI with:  .venv/bin/rosetta"
