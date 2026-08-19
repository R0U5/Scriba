#!/usr/bin/env bash
#
# RosettaAI launcher for Linux and macOS.
#
#   ./run.sh                        open the web interface
#   ./run.sh transcribe <url>       transcribe straight from the terminal
#   ./run.sh doctor                 check the setup
#   ./run.sh install-nvidia         add the Parakeet and Canary-Qwen engines
#
# The first run installs everything, including a suitable Python, into a local
# .venv. Later runs start immediately.
set -euo pipefail
cd "$(dirname "$0")"

PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"
NVIDIA_MARKER=".nvidia-enabled"

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return; fi
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [ -x "$candidate" ] && { echo "$candidate"; return; }
  done
}

uv_bin="$(find_uv || true)"
if [ -z "${uv_bin:-}" ]; then
  echo "Installing uv, the package manager RosettaAI uses (one time, ~15 MB)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv_bin="$(find_uv || true)"
  [ -n "${uv_bin:-}" ] || {
    echo "uv could not be installed automatically." >&2
    echo "Install it from https://docs.astral.sh/uv/ and run this script again." >&2
    exit 1
  }
fi

has_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

torch_sees_cuda() {
  "$uv_bin" run --no-sync --quiet python -c \
    "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True
}

# --- opt-in install of the NeMo engines ------------------------------------
if [ "${1:-}" = "install-nvidia" ]; then
  echo "Installing PyTorch and NVIDIA NeMo. This is several GB and runs once."
  if has_gpu; then
    "$uv_bin" sync --extra nvidia --extra cuda
  else
    "$uv_bin" sync --extra nvidia
  fi

  if has_gpu && ! torch_sees_cuda; then
    # PyPI ships a CPU-only torch wheel on some platforms, which would leave
    # these engines running on the CPU without ever saying so.
    echo "PyTorch has no CUDA support; reinstalling it from NVIDIA's index..."
    "$uv_bin" pip install --upgrade torch --index-url "$PYTORCH_CUDA_INDEX"
    if ! torch_sees_cuda; then
      echo >&2
      echo "PyTorch still reports no CUDA device. Parakeet and Canary-Qwen need one." >&2
      echo "Check that your NVIDIA driver is current, then run:" >&2
      echo "    $uv_bin pip install --upgrade --force-reinstall torch --index-url $PYTORCH_CUDA_INDEX" >&2
      exit 1
    fi
  fi

  touch "$NVIDIA_MARKER"
  echo "Done. Parakeet and Canary-Qwen now appear in the model menu."
  exit 0
fi

# --- normal launch ----------------------------------------------------------
if [ -f "$NVIDIA_MARKER" ]; then
  # The environment already holds the NeMo engines (and possibly a hand-placed
  # CUDA build of torch); --no-sync keeps uv from undoing that.
  exec "$uv_bin" run --no-sync --quiet rosetta "$@"
fi

extras=()
has_gpu && extras=(--extra cuda)
exec "$uv_bin" run --quiet "${extras[@]}" rosetta "$@"
