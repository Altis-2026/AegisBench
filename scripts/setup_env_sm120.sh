#!/usr/bin/env bash
# Environment setup for an RTX 5070 (Blackwell, sm_120) laptop.
#
# Blackwell consumer GPUs need a PyTorch build compiled with CUDA 12.8+.
# Stable wheels that predate sm_120 support will import fine but report
# "CUDA not available" or fail at kernel launch. If the current stable
# wheel already lists sm_120 (check with phase0_check_gpu.py), you can use
# stable instead of nightly — the check script is the source of truth.
#
# Python 3.11 is required (nightly wheels for 3.12 have been spotty).
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# PyTorch nightly with CUDA 12.8 (try cu130 if cu128 wheels disappear).
#
# The nightly torch/torchvision publishing pipelines are independent and
# occasionally drift a day apart, so the index can briefly contain a torch
# build with no matching torchvision (or vice versa), which pip's resolver
# correctly refuses to pair. If the direct install fails, fall back to
# installing torch alone, then torchvision without its exact-version pin —
# one day of nightly drift is essentially always ABI-compatible, and the
# import check right after installation proves it rather than assuming it.
TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
if ! pip install --pre torch torchvision --index-url "$TORCH_INDEX"; then
  echo
  echo "torch/torchvision nightly builds are out of sync on the index today."
  echo "Falling back: install torch, then torchvision without its exact pin."
  pip install --pre torch --index-url "$TORCH_INDEX"
  pip install --pre torchvision --index-url "$TORCH_INDEX" --no-deps
fi

# Everything else from PyPI. ultralytics must NOT drag in its own torch
# (that would clobber the GPU-specific nightly above), so it is installed
# with --no-deps -- which means EVERY other ultralytics runtime dependency
# has to be listed here explicitly. Missing any one of them surfaces only
# at runtime (e.g. ultralytics importing `requests` to fetch pretrained
# weights), so this list mirrors ultralytics' declared deps minus
# torch/torchvision: requests, scipy, ultralytics-thop included.
pip install numpy opencv-python pillow pyyaml matplotlib pandas \
  pycocotools pytest tqdm psutil "polars" py-cpuinfo \
  requests scipy ultralytics-thop
pip install --no-deps ultralytics

pip install -e .

echo
echo "Verifying torch/torchvision import compatibility..."
python -c "import torch, torchvision; \
print('torch', torch.__version__); print('torchvision', torchvision.__version__)"

echo
echo "Setup done. Now verify the GPU end-to-end:"
echo "  python scripts/phase0_check_gpu.py"
