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
pip install --pre torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Everything else from PyPI. ultralytics must NOT drag in its own torch:
# install with --no-deps and add its remaining deps explicitly.
pip install numpy opencv-python pillow pyyaml matplotlib pandas \
  pycocotools pytest tqdm psutil "polars" py-cpuinfo
pip install --no-deps ultralytics

pip install -e .

echo
echo "Setup done. Now verify the GPU end-to-end:"
echo "  python scripts/phase0_check_gpu.py"
