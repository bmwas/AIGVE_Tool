#!/usr/bin/env bash
set -euo pipefail

# AIGVE environment setup script
# This script creates a conda env, installs PyTorch (GPU or CPU),
# installs ONNX/protobuf, and installs project requirements without
# touching the conda-installed torch packages.
#
# Usage:
#   bash setup_env.sh [--env-name aigve] [--cpu]
# Examples:
#   bash setup_env.sh                      # GPU install (CUDA 11.8 runtime via conda)
#   bash setup_env.sh --env-name myenv     # GPU install into custom env name
#   bash setup_env.sh --cpu                # CPU-only install

ENV_NAME="aigve"
CPU_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"; shift; shift ;;
    --cpu)
      CPU_ONLY=1; shift ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found in PATH. Please install Miniconda/Anaconda and ensure 'conda' is available." >&2
  exit 1
fi

HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE_DIR"

# 1) Re-create env from environment.yml (minimal deps; no torch here)
# Remove if exists
conda env remove -n "$ENV_NAME" -y || true

# Use the provided environment.yml
conda env create -n "$ENV_NAME" -f environment.yml

# 2) Install PyTorch
# Remove any pip-installed torch packages the YAML may have pulled
conda run -n "$ENV_NAME" pip uninstall -y torch torchvision torchaudio || true

if [[ "$CPU_ONLY" -eq 1 ]]; then
  echo "Installing CPU-only PyTorch into env: $ENV_NAME"
  conda install -n "$ENV_NAME" -y -c pytorch \
    pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 cpuonly
else
  echo "Installing GPU PyTorch (CUDA 11.8 runtime) into env: $ENV_NAME"
  conda install -n "$ENV_NAME" -y -c pytorch -c nvidia \
    pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8
fi

# 3) Install ONNX/protobuf combo compatible with Torch 2.1
conda run -n "$ENV_NAME" pip install "onnx==1.14.1" "protobuf>=4.23.4,<4.24"

# 4) Fix requests warning (charset detection)
conda run -n "$ENV_NAME" pip install charset-normalizer

# 5) Ensure OpenCV (for cv2 usage in datasets)
conda run -n "$ENV_NAME" pip install opencv-python-headless

# 5b) Remove known conflicting extras not needed for FID/IS/FVD
# (These may pull incompatible transformers or heavy deps.)
conda run -n "$ENV_NAME" pip uninstall -y vbench mantis mantis-vl || true

# 6) Install remaining requirements WITHOUT touching torch packages
# - Filters out top-level torch/torchvision/torchaudio/pytorch pins
# - Skips Mantis git package (not required for FID/IS/FVD)
REQ_FILE="requirement.txt"
if [[ -f "$REQ_FILE" ]]; then
  TMP_REQ="$(mktemp)"
  awk 'BEGIN{IGNORECASE=0} \
       /^(pytorch|torch|torchvision|torchaudio)[[:space:]=]/ {next} \
       /Mantis\\.git/ {next} \
       {print}' "$REQ_FILE" > "$TMP_REQ"
  conda run -n "$ENV_NAME" pip install -r "$TMP_REQ" --no-deps
  rm -f "$TMP_REQ"
else
  echo "No requirement.txt found; skipping pip requirements."
fi

# 7) Quick sanity checks
conda run -n "$ENV_NAME" python - << 'PY'
import sys
print('Python:', sys.version)
try:
    import torch
    print('torch:', torch.__version__, 'CUDA:', getattr(torch.version, 'cuda', None), 'cuda_available:', torch.cuda.is_available())
except Exception as e:
    print('Torch import failed:', e)
try:
    import onnx
    print('onnx:', onnx.__version__)
except Exception as e:
    print('ONNX import failed:', e)
PY

echo "\nSetup complete. Activate the env and run, e.g.:"
echo "  conda activate $ENV_NAME"
echo "  python aigve/main_aigve.py aigve/configs/fid.py --work-dir ./output_fid"
