#!/usr/bin/env bash
set -euo pipefail

# AIGVE environment setup script
# This script creates a conda env, installs PyTorch (GPU-only),
# enforces a CUDA-enabled build (fails if CPU-only), installs ONNX/protobuf,
# and installs project requirements without
# touching the conda-installed torch packages.
#
# Usage:
#   bash setup_env.sh [--env-name aigve] [--with-nlp]
# Examples:
#   bash setup_env.sh                      # GPU install (CUDA 11.8 runtime via conda)
#   bash setup_env.sh --env-name myenv     # GPU install into custom env name
#   bash setup_env.sh --with-nlp           # Also install transformers + compatible tokenizers

ENV_NAME="aigve"
CPU_ONLY=0
NLP_EXTRAS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"; shift; shift ;;
    --cpu)
      echo "[FATAL] CPU-only install is disabled. This project enforces a GPU-enabled PyTorch build." >&2
      echo "        Please provision an NVIDIA GPU environment and do not pass --cpu." >&2
      exit 1 ;;
    --with-nlp)
      NLP_EXTRAS=1; shift ;;
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

# 1) Re-create env from environment.yml (includes fastapi, uvicorn, etc)
# Remove if exists
echo "Removing existing env: $ENV_NAME (if present)..."
conda env remove -n "$ENV_NAME" -y || true

# Use the provided environment.yml (has fastapi, uvicorn, and other deps)
echo "Creating fresh env: $ENV_NAME from environment.yml..."
conda env create -n "$ENV_NAME" -f environment.yml

# Ensure pip deps from environment.yml are installed
echo "Installing pip dependencies from environment.yml..."
conda run -n "$ENV_NAME" pip install -U pip
conda run -n "$ENV_NAME" pip install fastapi "uvicorn[standard]" pydantic typing-extensions

# 2) Install PyTorch (GPU build enforced)
# Remove any pip-installed torch packages the YAML may have pulled
echo "Cleaning up any pip-installed torch packages..."
conda run -n "$ENV_NAME" pip uninstall -y torch torchvision torchaudio || true

echo "Installing GPU PyTorch (CUDA 11.8 runtime) into env: $ENV_NAME"
# Ensure channels are prioritized correctly and force reinstall
conda install -n "$ENV_NAME" -y -c pytorch -c nvidia \
  pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 \
  --force-reinstall --override-channels

# 2b) Enforce GPU build (do not require a runtime GPU during build)
conda run -n "$ENV_NAME" python - << 'PY'
import sys
try:
    import torch
except Exception as e:
    print('FATAL: torch not importable after installation:', e, file=sys.stderr)
    sys.exit(1)
cuda_build = getattr(getattr(torch, 'version', None), 'cuda', None)
if not cuda_build:
    print('FATAL: CPU-only PyTorch was installed. A GPU-enabled build is required.', file=sys.stderr)
    sys.exit(1)
print('Verified GPU-enabled PyTorch build:', 'torch', torch.__version__, 'CUDA', cuda_build)
PY

# 3) Install ONNX/protobuf combo compatible with Torch 2.1
conda run -n "$ENV_NAME" pip install "onnx==1.14.1" "protobuf>=4.23.4,<4.24"

# 4) Fix requests warning (charset detection)
conda run -n "$ENV_NAME" pip install charset-normalizer chardet

# 5) Ensure OpenCV (for cv2 usage in datasets)
conda run -n "$ENV_NAME" pip install opencv-python-headless

# 5b) Remove known conflicting extras not needed for FID/IS/FVD
# (These may pull incompatible transformers or heavy deps.)
conda run -n "$ENV_NAME" pip uninstall -y vbench mantis mantis-vl || true

# 5c) Ensure consistent numeric stack (NumPy/SciPy) via conda to avoid ABI mismatches
echo "Installing NumPy/SciPy via conda-forge..."
# First remove any pip wheels that may have been installed from environment.yml
conda run -n "$ENV_NAME" pip uninstall -y numpy scipy || true
# Then remove conda records to force relinking in case pip deleted files under the hood
conda remove -n "$ENV_NAME" -y numpy scipy || true
# Finally, install conda-forge builds known to work well with PyTorch 2.1 and SciPy stack
conda install -n "$ENV_NAME" -y -c conda-forge --force-reinstall "numpy==1.26.4" "scipy==1.11.4"

# 6) Install remaining requirements WITHOUT touching torch packages
# - Filters out top-level torch/torchvision/torchaudio/pytorch pins
# - Skips Mantis git package (not required for FID/IS/FVD)
REQ_FILE="requirement.txt"
if [[ -f "$REQ_FILE" ]]; then
  echo "Installing requirements from requirement.txt (excluding torch/numpy/scipy)..."
  TMP_REQ="$(mktemp)"
  awk 'BEGIN{IGNORECASE=0} \
       /^(pytorch|torch|torchvision|torchaudio|numpy|scipy)[[:space:]=]/ {next} \
       /Mantis\.git/ {next} \
       {print}' "$REQ_FILE" > "$TMP_REQ"
  # Install with dependencies this time to get all transitive deps
  conda run -n "$ENV_NAME" pip install -r "$TMP_REQ"
  rm -f "$TMP_REQ"
else
  echo "No requirement.txt found; skipping pip requirements."
fi

# 6-extra) Explicitly install critical packages that might be missing
echo "Installing critical packages explicitly..."
conda run -n "$ENV_NAME" pip install \
  opencv-python-headless \
  pillow \
  imageio \
  imageio-ffmpeg \
  protobuf \
  onnx \
  onnxruntime \
  einops \
  decord \
  pandas \
  matplotlib \
  seaborn \
  tqdm \
  pyyaml \
  jsonschema \
  h5py \
  scikit-learn \
  scikit-image

# 6a) Ensure API server dependencies are installed
echo "Ensuring API server dependencies..."
conda run -n "$ENV_NAME" pip install \
  "fastapi>=0.68.0" \
  "uvicorn[standard]>=0.15.0" \
  "pydantic>=1.8.0" \
  "python-multipart" \
  "httptools" \
  "uvloop" \
  "websockets" \
  "watchfiles" \
  "python-dotenv" \
  "requests" \
  "charset-normalizer"

# 6b) Optional NLP extras (transformers + compatible tokenizers)
if [[ "$NLP_EXTRAS" -eq 1 ]]; then
  echo "Installing NLP extras: transformers + tokenizers (via conda-forge)"
  if ! conda install -n "$ENV_NAME" -y -c conda-forge "transformers>=4.44.0" "tokenizers>=0.20,<0.21" safetensors accelerate; then
    echo "[WARN] Could not install transformers/tokenizers via conda. Skipping NLP extras."
    echo "       If you need text-video metrics, try:"
    echo "         conda install -n $ENV_NAME -c conda-forge transformers tokenizers safetensors accelerate"
  fi
fi

# 6c) If transformers is already present (from environment.yml or requirements),
# ensure tokenizers is compatible to avoid runtime import errors.
if conda run -n "$ENV_NAME" python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('transformers') else 1)
PY
then
  echo "transformers detected in env; ensuring compatible tokenizers (>=0.20,<0.21)"
  conda install -n "$ENV_NAME" -y -c conda-forge "tokenizers>=0.20,<0.21" || echo "[WARN] tokenizers install skipped (conda failed). If needed, install manually."
fi

# 7) Quick sanity checks
echo "\nPerforming sanity checks..."
conda run -n "$ENV_NAME" python - << 'PY'
import sys
print('Python:', sys.version)
try:
    import torch
    print('torch:', torch.__version__, 'CUDA:', getattr(torch.version, 'cuda', None), 'cuda_available:', torch.cuda.is_available())
except Exception as e:
    print('Torch import failed:', e)
try:
    import torchvision
    print('torchvision:', torchvision.__version__)
except Exception as e:
    print('Torchvision import failed:', e)
try:
    import torchaudio
    print('torchaudio:', torchaudio.__version__)
except Exception as e:
    print('Torchaudio import failed:', e)
try:
    import onnx
    print('onnx:', onnx.__version__)
except Exception as e:
    print('ONNX import failed:', e)
try:
    import numpy, scipy
    print('numpy:', numpy.__version__, 'scipy:', scipy.__version__)
except Exception as e:
    print('NumPy/SciPy check failed:', e)
try:
    import cv2
    print('opencv-python-headless (cv2):', cv2.__version__)
except Exception as e:
    print('OpenCV import failed:', e)
try:
    import fastapi, uvicorn
    print('fastapi:', fastapi.__version__, 'uvicorn:', uvicorn.__version__)
except Exception as e:
    print('FastAPI/Uvicorn import failed:', e)
try:
    import mmengine
    print('mmengine:', mmengine.__version__)
except Exception as e:
    print('mmengine import failed:', e)
try:
    import mmcv
    print('mmcv:', mmcv.__version__)
except Exception as e:
    print('mmcv import failed (ok if not needed):', e)
try:
    import mmdet
    print('mmdet:', mmdet.__version__)
except Exception as e:
    print('mmdet import failed (ok if not needed):', e)
try:
    import importlib.util
    if importlib.util.find_spec('transformers'):
        import transformers
        try:
            import tokenizers
            print('transformers:', transformers.__version__, 'tokenizers:', tokenizers.__version__)
        except Exception as e:
            print('Transformers present but tokenizers check failed:', e)
    else:
        print('transformers: not installed (ok unless using text-video metrics)')
except Exception as e:
    print('Transformers/tokenizers check failed:', e)
try:
    import requests
    try:
        import charset_normalizer as _cn
        print('requests:', requests.__version__, 'charset_normalizer:', _cn.__version__)
    except Exception:
        import chardet as _cd
        print('requests:', requests.__version__, 'chardet:', _cd.__version__)
except Exception as e:
    print('Requests import failed:', e)
PY

# 7b) Enforce NumPy/SciPy presence (fatal if missing)
conda run -n "$ENV_NAME" python - << 'PY'
import sys
try:
    import numpy as _np, scipy as _sp
    print('Verified: NumPy/SciPy importable')
except Exception as e:
    print('FATAL: NumPy/SciPy not importable in this environment:', e)
    sys.exit(1)
PY

# 7c) Enforce torch + torchvision presence and GPU build (fatal if missing/CPU-only)
conda run -n "$ENV_NAME" python - << 'PY'
import sys
try:
    import torch, torchvision
except Exception as e:
    print('FATAL: torch/torchvision not importable in this environment:', e)
    sys.exit(1)
cuda_build = getattr(getattr(torch, 'version', None), 'cuda', None)
if not cuda_build:
    print('FATAL: torch is a CPU-only build. GPU-enabled PyTorch is required.', file=sys.stderr)
    sys.exit(1)
print('Verified: torch/torchvision importable; torch CUDA build:', cuda_build)
PY

# 7d) Enforce API server deps (fatal if missing)
echo "\nVerifying API server dependencies..."
conda run -n "$ENV_NAME" python - << 'PY'
import sys
try:
    import fastapi, uvicorn, pydantic
    print('Verified: FastAPI/Uvicorn/Pydantic importable')
    print('  fastapi:', fastapi.__version__)
    print('  uvicorn:', uvicorn.__version__)
    print('  pydantic:', pydantic.__version__)
except Exception as e:
    print('FATAL: API server dependencies not importable:', e)
    sys.exit(1)
PY

echo "\n===== Setup complete! ====="
echo "All dependencies installed successfully."
echo ""
echo "To use the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the API server:"
echo "  python -m uvicorn server.main:app --host 0.0.0.0 --port 2200"
echo ""
echo "To run CLI:"
echo "  python aigve/main_aigve.py aigve/configs/fid.py --work-dir ./output_fid"
