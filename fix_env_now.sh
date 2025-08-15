#!/bin/bash
# Quick fix script to install ALL dependencies in aigve env

echo "=== Fixing aigve environment with ALL dependencies ==="

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch (GPU build)..."
conda install -n aigve -y -c pytorch -c nvidia \
  pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8

# Install NumPy and SciPy
echo "Installing NumPy and SciPy..."
conda install -n aigve -y -c conda-forge \
  numpy=1.26.4 scipy=1.11.4

# Install other core dependencies
echo "Installing other dependencies..."
conda run -n aigve pip install \
  opencv-python-headless \
  charset-normalizer \
  chardet \
  onnx==1.14.1 \
  "protobuf>=4.23.4,<4.24"

# Verify installation
echo ""
echo "=== Verifying installation ==="
conda run -n aigve python - <<'EOF'
import sys
print("Python:", sys.executable)
try:
    import torch
    print("✓ torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("  cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("✗ torch:", e)
try:
    import numpy
    print("✓ numpy:", numpy.__version__)
except Exception as e:
    print("✗ numpy:", e)
try:
    import scipy
    print("✓ scipy:", scipy.__version__)
except Exception as e:
    print("✗ scipy:", e)
EOF

echo ""
echo "Done! Now you can run:"
echo "  conda activate aigve"
echo "  python -m uvicorn server.main:app --host 0.0.0.0 --port 2200"
