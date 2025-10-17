#!/usr/bin/env bash
set -euo pipefail

# Entry point for AIGVE container (non-conda version)
# - Runs scripts/prepare_annotations.py with system Python
# - Forwards all CLI arguments to the script

# Resolve working directory
cd /app

# Ensure uploads directory exists with proper permissions
if [ ! -d "/app/uploads" ]; then
  echo "[INFO] Creating /app/uploads directory..."
  mkdir -p /app/uploads
fi

# Ensure we can write to uploads directory
if [ ! -w "/app/uploads" ]; then
  echo "[WARN] /app/uploads is not writable by current user ($(id -u))"
  echo "       Attempting to fix permissions..."
  # Try to fix if running as root, otherwise just warn
  if [ "$(id -u)" -eq 0 ]; then
    chown -R 1000:1000 /app/uploads
    chmod 755 /app/uploads
    echo "[INFO] Fixed permissions as root"
  else
    echo "[ERROR] Cannot fix permissions - running as non-root user $(id -u)"
    echo ""
    echo "This is likely due to volume mount ownership issues."
    echo ""
    echo "SOLUTION 1: Fix host directory ownership (recommended)"
    echo "  On the host machine, run:"
    echo "    sudo chown $(id -u):$(id -g) ./uploads"
    echo "    sudo chmod 755 ./uploads"
    echo "    docker-compose restart aigve"
    echo ""
    echo "SOLUTION 2: Run pre-startup script"
    echo "  bash docker-compose-pre-start.sh"
    echo "  docker-compose up -d --force-recreate"
    echo ""
    echo "The API will continue starting but uploads will FAIL."
  fi
fi

# GPU detection and enforcement (default: require GPU)
REQUIRE_GPU="${REQUIRE_GPU:-1}"

# If CUDA_VISIBLE_DEVICES is set to an empty string, it masks all GPUs.
# Unset it to allow default (all devices) visibility.
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "[WARN] CUDA_VISIBLE_DEVICES is empty; unsetting to allow GPU visibility."
  unset CUDA_VISIBLE_DEVICES
fi

# Test CUDA availability
echo "[INFO] Testing CUDA availability..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" || true

if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "[FATAL] GPU required but CUDA not available" >&2
    echo "        Ensure you run with: docker run --gpus all ..." >&2
    nvidia-smi -L || echo "nvidia-smi not available" >&2
    exit 1
  fi
fi

# If first arg is a direct command, execute it
if [[ "$1" == "python3" || "$1" == "python" || "$1" == "bash" || "$1" == "sh" ]]; then
  exec "$@"
fi

# Serve API by default or when explicitly requested
PORT="${PORT:-2200}"
if [[ $# -eq 0 || "$1" == "api" ]]; then
  if [[ $# -gt 0 ]]; then shift; fi
  echo "[INFO] Starting API server on 0.0.0.0:${PORT}"
  exec uvicorn server.main:app --host 0.0.0.0 --port "${PORT}" "$@"
fi

# Help passthrough
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  exec python3 scripts/prepare_annotations.py --help
fi

# Otherwise, treat args as CLI
echo "[INFO] Running CLI: scripts/prepare_annotations.py $*"
exec python3 scripts/prepare_annotations.py "$@"
