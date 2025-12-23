#!/usr/bin/env bash
set -euo pipefail

# Entry point for AIGVE container (non-conda version)
# - Runs scripts/prepare_annotations.py with system Python
# - Forwards all CLI arguments to the script

# Resolve working directory
cd /app

echo "============================================================"
echo "🔧 PRE-STARTUP PERMISSION CHECK"
echo "============================================================"

# Directories that need to be writable
REQUIRED_DIRS="/app/uploads /app/.cache /app/.cache/torch /app/.cache/torch/hub /app/.cache/torch/hub/checkpoints"

for dir in $REQUIRED_DIRS; do
  if [ ! -d "$dir" ]; then
    echo "[INFO] Creating $dir..."
    mkdir -p "$dir" 2>/dev/null || true
  fi
done

# Ensure uploads directory exists with proper permissions
if [ ! -d "/app/uploads" ]; then
  echo "[INFO] Creating /app/uploads directory..."
  mkdir -p /app/uploads
fi

# Check and fix permissions for all required directories
PERMISSION_OK=true
for dir in $REQUIRED_DIRS; do
  if [ -d "$dir" ] && [ ! -w "$dir" ]; then
    PERMISSION_OK=false
    echo "[WARN] $dir is not writable by current user ($(id -u))"
  fi
done

if [ "$PERMISSION_OK" = false ]; then
  echo "       Attempting to fix permissions..."
  # Try to fix if running as root, otherwise just warn
  if [ "$(id -u)" -eq 0 ]; then
    for dir in $REQUIRED_DIRS; do
      if [ -d "$dir" ]; then
        chown -R 1000:1000 "$dir" 2>/dev/null || true
        chmod -R 755 "$dir" 2>/dev/null || true
      fi
    done
    echo "[INFO] ✅ Fixed permissions as root"
  else
    echo "[ERROR] Cannot fix permissions - running as non-root user $(id -u)"
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PERMISSION FIX REQUIRED (Run on HOST machine first!)    ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Before starting Docker, run:                            ║"
    echo "║                                                          ║"
    echo "║  mkdir -p uploads && chmod 777 uploads                   ║"
    echo "║                                                          ║"
    echo "║  Or with sudo:                                           ║"
    echo "║  sudo chown \$(id -u):\$(id -g) uploads                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "The API will continue starting but uploads may FAIL."
  fi
else
  echo "[INFO] ✅ All directories have correct permissions"
fi

echo "============================================================"

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
    echo "        Ensure you run with: docker run --gpus '\"device=1\"' ..." >&2
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
