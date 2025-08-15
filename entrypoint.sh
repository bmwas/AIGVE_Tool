#!/usr/bin/env bash
set -euo pipefail

# Entry point for AIGVE container
# - Ensures conda is available
# - Runs scripts/prepare_annotations.py inside the 'aigve' conda env
# - Forwards all CLI arguments to the script
#
# Examples:
#   docker run --rm --gpus all -v "$PWD/data":/app/data ghcr.io/bmwas/aigve:latest \
#     --input-dir /app/data --compute --categories distribution_based --max-seconds 8 --fps 25
#

# Ensure conda is on PATH
if ! command -v conda >/dev/null 2>&1; then
  # Attempt to add Miniconda to PATH
  if [[ -d "/opt/conda/bin" ]]; then
    export PATH="/opt/conda/bin:$PATH"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[FATAL] 'conda' not found in PATH inside container." >&2
  exit 1
fi

# Resolve working directory
cd /app

# GPU detection and optional enforcement
REQUIRE_GPU="${REQUIRE_GPU:-0}"
TORCH_CUDA=$(conda run -n aigve python - <<'PY'
import json
try:
    import torch
    info = {
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
except Exception as e:
    info = {"error": str(e), "cuda_available": False}
print(json.dumps(info))
PY
)
echo "[INFO] CUDA check: ${TORCH_CUDA}"
if [[ "$REQUIRE_GPU" == "1" || "$REQUIRE_GPU" == "true" || "$REQUIRE_GPU" == "True" ]]; then
  if ! echo "$TORCH_CUDA" | grep -q '"cuda_available": true'; then
    echo "[FATAL] REQUIRE_GPU=1 set but torch.cuda.is_available() is false. Ensure '--gpus all' at runtime and correct drivers/toolkit." >&2
    exit 1
  fi
fi

# Serve API by default or when explicitly requested
PORT="${PORT:-2200}"
if [[ $# -eq 0 || "$1" == "api" ]]; then
  # Allow passing extra uvicorn args after 'api'
  if [[ $# -gt 0 ]]; then shift; fi
  echo "[INFO] Starting API server on 0.0.0.0:${PORT}"
  exec conda run -n aigve uvicorn server.main:app --host 0.0.0.0 --port "${PORT}" "$@"
fi

# Help passthrough for CLI usage
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "[INFO] Showing help for scripts/prepare_annotations.py"
  exec conda run -n aigve python scripts/prepare_annotations.py --help
fi

# Otherwise, treat args as CLI for prepare_annotations.py
echo "[INFO] Running CLI: scripts/prepare_annotations.py $*"
exec conda run -n aigve python scripts/prepare_annotations.py "$@"
