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

# GPU detection and enforcement (default: require GPU)
REQUIRE_GPU="${REQUIRE_GPU:-1}"
# If CUDA_VISIBLE_DEVICES is set to an empty string, it masks all GPUs.
# Unset it to allow default (all devices) visibility.
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "[WARN] CUDA_VISIBLE_DEVICES is empty; unsetting to allow GPU visibility."
  unset CUDA_VISIBLE_DEVICES
fi
# Run probe and capture both stdout and stderr without aborting on failure
set +e
TORCH_CUDA_OUTPUT=$(conda run -n aigve python - <<'PY' 2>&1
import json, os, traceback
try:
    import torch
    info = {
        "torch": getattr(torch, "__version__", None),
        "is_built": bool(getattr(getattr(torch, 'backends', None), 'cuda', None) and torch.backends.cuda.is_built()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    # Attempt to load libcuda explicitly
    try:
        import ctypes
        ctypes.CDLL("libcuda.so.1")
        info["libcuda_loaded"] = True
    except Exception as e:
        info["libcuda_loaded"] = False
        info["libcuda_error"] = str(e)
    # Attempt to initialize CUDA to surface error messages
    try:
        torch.cuda.init()
        info["cuda_init_error"] = None
    except Exception as e:
        info["cuda_init_error"] = str(e)
    if info["cuda_available"]:
        info["devices"] = [{"index": i, "name": torch.cuda.get_device_name(i)} for i in range(info["device_count"])]
    else:
        info["devices"] = []
except Exception as e:
    info = {
        "error": str(e),
        "traceback": traceback.format_exc(),
        "cuda_available": False,
    }
print(json.dumps(info))
PY
)
PROBE_RC=$?
set -e
echo "[INFO] CUDA check (rc=${PROBE_RC}): ${TORCH_CUDA_OUTPUT}"
if [[ "$REQUIRE_GPU" == "1" || "$REQUIRE_GPU" == "true" || "$REQUIRE_GPU" == "True" ]]; then
  if [[ ${PROBE_RC} -ne 0 ]] || ! echo "$TORCH_CUDA_OUTPUT" | grep -q '"cuda_available": true'; then
    echo "[FATAL] GPU required but CUDA not available or probe failed." >&2
    echo "        Hints:" >&2
    echo "        - Ensure you run with: docker run --gpus all ..." >&2
    echo "        - Host must have NVIDIA driver + NVIDIA Container Toolkit installed." >&2
    echo "        Diagnostics:" >&2
    command -v nvidia-smi >/dev/null 2>&1 && { echo '--- nvidia-smi -L ---' >&2; nvidia-smi -L >&2; } || echo "[diag] nvidia-smi not found in container PATH" >&2
    command -v nvcc >/dev/null 2>&1 && { echo '--- nvcc --version ---' >&2; nvcc --version >&2; } || echo "[diag] nvcc not found (ok for runtime images)" >&2
    echo '--- /dev/nvidia* ---' >&2
    ls -l /dev/nvidia* 2>&1 || echo "[diag] /dev/nvidia* not present" >&2
    echo '--- Env ---' >&2
    echo "CUDA_HOME=${CUDA_HOME:-}" >&2
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
    echo "PATH=${PATH}" >&2
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
    echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}" >&2
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
