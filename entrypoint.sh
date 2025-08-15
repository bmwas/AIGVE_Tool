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
