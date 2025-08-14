#!/usr/bin/env bash
set -euo pipefail

# Entry point for AIGVE container
# - Ensures conda is available
# - Runs scripts/prepare_annotations.py inside the 'aigve' conda env
# - Forwards all CLI arguments to the script
#
# Examples:
#   docker run --rm --gpus all -v "$PWD/data":/app/data aigve:latest \
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

# Help passthrough
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "[INFO] Showing help for scripts/prepare_annotations.py"
  exec conda run -n aigve python scripts/prepare_annotations.py --help
fi

# Run the preparation / evaluation script with provided arguments
exec conda run -n aigve python scripts/prepare_annotations.py "$@"
