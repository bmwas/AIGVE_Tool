# syntax=docker/dockerfile:1.6
# Base: NVIDIA CUDA 11.8 on Ubuntu 22.04 (using NVIDIA registry for better compatibility)
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Working directory
WORKDIR /app

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      software-properties-common \
      build-essential \
      nasm \
      git \
      git-lfs \
      libass-dev \
      nano \
      libvpx-dev \
      libfreetype6-dev \
      libsdl2-dev \
      libvorbis-dev \
      libx264-dev \
      libx265-dev \
      libnuma-dev \
      libmp3lame-dev \
      libopus-dev \
      libfdk-aac-dev \
      zlib1g-dev \
      libssl-dev \
      libavcodec-dev \
      libavformat-dev \
      libavutil-dev \
      libswscale-dev \
      libavfilter-dev \
      autoconf \
      automake \
      cmake \
      libtool \
      yasm \
      libgl1 \
      make \
      gcc \
      g++ \
      pkg-config \
      wget \
      curl \
      unzip \
      vim \
      ninja-build \
      libpng-dev \
      libqhull-dev \
      libp11-kit0 \
      ffmpeg \
      ca-certificates \
      && git lfs install \
      && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip python3-distutils && \
    rm -rf /var/lib/apt/lists/*

# Create python3 symlink if it doesn't exist
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# ------------------------------
# CUDA environment
# ------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional runtime tunings
ENV TRANSFORMERS_ATTENTION_IMPLEMENTATION=eager
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ------------------------------
# Install NumPy FIRST (before PyTorch) to ensure correct version
# ------------------------------
RUN python3 -m pip install --no-cache-dir numpy==1.26.4

# ------------------------------
# Install PyTorch with CUDA 11.8
# ------------------------------
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA installation
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# ------------------------------
# Install core dependencies (numpy already installed)
# ------------------------------
RUN python3 -m pip install --no-cache-dir \
    scipy==1.11.4 \
    opencv-python-headless \
    pillow \
    imageio \
    imageio-ffmpeg \
    "onnx==1.14.1" \
    "protobuf>=4.23.4,<4.24" \
    onnxruntime \
    einops==0.8.1 \
    decord==0.6.0 \
    pandas==2.2.3 \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    jsonschema \
    h5py==3.12.1 \
    scikit-learn \
    scikit-image \
    sympy==1.13.3 \
    sentencepiece==0.2.0 \
    fvcore \
    pytorchvideo \
    gdown==5.2.0 \
    charset-normalizer \
    chardet

# ------------------------------
# Install mmcv, mmdet, mmengine (with numpy constraint)
# ------------------------------
RUN python3 -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    mmcv==2.2.0 \
    mmdet==3.3.0 \
    mmengine==0.10.6

# ------------------------------
# Install transformers and tokenizers (with numpy constraint)
# ------------------------------
RUN python3 -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    transformers==4.46.3 \
    "tokenizers>=0.20,<0.21" \
    safetensors \
    accelerate

# ------------------------------
# Install CLIP from GitHub (with numpy constraint)
# ------------------------------
RUN python3 -m pip install --no-cache-dir "numpy==1.26.4" git+https://github.com/openai/CLIP.git

# ------------------------------
# Install API server dependencies
# ------------------------------
RUN python3 -m pip install --no-cache-dir \
    "fastapi>=0.68.0" \
    "uvicorn[standard]>=0.15.0" \
    "pydantic>=1.8.0" \
    python-multipart \
    httptools \
    uvloop \
    websockets \
    watchfiles \
    python-dotenv \
    requests

# ------------------------------
# Install documentation tools (optional)
# ------------------------------
RUN python3 -m pip install --no-cache-dir \
    mkdocs-material \
    neoteroi-mkdocs \
    mkdocs-macros-plugin \
    mkdocs-jupyter \
    mkdocstrings \
    mkdocs-rss-plugin \
    mkdocs-exclude \
    mkdocstrings[python] || true

# ------------------------------
# Install cd-fvd for CD-FVD metrics
# ------------------------------
RUN python3 -m pip install --no-cache-dir cd-fvd

# ------------------------------
# FINAL: Force downgrade NumPy to 1.26.4 after all installations
# ------------------------------
RUN python3 -m pip install --force-reinstall --no-deps numpy==1.26.4

# Verify NumPy version
RUN python3 -c "import numpy; assert numpy.__version__.startswith('1.26'), f'NumPy {numpy.__version__} != 1.26.x'; print(f'NumPy version: {numpy.__version__}')"

# Copy the repository
COPY . /app/

# Create a simple entrypoint script
RUN echo '#!/usr/bin/env bash' > /app/entrypoint_noconda.sh && \
    echo 'set -euo pipefail' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo '# GPU detection' >> /app/entrypoint_noconda.sh && \
    echo 'REQUIRE_GPU="${REQUIRE_GPU:-1}"' >> /app/entrypoint_noconda.sh && \
    echo 'if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then' >> /app/entrypoint_noconda.sh && \
    echo '  echo "[WARN] CUDA_VISIBLE_DEVICES is empty; unsetting to allow GPU visibility."' >> /app/entrypoint_noconda.sh && \
    echo '  unset CUDA_VISIBLE_DEVICES' >> /app/entrypoint_noconda.sh && \
    echo 'fi' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo '# Test CUDA availability' >> /app/entrypoint_noconda.sh && \
    echo 'echo "[INFO] Testing CUDA availability..."' >> /app/entrypoint_noconda.sh && \
    echo 'python3 -c "import torch; print(f'"'"'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}'"'"')" || true' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo 'if [[ "$REQUIRE_GPU" == "1" ]]; then' >> /app/entrypoint_noconda.sh && \
    echo '  if ! python3 -c "import torch; assert torch.cuda.is_available(), '"'"'CUDA not available'"'"'"; then' >> /app/entrypoint_noconda.sh && \
    echo '    echo "[FATAL] GPU required but CUDA not available" >&2' >> /app/entrypoint_noconda.sh && \
    echo '    echo "        Ensure you run with: docker run --gpus all ..." >&2' >> /app/entrypoint_noconda.sh && \
    echo '    nvidia-smi -L || echo "nvidia-smi not available" >&2' >> /app/entrypoint_noconda.sh && \
    echo '    exit 1' >> /app/entrypoint_noconda.sh && \
    echo '  fi' >> /app/entrypoint_noconda.sh && \
    echo 'fi' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo '# If first arg is a direct command, execute it' >> /app/entrypoint_noconda.sh && \
    echo 'if [[ "$1" == "python3" || "$1" == "python" || "$1" == "bash" || "$1" == "sh" ]]; then' >> /app/entrypoint_noconda.sh && \
    echo '  exec "$@"' >> /app/entrypoint_noconda.sh && \
    echo 'fi' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo '# Serve API by default or when explicitly requested' >> /app/entrypoint_noconda.sh && \
    echo 'PORT="${PORT:-2200}"' >> /app/entrypoint_noconda.sh && \
    echo 'if [[ $# -eq 0 || "$1" == "api" ]]; then' >> /app/entrypoint_noconda.sh && \
    echo '  if [[ $# -gt 0 ]]; then shift; fi' >> /app/entrypoint_noconda.sh && \
    echo '  echo "[INFO] Starting API server on 0.0.0.0:${PORT}"' >> /app/entrypoint_noconda.sh && \
    echo '  exec uvicorn server.main:app --host 0.0.0.0 --port "${PORT}" "$@"' >> /app/entrypoint_noconda.sh && \
    echo 'fi' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo '# Help passthrough' >> /app/entrypoint_noconda.sh && \
    echo 'if [[ "$1" == "-h" || "$1" == "--help" ]]; then' >> /app/entrypoint_noconda.sh && \
    echo '  exec python3 scripts/prepare_annotations.py --help' >> /app/entrypoint_noconda.sh && \
    echo 'fi' >> /app/entrypoint_noconda.sh && \
    echo '' >> /app/entrypoint_noconda.sh && \
    echo '# Otherwise, treat args as CLI' >> /app/entrypoint_noconda.sh && \
    echo 'echo "[INFO] Running CLI: scripts/prepare_annotations.py $*"' >> /app/entrypoint_noconda.sh && \
    echo 'exec python3 scripts/prepare_annotations.py "$@"' >> /app/entrypoint_noconda.sh

RUN chmod +x /app/entrypoint_noconda.sh

# Default entrypoint
EXPOSE 2200
ENTRYPOINT ["/app/entrypoint_noconda.sh"]
CMD ["api"]