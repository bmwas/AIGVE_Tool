# syntax=docker/dockerfile:1.6
# Base: NVIDIA CUDA 11.8 on Ubuntu 22.04 (using Docker Hub)
ARG BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04
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
# Install cd-fvd as root to create package structure
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

# RADICAL FIX: Create user and dedicated model directory with proper permissions  
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    # Create all required directories
    mkdir -p /app/models/cdfvd/third_party/VideoMAEv2 && \
    mkdir -p /app/models/cdfvd/third_party/i3d && \
    mkdir -p /app/uploads && \
    mkdir -p /app/.cache/huggingface/hub && \
    mkdir -p /app/.cache/torch && \
    # Install curl for downloading
    apt-get update && apt-get install -y curl && \
    # Download CD-FVD model files directly to our dedicated directory  
    cd /app/models/cdfvd/third_party && \
    curl -L -o VideoMAEv2/vit_g_hybrid_pt_1200e_ssv2_ft.pth https://huggingface.co/OpenGVLab/VideoMAEv2/resolve/main/vit_g_hybrid_pt_1200e_ssv2_ft.pth && \
    curl -L -o i3d/i3d_pretrained_400.pt https://github.com/piergiaj/pytorch-i3d/releases/download/v0.1/i3d_pretrained_400.pt && \
    # Verify downloads
    ls -la VideoMAEv2/ && ls -la i3d/ && \
    # Remove original dist-packages third_party if it exists and create symlink
    rm -rf /usr/local/lib/python3.10/dist-packages/cdfvd/third_party && \
    ln -s /app/models/cdfvd/third_party /usr/local/lib/python3.10/dist-packages/cdfvd/third_party && \
    # Set proper ownership for user 1000:1000  
    chown -R 1000:1000 /app && \
    chmod -R 755 /app/models && \
    chmod -R 777 /app/uploads /app/.cache && \
    # Clean up apt cache
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for cache and model directories
ENV HOME=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface  
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV CDFVD_MODEL_DIR=/app/models/cdfvd/third_party

# Create uploads directory with proper permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Install requirements (excluding cd-fvd which will be installed via git clone)
USER root
RUN pip3 install --no-cache-dir -r /app/requirement.txt

# Install cd-fvd via git clone as required
RUN git clone https://github.com/songweige/content-debiased-fvd.git && \
    cd ./content-debiased-fvd && \
    pip install -e . && \
    cd / && rm -rf /app/content-debiased-fvd && \
    pip3 show cd-fvd

# Copy and set up entrypoint script  
RUN chmod +x /app/entrypoint.sh

# Switch to user 1000 and verify directory access works
USER 1000
RUN touch /app/.cache/test_write && rm /app/.cache/test_write && \
    touch /app/uploads/test_write && rm /app/uploads/test_write && \
    echo "Directory access tests passed for user 1000"

# Default entrypoint
EXPOSE 2200
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]