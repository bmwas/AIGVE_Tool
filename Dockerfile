# syntax=docker/dockerfile:1.6
# Base: NVIDIA CUDA 12.9 on Ubuntu 22.04 (devel image provides build tools)
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Use bash for RUN steps (needed for conda activation in later layers)
SHELL ["/bin/bash", "-lc"]

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

# ------------------------------
# CUDA env (provided by base image)
# ------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional runtime tunings
ENV TRANSFORMERS_ATTENTION_IMPLEMENTATION=eager
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ------------------------------
# Install Miniconda (conda)
# ------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniconda.sh \
    && conda config --set always_yes yes \
    # Accept Anaconda Terms of Service for non-interactive builds (needed for 'defaults' channel)
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda config --add channels conda-forge \
    && conda config --add channels pytorch \
    && conda update -n base -c defaults conda

# Pre-copy only env files + setup script to leverage Docker layer caching
COPY environment.yml /app/environment.yml
COPY setup_env.sh   /app/setup_env.sh
COPY requirement.txt /app/requirement.txt
RUN chmod +x /app/setup_env.sh

# ------------------------------
# Create project conda env exactly like setup_env.sh (GPU build by default)
# You can switch to CPU build by passing: --build-arg CPU_ONLY=1
# ------------------------------
ARG CPU_ONLY=0
RUN if [[ ${CPU_ONLY} -eq 1 ]]; then \
      echo "[Build] Creating CPU-only env via setup_env.sh"; \
      bash /app/setup_env.sh --env-name aigve --cpu; \
    else \
      echo "[Build] Creating GPU env via setup_env.sh"; \
      bash /app/setup_env.sh --env-name aigve; \
    fi

# Copy the rest of the repository
COPY . /app/

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh || true

# Default entrypoint serves API (see entrypoint.sh). Users can run CLI by passing flags.
#   docker run --gpus all --rm -p 2200:2200 -v "$PWD/data":/app/data ghcr.io/bmwas/aigve:latest
#   docker run --gpus all --rm -v "$PWD/data":/app/data ghcr.io/bmwas/aigve:latest --input-dir /app/data --compute --categories distribution_based --max-seconds 8
EXPOSE 2200
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
