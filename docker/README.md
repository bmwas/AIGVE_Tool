# Docker Setup for AIGVE

This document provides comprehensive Docker commands for running AIGVE with proper memory allocation for all metrics.

## Prerequisites

- Docker installed
- NVIDIA driver + NVIDIA Container Toolkit (for GPU support)
- Sufficient GPU memory (8GB+ recommended for NN-based metrics)

## Quick Start

### Build the Image

```bash
docker build --no-cache -t ghcr.io/bmwas/aigve:latest .
```

## Model Checkpoints (IMPORTANT!)

NN-based metrics (GSTVQA, SimpleVQA, LightVQA+) require model checkpoints. These are **automatically downloaded** at server startup when possible.

### Auto-Downloaded Models ✅

| Model | Source | Auto-Download |
|-------|--------|---------------|
| GSTVQA | Bundled with repo | ✅ Yes |
| SimpleVQA | Google Drive | ✅ Yes (via gdown) |
| LightVQA+ Swin | GitHub Releases | ✅ Yes |

### Optional: LightVQA+ (Manual Download)

**LightVQA+ Main Model** is **OPTIONAL** and requires manual download. All other metrics work without it:

1. **Download from ONE of these sources:**
   - JBOX: https://jbox.sjtu.edu.cn/l/S1bbm1
   - Baidu: https://pan.baidu.com/s/1JZMsibiVDDSQVdrRob1clw (password: `ui9v`)

2. **Mount the file into Docker:**

```bash
docker run --rm \
    --gpus '"device=1"' \
    --memory=16g \
    --shm-size=4g \
    -p 2200:2200 \
    -v "$PWD/uploads":/app/uploads \
    -v "/path/to/last2_SI+TI_epoch_19_SRCC_0.925264.pth":/app/aigve/metrics/video_quality_assessment/nn_based/lightvqa_plus/Light_VQA_plus/ckpts/last2_SI+TI_epoch_19_SRCC_0.925264.pth \
    ghcr.io/bmwas/aigve:latest
```

**Or place in a local `ckpts` directory and mount:**

```bash
mkdir -p ckpts
# Download lightvqa+ model to ckpts/last2_SI+TI_epoch_19_SRCC_0.925264.pth

docker run --rm \
    --gpus '"device=1"' \
    --memory=16g \
    --shm-size=4g \
    -p 2200:2200 \
    -v "$PWD/uploads":/app/uploads \
    -v "$PWD/ckpts":/app/ckpts \
    ghcr.io/bmwas/aigve:latest
```

## Running AIGVE

### ⚡ Standard Run (Distribution Metrics Only - Fast)

For **FID, IS, FVD** metrics only (low memory usage):

```bash
docker run --rm \
    --gpus '"device=1"' \
    -p 2200:2200 \
    -v "$PWD/data":/app/data \
    -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest
```

### 🧠 Full Metrics Run (All Metrics - Requires More Memory)

For **ALL metrics** including NN-based (GSTVQA, SimpleVQA, LightVQA+):

```bash
docker run --rm \
    --gpus '"device=1"' \
    --memory=16g \
    --shm-size=4g \
    -p 2200:2200 \
    -v "$PWD/data":/app/data \
    -v "$PWD/out":/app/out \
    -v "$PWD/uploads":/app/uploads \
    ghcr.io/bmwas/aigve:latest
```

**Memory flags explained:**
- `--memory=16g`: Allocates 16GB RAM to the container (increase if still OOM)
- `--shm-size=4g`: Shared memory for PyTorch DataLoader workers

### 🖥️ GPU Device Selection

| Device | Command |
|--------|---------|
| GPU 0 (Blackwell/Primary) | `--gpus '"device=0"'` |
| GPU 1 (Secondary) | `--gpus '"device=1"'` |
| All GPUs | `--gpus all` |

## Running Metrics by Category

### Option 1: Distribution Metrics Only (Fast - ~30 seconds)

```bash
python scripts/call_aigve_api.py \
    --reference-video /path/to/reference.mp4 \
    --generated-video /path/to/generated.mp4 \
    --categories distribution_based
```

### Option 2: NN-Based Metrics Only (Slower - ~2-5 minutes)

```bash
python scripts/call_aigve_api.py \
    --reference-video /path/to/reference.mp4 \
    --generated-video /path/to/generated.mp4 \
    --categories nn_based_video
```

### Option 3: All Metrics (Requires More Memory)

First, start the container with increased memory:
```bash
docker run --rm \
    --gpus '"device=1"' \
    --memory=16g \
    --shm-size=4g \
    -p 2200:2200 \
    -v "$PWD/data":/app/data \
    -v "$PWD/out":/app/out \
    -v "$PWD/uploads":/app/uploads \
    ghcr.io/bmwas/aigve:latest
```

Then run with the new `--all-metrics` flag:
```bash
python scripts/call_aigve_api.py \
    --reference-video /path/to/reference.mp4 \
    --generated-video /path/to/generated.mp4 \
    --all-metrics
```

This computes:
- **Distribution-based**: FID, IS, FVD (AIGVE native)
- **NN-based video**: GSTVQA, SimpleVQA, LightVQA+
- **CD-FVD**: I3D + VideoMAE (automatic)

Or specify categories manually:
```bash
python scripts/call_aigve_api.py \
    --reference-video /path/to/reference.mp4 \
    --generated-video /path/to/generated.mp4 \
    --categories distribution_based,nn_based_video
```

## Memory Optimization Tips

### If You Get OOM (Out of Memory) Errors

1. **Reduce max-seconds** (fewer frames = less memory):
   ```bash
   python scripts/call_aigve_api.py \
       --reference-video /path/to/reference.mp4 \
       --generated-video /path/to/generated.mp4 \
       --categories distribution_based,nn_based_video \
       --max-seconds 5
   ```

2. **Run categories separately**:
   ```bash
   # First run distribution metrics
   python scripts/call_aigve_api.py \
       --reference-video /path/to/reference.mp4 \
       --generated-video /path/to/generated.mp4 \
       --categories distribution_based

   # Then run NN-based metrics
   python scripts/call_aigve_api.py \
       --reference-video /path/to/reference.mp4 \
       --generated-video /path/to/generated.mp4 \
       --categories nn_based_video
   ```

3. **Increase Docker memory limit**:
   ```bash
   docker run --rm \
       --gpus '"device=1"' \
       --memory=24g \
       --shm-size=8g \
       -p 2200:2200 \
       ghcr.io/bmwas/aigve:latest
   ```

## Docker Compose

For production deployments, use docker-compose with memory limits:

```yaml
version: '3.8'
services:
  aigve:
    image: ghcr.io/bmwas/aigve:latest
    ports:
      - "2200:2200"
    volumes:
      - ./data:/app/data
      - ./out:/app/out
      - ./uploads:/app/uploads
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    shm_size: '4gb'
    user: "${UID:-1000}:${GID:-1000}"
```

Start with:
```bash
docker-compose up -d aigve
```

## Troubleshooting

### Error: Return code -9 (OOM Killed)

This indicates the container ran out of memory during NN-based metric computation.

**Solutions:**
1. Increase `--memory` flag (try 24g or 32g)
2. Run distribution and NN metrics separately
3. Reduce `--max-seconds` to process fewer frames

### Error: CUDA out of memory

This indicates GPU memory exhaustion.

**Solutions:**
1. Use a single GPU with more VRAM
2. Reduce video resolution before processing
3. Process shorter video clips

### Metrics Not Computed

If some metrics are skipped, check:
1. Model checkpoints are downloaded (see `scripts/download_model_checkpoints.py`)
2. Container has sufficient memory
3. Logs for specific error messages

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Health check with CUDA info |
| `GET /help` | CLI help |
| `POST /run` | Run with server-side paths |
| `POST /run_upload` | Upload videos and run metrics |

## Quick Test

```bash
# Check server health
curl http://localhost:2200/healthz

# Compute ALL metrics using Python client (recommended)
python scripts/call_aigve_api.py \
    --reference-video reference.mp4 \
    --generated-video generated.mp4 \
    --all-metrics

# Or using curl to upload and compute all metrics
curl -X POST http://localhost:2200/run_upload \
    -F "videos=@reference.mp4" \
    -F "videos=@generated.mp4" \
    -F "compute=true" \
    -F "categories=distribution_based,nn_based_video" \
    -F "use_cdfvd=true" \
    -F "max_seconds=8"
```

## All Metrics Summary

| Category | Metrics | Default | Flag Required |
|----------|---------|---------|---------------|
| Distribution-based | FID, IS, FVD | ✅ Yes | - |
| NN-based video | GSTVQA, SimpleVQA, LightVQA+ | ❌ No | `--categories nn_based_video` |
| CD-FVD | I3D, VideoMAE | ✅ Yes | Automatic |

**To compute ALL metrics, use `--all-metrics` flag.**

