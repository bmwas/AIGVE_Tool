# Quick Start: RTX 6000 96GB Blackwell GPU Image

## Build the Image

```bash
./build-blackwell.sh
```

## Run with Docker Compose

```bash
# Setup directories
bash docker-compose-pre-start.sh

# Start container
docker-compose -f docker-compose.blackwell.yml up -d --build

# Check logs
docker-compose -f docker-compose.blackwell.yml logs -f
```

## Run with Docker

```bash
docker run --gpus '"device=0"' -p 2200:2200 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/uploads:/app/uploads \
    aigve-blackwell:latest
```

## Verify CUDA

```bash
docker exec -it <container-name> python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

Expected output: `CUDA: True, Version: 12.4`

## Files Created

- `Dockerfile.blackwell` - Dockerfile for Blackwell architecture
- `docker-compose.blackwell.yml` - Docker Compose configuration
- `build-blackwell.sh` - Build script
- `BLACKWELL_IMAGE.md` - Full documentation

## Notes

- Original `Dockerfile` remains unchanged
- Both images can coexist
- Only CUDA 12.4 and PyTorch 2.4.0 differ from main image

