# AIGVE Docker Image for RTX 6000 96GB Blackwell GPU

This directory contains a parallel Docker image configuration optimized for the **NVIDIA RTX 6000 96GB Blackwell GPU**. This image is separate from the main `Dockerfile` and does not affect your existing setup.

## Key Differences from Main Image

- **CUDA Version**: Uses CUDA 12.4 (required for Blackwell architecture) instead of CUDA 11.8
- **PyTorch Version**: Uses PyTorch 2.4.0 with CUDA 12.4 support instead of PyTorch 2.1.0 with CUDA 11.8
- **Base Image**: `nvidia/cuda:12.4.0-devel-ubuntu22.04` instead of `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Everything Else**: Remains identical to the main image

## Building the Image

### Option 1: Using the Build Script (Recommended)

```bash
./build-blackwell.sh
```

This will build the image with the default name `aigve-blackwell:latest`.

### Option 2: Using Docker Directly

```bash
docker build \
    -f Dockerfile.blackwell \
    -t aigve-blackwell:latest \
    --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
    .
```

### Custom Image Name/Tag

```bash
IMAGE_NAME=my-aigve-blackwell IMAGE_TAG=v1.0 ./build-blackwell.sh
```

## Running the Container

### Option 1: Using Docker Compose (Recommended)

A `docker-compose.blackwell.yml` file is provided for easy deployment:

```bash
# First, ensure directories have correct permissions
bash docker-compose-pre-start.sh

# Build and start the container
docker-compose -f docker-compose.blackwell.yml up -d --build

# View logs
docker-compose -f docker-compose.blackwell.yml logs -f aigve-blackwell

# Stop the container
docker-compose -f docker-compose.blackwell.yml down
```

### Option 2: Using Docker Directly

#### Basic Run

```bash
docker run --gpus '"device=0"' -p 2200:2200 aigve-blackwell:latest
```

#### With Volume Mounts

```bash
docker run --gpus '"device=0"' \
    -p 2200:2200 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/uploads:/app/uploads \
    aigve-blackwell:latest
```

#### Interactive Shell

```bash
docker run --gpus '"device=0"' -it --entrypoint /bin/bash aigve-blackwell:latest
```

## Verifying CUDA and PyTorch

Once the container is running, verify the setup:

```bash
docker exec -it <container-name> python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

You should see:
- PyTorch: 2.4.0
- CUDA Version: 12.4
- CUDA Available: True
- GPU Count: 1 (or more if you have multiple GPUs)

## System Requirements

- **NVIDIA Driver**: Must support CUDA 12.4 (typically driver version 550.54.15 or newer)
- **Docker**: Version 19.03 or newer with NVIDIA Container Toolkit installed
- **GPU**: RTX 6000 96GB Blackwell or compatible Blackwell architecture GPU

## Notes

- This image is completely independent from the main `Dockerfile`
- Both images can coexist on the same system
- The main image (`Dockerfile`) remains unchanged
- All application code, dependencies, and configurations are identical between both images
- Only the CUDA and PyTorch versions differ to support the Blackwell architecture

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Ensure Docker has GPU access: `docker run --rm --gpus '"device=0"' nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
3. Check that the NVIDIA Container Toolkit is installed

### Build Failures

If the build fails:

1. Ensure you have sufficient disk space (image is ~10-15GB)
2. Check your internet connection (downloads model files during build)
3. Verify the base image is available: `docker pull nvidia/cuda:12.4.0-devel-ubuntu22.04`

### PyTorch Version Conflicts

If you encounter PyTorch-related errors, ensure you're using the Blackwell image and not the main image. The requirement.txt file specifies PyTorch 2.1.0, but the Blackwell image installs PyTorch 2.4.0 separately to support CUDA 12.4.

## Support

For issues specific to the Blackwell image, check:
- CUDA compatibility with your GPU
- NVIDIA driver version compatibility
- PyTorch CUDA 12.4 support

For general AIGVE issues, refer to the main project documentation.

