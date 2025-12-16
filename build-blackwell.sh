#!/usr/bin/env bash
# Build script for AIGVE Blackwell GPU Docker image
# This builds a parallel image optimized for RTX 6000 96GB Blackwell GPU

set -euo pipefail

echo "============================================================"
echo "Building AIGVE Docker Image for RTX 6000 96GB Blackwell GPU"
echo "============================================================"
echo ""

# Image name and tag
IMAGE_NAME="${IMAGE_NAME:-aigve-blackwell}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "Image name: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Dockerfile: Dockerfile.blackwell"
echo ""

# Check if Dockerfile.blackwell exists
if [ ! -f "Dockerfile.blackwell" ]; then
    echo "ERROR: Dockerfile.blackwell not found!"
    exit 1
fi

# Build the image
echo "Starting build..."
echo "------------------------------------------------------------"

docker build \
    -f Dockerfile.blackwell \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
    .

echo ""
echo "============================================================"
echo "Build completed successfully!"
echo "============================================================"
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 2200:2200 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Or with docker-compose (if configured):"
echo "  docker-compose -f docker-compose.blackwell.yml up -d"
echo ""

