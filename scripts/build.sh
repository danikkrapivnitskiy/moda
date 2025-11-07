#!/bin/bash
set -e

echo "🔨 Simple MoDA Docker Build"
echo "==========================="
echo ""

# Configuration
IMAGE_NAME="moda-runpod"
DOCKER_USER="${DOCKER_USER:-krapiunitski12}"

# Auto-read version from runpod_config.yaml if TAG not set
if [ -z "$TAG" ]; then
    if [ -f "runpod_config.yaml" ]; then
        VERSION=$(grep -A 2 "runpod:" runpod_config.yaml | grep "version:" | awk '{print $2}' | tr -d '"' | tr -d "'")
        if [ -n "$VERSION" ]; then
            TAG="v${VERSION}"
        else
            TAG="latest"
        fi
    else
        TAG="latest"
    fi
fi

FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

echo "Building: ${FULL_IMAGE}"
echo ""

docker buildx build \
  --platform linux/amd64 \
  --load \
  -t "${FULL_IMAGE}" \
  -f Dockerfile \
  .

echo ""
echo "✅ Build complete!"
echo "   Image: ${FULL_IMAGE}"

