#!/bin/bash
set -e

echo "🔨 Simple MoDA Docker Build"
echo "==========================="
echo ""

# Configuration
IMAGE_NAME="moda-runpod"
DOCKER_USER="${DOCKER_USER:-krapiunitski12}"
TAG="${TAG:-latest}"
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

