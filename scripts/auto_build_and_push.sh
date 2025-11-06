#!/bin/bash
set -e

echo "🚀 MoDA Auto Build and Push Script"
echo "===================================="
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
            echo "📋 Auto-detected version from runpod_config.yaml: ${TAG}"
        else
            TAG="latest"
            echo "⚠️  Version not found in runpod_config.yaml, using 'latest'"
        fi
    else
        TAG="latest"
        echo "⚠️  runpod_config.yaml not found, using 'latest'"
    fi
else
    echo "📋 Using explicit TAG: ${TAG}"
fi

FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Building and pushing: ${FULL_IMAGE}"
echo ""

# Check if we're in the MoDA directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Not in MoDA directory!${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    exit 1
fi

# Build
echo "📦 Building image..."
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t "${FULL_IMAGE}" \
  --build-arg BUILDDATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="${TAG}" \
  --build-arg HUGGINGFACE_USERNAME="${HUGGINGFACE_USERNAME}" \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  -f Dockerfile \
  .

echo ""
echo -e "${GREEN}✅${NC} Build complete"
echo ""

# Quick test
echo "🧪 Running quick test..."
if docker run --rm --platform linux/amd64 "${FULL_IMAGE}" python3 -c "
import sys
sys.path.insert(0, '/app/src')
from models.inference.moda_test import LiveVASAPipeline
print('✅ Quick test passed')
" > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} Quick test passed"
else
    echo -e "${RED}❌${NC} Quick test failed"
    exit 1
fi

# Push
echo ""
echo "🚀 Pushing to Docker Hub..."
docker push "${FULL_IMAGE}"

echo ""
echo -e "${GREEN}✅ Complete!${NC}"
echo ""
echo "🔗 Docker Hub: https://hub.docker.com/r/${DOCKER_USER}/${IMAGE_NAME}"
echo "📦 Image: ${FULL_IMAGE}"

