#!/bin/bash
set -e

echo "⚡ MoDA Quick Push Script"
echo "=============================="
echo ""

# Configuration
IMAGE_NAME="moda-runpod"
DOCKER_USER="${DOCKER_USER}"

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

# Auto-login if DOCKER_TOKEN is set
if [ -n "$DOCKER_TOKEN" ]; then
    echo "🔐 Logging in to Docker Hub..."
    if echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Logged in successfully"
    else
        echo -e "${RED}❌${NC} Docker login failed"
        exit 1
    fi
    echo ""
fi

echo "🚀 Quick push for: ${FULL_IMAGE}"
echo ""

# Check if image exists locally
if ! docker images "${FULL_IMAGE}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${FULL_IMAGE}"; then
    echo -e "${RED}❌ Image ${FULL_IMAGE} not found locally!${NC}"
    echo "   Run ./scripts/build_docker.sh first"
    exit 1
fi

echo -e "${GREEN}✅${NC} Image found locally"
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
if docker push "${FULL_IMAGE}"; then
    echo ""
    echo -e "${GREEN}✅ Push successful!${NC}"
    echo ""
    echo "🔗 Docker Hub: https://hub.docker.com/r/${DOCKER_USER}/${IMAGE_NAME}"
    echo "📦 Image: ${FULL_IMAGE}"
else
    echo -e "${RED}❌ Push failed!${NC}"
    exit 1
fi

