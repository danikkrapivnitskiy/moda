#!/bin/bash
set -e

echo "🐳 Building Docker image for RunPod deployment"
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
NC='\033[0m' # No Color

# Auto-login if DOCKER_TOKEN is set
if [ -n "$DOCKER_TOKEN" ]; then
    echo "🔐 Logging in to Docker Hub..."
    if echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Logged in successfully"
    else
        echo -e "${YELLOW}⚠️${NC} Docker login failed (continuing with build)"
    fi
    echo ""
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    echo "   Please start Docker Desktop and try again"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is running"
echo ""

# Check if we're in the MoDA directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Not in MoDA directory!${NC}"
    echo "   Please run this script from the MoDA directory"
    exit 1
fi

echo -e "${GREEN}✓${NC} MoDA directory confirmed"
echo ""

# Build image
echo -e "${YELLOW}📦 Building image: ${FULL_IMAGE}${NC}"
echo "   Docker Hub: ${DOCKER_USER}"
echo "   Tag: ${TAG}"
if [ "$TAG" != "latest" ]; then
    echo -e "   ${GREEN}✓${NC} Using versioned tag (prevents unnecessary image checks)"
fi
echo "   This may take 15-20 minutes..."
echo ""

docker buildx build \
  --platform linux/amd64 \
  --push \
  -t "${FULL_IMAGE}" \
  --build-arg BUILDDATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="${TAG}" \
  --build-arg HUGGINGFACE_USERNAME="${HUGGINGFACE_USERNAME}" \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  -f Dockerfile \
  .

echo ""
echo -e "${GREEN}✓${NC} Image built and pushed successfully!"
echo ""
echo "📦 Image: ${FULL_IMAGE}"
echo "🔗 Docker Hub: https://hub.docker.com/r/${DOCKER_USER}/${IMAGE_NAME}"
echo ""
echo "Next steps:"
echo "1. Update RunPod endpoint to use the new image: ${FULL_IMAGE}"
echo "2. Or create new endpoint with: ${FULL_IMAGE}"
echo "3. Recommended GPU: RTX 4090 (24GB) or A100 (40GB/80GB)"
echo "4. Set disk space: 100GB+"
echo "5. Environment variables: HUGGINGFACE_USERNAME=krapiunitski"
if [ "$TAG" != "latest" ]; then
    echo ""
    echo -e "${GREEN}💡 Tip:${NC} Using versioned tag (${TAG}) prevents Docker from checking"
    echo "   registry for updates, reducing cold start time on RunPod"
fi
echo ""
echo -e "${GREEN}✓${NC} Ready for deployment!"

