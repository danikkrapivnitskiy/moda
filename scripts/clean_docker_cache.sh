#!/bin/bash
set -e

echo "🧹 Cleaning Docker build cache"
echo "=============================="
echo ""

# Configuration
IMAGE_NAME="moda-runpod"
DOCKER_USER="${DOCKER_USER:-krapiunitski12}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "This will remove:"
echo "1. Docker build cache"
echo "2. Dangling images"
echo "3. Stopped containers"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "🧹 Cleaning build cache..."
docker buildx prune -f

echo ""
echo "🧹 Removing dangling images..."
docker image prune -f

echo ""
echo "🧹 Removing stopped containers..."
docker container prune -f

echo ""
echo -e "${GREEN}✅${NC} Docker cache cleaned!"
echo ""

# Show disk space saved
echo "Current Docker disk usage:"
docker system df

