#!/bin/bash
set -e

echo "🐳 MoDA Docker Image Test Script"
echo ""

# Check if we're in the MoDA directory
if [ ! -f "app.py" ]; then
    echo "❌ Not in MoDA directory!"
    echo "   Please run this script from the MoDA directory"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "   Please start Docker Desktop and try again"
    exit 1
fi

# Configuration
IMAGE_NAME="moda-runpod"
DOCKER_USER="${DOCKER_USER:-krapiunitski12}"
TAG="${TAG:-latest}"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

# Menu
echo "Docker Image Test Options:"
echo "1. Build Docker image"
echo "2. Test Docker image locally"
echo "3. Check if image exists"
echo "4. Run container interactively"
echo "5. Exit"
echo ""

read -p "Select option (1-5): " option

case $option in
    1)
        echo "🔨 Building Docker image..."
        ./scripts/build_docker.sh
        ;;
    2)
        echo "🧪 Testing Docker image locally..."
        if ! docker images | grep -q "${DOCKER_USER}/${IMAGE_NAME}.*${TAG}"; then
            echo "❌ Image ${FULL_IMAGE} not found!"
            echo "   Build it first with option 1"
            exit 1
        fi
        
        echo "✓ Image found: ${FULL_IMAGE}"
        echo "Running container test..."
        
        # Run container with test
        docker run --rm --gpus all "${FULL_IMAGE}" python3 -c "
import sys
try:
    sys.path.insert(0, '/app/src')
    from models.inference.moda_test import LiveVASAPipeline
    print('✅ MoDA import successful')
    
    # Test basic initialization (will download models on first run)
    print('Testing model initialization...')
    pipe = LiveVASAPipeline('configs/audio2motion/inference/inference.yaml', load_motion_generator=True)
    print('✅ Model initialization successful')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"
        ;;
    3)
        echo "🔍 Checking if image exists..."
        if docker images | grep -q "${DOCKER_USER}/${IMAGE_NAME}.*${TAG}"; then
            echo "✅ Image found: ${FULL_IMAGE}"
            docker images | grep "${DOCKER_USER}/${IMAGE_NAME}"
        else
            echo "❌ Image not found: ${FULL_IMAGE}"
            echo "   Build it with option 1"
        fi
        ;;
    4)
        echo "🚀 Running container interactively..."
        if ! docker images | grep -q "${DOCKER_USER}/${IMAGE_NAME}.*${TAG}"; then
            echo "❌ Image ${FULL_IMAGE} not found!"
            echo "   Build it first with option 1"
            exit 1
        fi
        
        echo "Starting interactive container..."
        echo "Type 'exit' to quit"
        docker run --rm -it --gpus all "${FULL_IMAGE}" /bin/bash
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✓ Done!"

