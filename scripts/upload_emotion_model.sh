#!/bin/bash
# Script to upload emo_model.pth to HuggingFace repository ${HUGGINGFACE_USERNAME}/moda-emotion-model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EMO_MODEL_PATH="$PROJECT_ROOT/emo_model.pth"
REPO_ID="${HUGGINGFACE_USERNAME:?Set HUGGINGFACE_USERNAME}/moda-emotion-model"

echo "=========================================="
echo "Uploading DICE-Talk Emotion Model to HuggingFace"
echo "=========================================="

# Check if emo_model.pth exists
if [ ! -f "$EMO_MODEL_PATH" ]; then
    echo "[ERROR] emo_model.pth not found at: $EMO_MODEL_PATH"
    echo ""
    echo "Please ensure emo_model.pth is in the MoDA root directory"
    exit 1
fi

# Check file size
FILE_SIZE=$(du -h "$EMO_MODEL_PATH" | cut -f1)
echo "[INFO] Found emo_model.pth"
echo "       Location: $EMO_MODEL_PATH"
echo "       Size: $FILE_SIZE"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "[ERROR] huggingface-cli not found"
    echo "        Install with: pip install 'huggingface_hub[cli]'"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN environment variable is not set"
    echo ""
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

echo ""
echo "[INFO] Repository: $REPO_ID"
echo "[INFO] This will create the repository if it doesn't exist"
echo ""
read -p "Continue with upload? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "[INFO] Upload cancelled"
    exit 0
fi

# Upload file to HuggingFace
echo ""
echo "[INFO] Uploading emo_model.pth to HuggingFace..."
echo "       This may take a few minutes (~286 MB)..."

# Use huggingface-cli to upload
# First, create/ensure repository exists
huggingface-cli repo create "$REPO_ID" \
    --type model \
    --private \
    --token "$HF_TOKEN" 2>/dev/null || echo "[INFO] Repository already exists or creation skipped"

# Upload the file
huggingface-cli upload "$REPO_ID" \
    "$EMO_MODEL_PATH" \
    "emo_model.pth" \
    --token "$HF_TOKEN" \
    --commit-message "Add emo_model.pth for MoDA emotion integration"

if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] Successfully uploaded emo_model.pth to $REPO_ID"
    echo ""
    echo "Repository URL: https://huggingface.co/$REPO_ID"
    echo ""
    echo "The model will be automatically downloaded by:"
    echo "  - Dockerfile (during Docker build)"
    echo "  - runpod_server.py (during RunPod initialization)"
    echo "  - download_emotion_model.sh (for local development)"
else
    echo ""
    echo "[ERROR] Failed to upload emo_model.pth"
    echo ""
    echo "Possible issues:"
    echo "  1. Invalid HF_TOKEN"
    echo "  2. Network connection problem"
    echo "  3. Repository permissions"
    exit 1
fi

