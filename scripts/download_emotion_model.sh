#!/bin/bash
# Script to download DICE-Talk emotion model for MoDA integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints/DICE-Talk"

echo "=========================================="
echo "Downloading DICE-Talk Emotion Model"
echo "=========================================="

# Create directory if it doesn't exist
mkdir -p "$CHECKPOINTS_DIR"

# Check if model already exists
if [ -f "$CHECKPOINTS_DIR/emo_model.pth" ]; then
    echo "[INFO] emo_model.pth already exists at: $CHECKPOINTS_DIR/emo_model.pth"
    echo "       File size: $(du -h "$CHECKPOINTS_DIR/emo_model.pth" | cut -f1)"
    read -p "       Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[OK] Using existing model"
        exit 0
    fi
    rm -f "$CHECKPOINTS_DIR/emo_model.pth"
fi

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "[ERROR] huggingface-cli not found"
    echo "        Install with: pip install 'huggingface_hub[cli]'"
    exit 1
fi

# Try to download from HuggingFace
echo "[INFO] Attempting to download from HuggingFace..."

HF_REPO="${HUGGINGFACE_USERNAME:+${HUGGINGFACE_USERNAME}/moda-emotion-model}"

# Download from lightweight repository (only emo_model.pth, ~286 MB)
if [ -n "$HF_REPO" ]; then
  echo "[1/2] Downloading from ${HF_REPO} (lightweight, ~286 MB)..."
  if [ -n "$HF_TOKEN" ]; then
    if huggingface-cli download "$HF_REPO" \
        --include "emo_model.pth" \
        --local-dir "$CHECKPOINTS_DIR" \
        --token "$HF_TOKEN" 2>/dev/null; then
        echo "[OK] Downloaded from lightweight repository (~286 MB)"
        exit 0
    fi
  else
    echo "[INFO] HF_TOKEN not set, trying without token..."
    if huggingface-cli download "$HF_REPO" \
        --include "emo_model.pth" \
        --local-dir "$CHECKPOINTS_DIR" 2>/dev/null; then
        echo "[OK] Downloaded from lightweight repository (public, ~286 MB)"
        exit 0
    fi
  fi
else
  echo "[INFO] HUGGINGFACE_USERNAME not set, skipping private mirror download"
fi

# Option 2: Check if local DICE-Talk project exists
DICE_TALK_DIR="$PROJECT_ROOT/../DICE-Talk"
if [ -f "$DICE_TALK_DIR/checkpoints/DICE-Talk/emo_model.pth" ]; then
    echo "[INFO] Found model in local DICE-Talk project"
    cp "$DICE_TALK_DIR/checkpoints/DICE-Talk/emo_model.pth" "$CHECKPOINTS_DIR/"
    echo "[OK] Copied from local DICE-Talk project"
    exit 0
fi

echo "[ERROR] Failed to download emo_model.pth"
echo ""
echo "Options:"
echo "1. Download manually from HuggingFace:"
echo "   huggingface-cli download EEEELY/DICE-Talk --include 'DICE-Talk/emo_model.pth' --local-dir checkpoints"
echo ""
echo "2. Download from Google Drive (see DICE-Talk README):"
echo "   https://drive.google.com/drive/folders/1l1Ojt-4yMfYQCCnNs_NgkzQC2-OoAksN"
echo ""
echo "3. Copy from local DICE-Talk project:"
echo "   cp DICE-Talk/checkpoints/DICE-Talk/emo_model.pth MoDA/checkpoints/DICE-Talk/"
echo ""
exit 1

