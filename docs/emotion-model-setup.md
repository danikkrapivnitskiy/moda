# DICE-Talk Emotion Model Setup Guide

This guide explains how to obtain and configure the `emo_model.pth` file for enhanced emotion support in MoDA.

## Important: Only `emo_model.pth` is Required

**For MoDA integration, you only need `emo_model.pth` from DICE-Talk.**

The `emo_model.pth` checkpoint contains the complete `EmotionModel` with all its internal components:
- `emo_linear` (AudioProjModel) - already included in the checkpoint
- `kv_tokens_linear` (AudioProjModel) - already included in the checkpoint  
- `codebook` (emotion_bank with 64 emotion codes) - already included
- `attention` (QKVAttention) - already included
- `classifier` (optional) - already included

**You do NOT need:**
- `unet.pth` - Used for SVD UNet (MoDA uses DiT instead)
- `pose_guider.pth` - Used for pose guidance (not used in MoDA)
- `audio_linear.pth` - Separate audio component (AudioProjModel is already inside EmotionModel)

**Optional:** `.npy` emotion feature files for more precise emotion generation (not required).

## Quick Start

### Option 0: Automatic Download in Docker (Already Configured)

**Для RunPod/Docker:** Модель уже настроена на автоматическую загрузку при сборке Docker образа!

При сборке Docker образа с правильными build arguments:
```bash
docker build \
  --build-arg HUGGINGFACE_USERNAME=your-huggingface-username \
  --build-arg HF_TOKEN=hf_your_token_here \
  -t moda:latest .
```

Модель `emo_model.pth` автоматически скачается из `your-huggingface-username/dice-talk-checkpoints` и будет включена в образ.

**Путь в контейнере:** `/app/models_cache/checkpoints/DICE-Talk/emo_model.pth`

### Option 1: Automatic Download Script (For Local Development)

```bash
cd MoDA
./scripts/download_emotion_model.sh
```

The script will:
1. Try to download from HuggingFace (EEEELY/DICE-Talk)
2. Try your private repository (if HF_TOKEN is set)
3. Copy from local DICE-Talk project if available

### Option 2: Manual Download from HuggingFace

```bash
# Install HuggingFace CLI
pip install "huggingface_hub[cli]"

# Download from original DICE-Talk repository
huggingface-cli download EEEELY/DICE-Talk \
  --include "DICE-Talk/emo_model.pth" \
  --local-dir checkpoints

# Or from your repository (requires token)
export HF_TOKEN="hf_your_token_here"
huggingface-cli download your-huggingface-username/dice-talk-checkpoints \
  --include "DICE-Talk/emo_model.pth" \
  --local-dir checkpoints \
  --token "$HF_TOKEN"
```

### Option 3: Copy from Local DICE-Talk Project

If you already have DICE-Talk set up locally:

```bash
# Create directory
mkdir -p MoDA/checkpoints/DICE-Talk

# Copy model
cp DICE-Talk/checkpoints/DICE-Talk/emo_model.pth MoDA/checkpoints/DICE-Talk/
```

### Option 4: Download from Google Drive

According to DICE-Talk README, models are also available on [Google Drive](https://drive.google.com/drive/folders/1l1Ojt-4yMfYQCCnNs_NgkzQC2-OoAksN?usp=drive_link).

1. Download the archive
2. Extract `emo_model.pth` from `DICE-Talk/` folder
3. Place it in `MoDA/checkpoints/DICE-Talk/emo_model.pth`

## File Structure

After downloading, your directory structure should look like:

```
MoDA/
├── checkpoints/
│   └── DICE-Talk/
│       └── emo_model.pth  # ~4.2 MB
├── examples/
│   └── emo/
│       ├── happy.npy
│       ├── sad.npy
│       └── ... (other emotion .npy files)
```

## Configuration

### For Docker/RunPod (Automatic)

**Путь НЕ нужен!** Модель автоматически:
1. Скачивается при сборке Docker образа из `your-huggingface-username/dice-talk-checkpoints`
2. Ищется в стандартных местах при запуске
3. Скачивается из HuggingFace если не найдена (через `ensure_models_downloaded()`)

В RunPod просто включите в `runpod_config.yaml`:

```yaml
emotion:
  use_enhanced_emotion: true
  # emotion_model_path: null  # Не нужно указывать - автоопределение!
  emotion_examples_dir: "examples/emo"
```

Или через environment variables:
```
MODA_EMOTION_USE_ENHANCED_EMOTION=true
# MODA_EMOTION_EMOTION_MODEL_PATH не нужен - автоопределение!
```

**Автоматический поиск в:**
- `/workspace/checkpoints/DICE-Talk/emo_model.pth` (persistent storage)
- `/app/models_cache/checkpoints/DICE-Talk/emo_model.pth` (pre-built cache)
- `checkpoints/DICE-Talk/emo_model.pth` (local development)

### For Local Development

**Путь НЕ обязателен!** Модель автоматически ищется в стандартных местах.

Просто включите в `configs/audio2motion/model/config.yaml`:

```yaml
motion_generator:
  params:
    use_enhanced_emotion: true
    # emotion_model_path: null  # Опционально - автоопределение!
    emotion_examples_dir: "examples/emo"
```

Если хотите указать путь явно (опционально):
```yaml
    emotion_model_path: "checkpoints/DICE-Talk/emo_model.pth"
```

## Verification

Check if the model is loaded correctly:

```python
import torch
model_path = "checkpoints/DICE-Talk/emo_model.pth"
state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
print(f"Model loaded: {len(state_dict)} parameters")
print(f"Keys: {list(state_dict.keys())[:5]}...")  # Show first 5 keys
```

## Emotion .npy Files

The emotion .npy files are optional but recommended for best results. You can:

1. **Copy from DICE-Talk project:**
   ```bash
   cp -r DICE-Talk/examples/emo/* MoDA/examples/emo/
   ```

2. **Download from HuggingFace:**
   ```bash
   huggingface-cli download EEEELY/DICE-Talk \
     --include "examples/emo/*.npy" \
     --local-dir .
   ```

## Troubleshooting

### Model not found error

If you see `[WARN] Emotion checkpoint not found`, check:
- File exists at the configured path
- Path is relative to project root or absolute
- File permissions are correct

### Enhanced emotions disabled

If enhanced emotions are disabled:
- Check `use_enhanced_emotion: true` in config
- Verify model file exists and is readable
- Check logs for initialization errors

### Fallback to simple emotions

The system will automatically fall back to simple emotion codes (0-8) if:
- Model file is not found
- Model fails to load
- `use_enhanced_emotion: false` in config

This is normal and expected - the system will still work with basic emotions.

## File Size

- `emo_model.pth`: ~4.2 MB
- Total emotion module: ~50 MB GPU memory when loaded

## References

- Original DICE-Talk repository: https://huggingface.co/EEEELY/DICE-Talk
- DICE-Talk paper: https://arxiv.org/abs/2504.18087
- Google Drive backup: https://drive.google.com/drive/folders/1l1Ojt-4yMfYQCCnNs_NgkzQC2-OoAksN

