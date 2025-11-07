# Emotion Examples Directory

This directory contains emotion feature files (.npy) for enhanced emotion control.

## Usage

Place DICE-Talk emotion .npy files here to enable enhanced emotion processing.

## Expected Files

- `happy.npy` - Happiness emotion features
- `sad.npy` - Sadness emotion features
- `angry.npy` - Anger emotion features
- `surprised.npy` - Surprise emotion features
- `neutral.npy` - Neutral emotion features
- `contempt.npy` - Contempt emotion features
- `disgusted.npy` - Disgust emotion features
- `fear.npy` - Fear emotion features

## Source

These files can be copied from DICE-Talk project:
```
DICE-Talk/examples/emo/*.npy → MoDA/examples/emo/
```

Or downloaded from HuggingFace repository `krapiunitski/dice-talk-checkpoints`.

## Format

Each .npy file contains a dictionary with a 'mu' key:
```python
import numpy as np
data = np.load('happy.npy', allow_pickle=True)
emotion_features = data.item()['mu']  # Shape: (1, 1, 1, 1, 256)
```

## Configuration

Set `emotion_examples_dir: "examples/emo"` in `runpod_config.yaml` or `configs/audio2motion/model/config.yaml` to use these files.

