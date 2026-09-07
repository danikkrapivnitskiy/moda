# DICE-Talk Emotion Integration Flow

## Overview

This document describes the complete flow of how the enhanced emotion system works in MoDA, from input to output.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RunPod Server Handler                         │
│  - Receives request with image, audio, emotion                   │
│  - Processes emotion input (int code, .npy file, base64)        │
│  - Calls LiveVASAPipeline                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LiveVASAPipeline                                 │
│  - AudioProcessor: extracts audio features                        │
│  - MotionGenerator: generates motion parameters                   │
│  - MotionProcesser: renders video with LivePortrait             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MotionDiffusion (DiT)                                │
│  - TalkingHeadDiT: diffusion transformer                         │
│  - EmotionAdapter: enhanced emotion processing                   │
│  - Generates motion parameters (facial expressions, head pose)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LivePortrait Renderer                                │
│  - Appearance extractor                                          │
│  - Warping network                                               │
│  - SPADE generator                                               │
│  - Output: video with enhanced facial expressions               │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Flow

### Phase 1: Initialization (Startup)

#### 1.1 Model Loading

```python
# Location: runpod_server.py -> get_pipe() -> ensure_models_downloaded()

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Check Persistent Storage                            │
│   /workspace/checkpoints/DICE-Talk/emo_model.pth           │
│   ✅ Found → Use it                                         │
│   ❌ Not found → Continue to Step 2                        │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Check Pre-built Docker Cache                        │
│   /app/models_cache/checkpoints/DICE-Talk/emo_model.pth     │
│   ✅ Found → Copy to persistent storage                     │
│   ❌ Not found → Continue to Step 3                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Download from HuggingFace (Optional)                │
│   your-huggingface-username/dice-talk-checkpoints                         │
│   ✅ Success → Save to persistent storage                    │
│   ❌ Failed → Continue without enhanced emotions            │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2 Pipeline Initialization

```python
# Location: moda_test.py -> LiveVASAPipeline.__init__()

1. Load AudioProcessor
   └─> Extracts audio features from WAV files

2. Load MotionDiffusion
   └─> Loads DiT model with emotion support
       ├─> TalkingHeadDiT.__init__()
       │   ├─> Check use_enhanced_emotion flag
       │   ├─> Initialize EmotionAdapter (if enabled)
       │   │   ├─> Load emo_model.pth checkpoint
       │   │   ├─> Initialize EmotionModel
       │   │   ├─> Create attention pooling layer
       │   │   └─> Create projection layer (1024 → hidden_size)
       │   └─> Initialize LabelEmbedder (fallback)
       └─> Load DiT weights

3. Load MotionProcesser
   └─> LivePortrait renderer for video generation
```

### Phase 2: Request Processing (RunPod Handler)

#### 2.1 Input Processing

```python
# Location: runpod_server.py -> handler()

Request JSON:
{
    "image": "base64_encoded_image",
    "audio": "base64_encoded_audio",
    "emotion": 4  # or "path/to/emotion.npy" or base64 .npy
}

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Decode Inputs                                        │
│   - Decode base64 image → temp file                          │
│   - Decode base64 audio → temp file                          │
│   - Process emotion input                                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Process Emotion Input                               │
│   process_emotion_input(emotion_input, temp_dir)            │
│                                                              │
│   Case 1: Integer (0-8)                                     │
│   └─> Return as-is, will be processed by EmotionAdapter      │
│                                                              │
│   Case 2: String path to .npy file                         │
│   └─> Validate path exists, return path                     │
│                                                              │
│   Case 3: Base64 encoded .npy file                          │
│   └─> Decode → Save to temp file → Return path              │
│                                                              │
│   Case 4: Emotion name string ("happy", "sad", etc.)        │
│   └─> Map to integer code (0-8)                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Call Pipeline                                        │
│   pipeline.driven_sample(                                    │
│       image_path, audio_path,                                │
│       emo=processed_emotion                                  │
│   )                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Audio Processing

```python
# Location: moda_test.py -> process_audio()

┌─────────────────────────────────────────────────────────────┐
│ AudioProcessor.extract_features()                            │
│   - Load WAV file                                            │
│   - Extract audio embeddings using Wav2Vec2/HuBERT          │
│   - Output: (B, T, D) audio feature tensor                  │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Emotion Processing (Enhanced Path)

#### 4.1 EmotionAdapter Forward Pass

```python
# Location: emotion_adapter.py -> forward()

Input: emotion_input (int, str path, or torch.Tensor)

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Input Type Detection                                 │
│                                                              │
│   Type: int (0-8)                                           │
│   └─> Try to find .npy file in emotion_examples_dir         │
│       ├─> Found → Load .npy → Process with EmotionModel     │
│       └─> Not found → Return None (fallback to LabelEmbedder)│
│                                                              │
│   Type: str (path to .npy)                                  │
│   └─> Load .npy file → Process with EmotionModel            │
│                                                              │
│   Type: torch.Tensor                                        │
│   └─> Ensure shape (1, 1, 1, 1, 256) → Process              │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Load Emotion Features                                │
│   load_emotion_from_npy(npy_path)                            │
│                                                              │
│   - np.load(npy_path, allow_pickle=True)                    │
│   - Extract data.item()['mu'] (DICE-Talk format)           │
│   - Convert to torch.Tensor                                 │
│   - Ensure shape: (1, 1, 1, 1, 256)                         │
│   - Output: emo_features (B, L, D1, D2, 256)                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: EmotionModel Processing                              │
│   _process_with_emotion_model(emo_features)                  │
│                                                              │
│   ┌──────────────────────────────────────────────┐         │
│   │ EmotionModel.forward(emo_features)            │         │
│   │                                                │         │
│   │ 1. VQ-VAE Codebook                            │         │
│   │    emo_retrieval, vq_loss, indices =          │         │
│   │        codebook(emo_features)                 │         │
│   │    → Retrieves 64 emotion codes               │         │
│   │                                                │         │
│   │ 2. AudioProjModel (emo_linear)                │         │
│   │    emo_prompts_q = emo_linear(emo_features)    │         │
│   │    → (B, L, 32, 1024)                        │         │
│   │                                                │         │
│   │ 3. AudioProjModel (kv_tokens_linear)          │         │
│   │    kv_tokens = kv_tokens_linear(emo_retrieval) │         │
│   │    → (B, L, 32, 1024)                        │         │
│   │                                                │         │
│   │ 4. QKVAttention                               │         │
│   │    final_emo_prompts, attn_weights =         │         │
│   │        attention(emo_prompts_q, kv_tokens)    │         │
│   │    → (B, L, 32, 1024)                        │         │
│   └──────────────────────────────────────────────┘         │
│                                                              │
│   Output: emo_tokens (B, L, 32, 1024)                      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Attention Pooling                                    │
│                                                              │
│   - Reshape: (B, L, 32, 1024) → (B*L, 32, 1024)            │
│   - Compute attention weights:                               │
│     attn_weights = attention_pool(emo_tokens_flat)            │
│     → (B*L, 32, 1)                                          │
│   - Softmax normalization                                    │
│   - Weighted sum: (B*L, 32, 1024) → (B*L, 1024)            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Projection to hidden_size                           │
│                                                              │
│   - Linear projection: (B*L, 1024) → (B*L, hidden_size)    │
│   - Reshape: (B*L, hidden_size) → (B, L, hidden_size)       │
│   - Squeeze if L=1: (B, L, hidden_size) → (B, hidden_size) │
│                                                              │
│   Output: emo_embeds (B, hidden_size)                       │
│   Example: (1, 1152) for hidden_size=1152                   │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2 Fallback Path (Simple Emotion)

```python
# Location: talking_head_dit.py -> forward()

If EmotionAdapter returns None or use_enhanced_emotion=False:

┌─────────────────────────────────────────────────────────────┐
│ LabelEmbedder (Simple Emotion Embedding)                    │
│                                                              │
│   - Input: emo_tensor (int 0-8)                             │
│   - Embedding lookup: embedding_table[emo_tensor]           │
│   - Output: emo_embeds (B, hidden_size)                     │
│                                                              │
│   Note: Less expressive than EmotionModel, but faster       │
└─────────────────────────────────────────────────────────────┘
```

### Phase 5: Motion Generation (DiT Forward Pass)

```python
# Location: talking_head_dit.py -> forward()

Inputs:
  - motion: (B, N, motion_dim) - previous motion parameters
  - audio: (B, N, M, audio_dim) - audio embeddings
  - emo: int, str, or tensor - emotion input
  - times: (B,) - diffusion timesteps

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Embedding Extraction                                 │
│                                                              │
│   motion_embeds = motion_embedder(motion)                   │
│     → (B, N, hidden_size)                                   │
│                                                              │
│   audio_embeds = audio_embedder(audio)                       │
│     → (B, N, M, hidden_size)                                │
│     → Reshape: (B, N*M, hidden_size)                        │
│                                                              │
│   time_embeds = time_embedder(times)                         │
│     → (B, hidden_size)                                      │
│                                                              │
│   emo_embeds = emotion_adapter(emo) or emo_embedder(emo)    │
│     → (B, hidden_size)                                      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Condition Combination                               │
│                                                              │
│   c = time_embeds + emo_embeds                               │
│     → (B, hidden_size)                                      │
│                                                              │
│   emo_embeds = emo_embeds.unsqueeze(1).repeat(1, N, 1)      │
│     → (B, N, hidden_size)                                   │
│                                                              │
│   audio_cond_embeds = identity_embedder(audio_cond)          │
│     → (B, N, hidden_size)                                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: DiT Blocks Processing                               │
│                                                              │
│   blocks4 (3 blocks):                                       │
│     motion_embeds, audio_embeds, emo_embeds,                │
│     audio_cond_embeds = block(...)                          │
│                                                              │
│   Concatenate:                                               │
│     audio_embeds = cat([audio_embeds, emo_embeds,          │
│                         audio_cond_embeds], dim=1)          │
│                                                              │
│   blocks2 (6 blocks):                                       │
│     motion_embeds, audio_embeds = block(...)                 │
│                                                              │
│   Concatenate:                                               │
│     motion_embeds = cat([motion_embeds, audio_embeds], dim=1)│
│                                                              │
│   blocks (12 blocks):                                       │
│     motion_embeds = block(...)                               │
│                                                              │
│   Extract sequence:                                         │
│     motion_embeds = motion_embeds[:, :N, :]                 │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Final Layer                                          │
│                                                              │
│   out = final_layer(motion_embeds, c)                        │
│     → (B, N, out_channels)                                  │
│                                                              │
│   Output: Motion parameters (facial expressions, head pose) │
└─────────────────────────────────────────────────────────────┘
```

### Phase 6: Video Rendering (LivePortrait)

```python
# Location: motion_processer.py -> driven_by_audio()

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Motion to Keypoints                                  │
│                                                              │
│   - Convert motion parameters to keypoint sequences          │
│   - Apply rescaling and cropping info                       │
│   - Output: kp_infos (list of keypoint frames)              │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: LivePortrait Rendering                               │
│                                                              │
│   For each frame:                                            │
│     1. Appearance Feature Extractor                          │
│        - Extract appearance features from reference image    │
│                                                              │
│     2. Warping Network                                       │
│        - Generate dense motion field from keypoints          │
│                                                              │
│     3. SPADE Generator                                       │
│        - Generate frame with appearance + motion             │
│        - Apply lip modulation (MoDA's strength)             │
│                                                              │
│   Output: Video frames with enhanced facial expressions      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Video Assembly                                       │
│                                                              │
│   - Combine frames into video                               │
│   - Add audio track                                          │
│   - Apply smoothing (optional)                               │
│   - Output: Final video file (.mp4)                         │
└─────────────────────────────────────────────────────────────┘
```

## Complete Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    RunPod Request                            │
│  {image, audio, emotion}                                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              process_emotion_input()                         │
│  - Decode base64 .npy (if needed)                            │
│  - Validate integer codes                                     │
│  - Map emotion names to codes                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│            LiveVASAPipeline.driven_sample()                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. process_audio()                                      │ │
│  │    AudioProcessor → audio embeddings                     │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 2. load_emotion_features()                              │ │
│  │    - Process emotion input                              │ │
│  │    - Return emotion code or .npy path                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 3. motion_generator.sample()                            │ │
│  │    ┌──────────────────────────────────────────────────┐│ │
│  │    │ MotionDiffusion.sample()                         ││ │
│  │    │  ┌────────────────────────────────────────────┐  ││ │
│  │    │  │ TalkingHeadDiT.forward()                   │  ││ │
│  │    │  │  ┌──────────────────────────────────────┐ │  ││ │
│  │    │  │  │ EmotionAdapter.forward()             │ │  ││ │
│  │    │  │  │  - Load .npy → (1,1,1,1,256)         │ │  ││ │
│  │    │  │  │  - EmotionModel → (B,L,32,1024)        │ │  ││ │
│  │    │  │  │  - Pooling → (B,1024)                 │ │  ││ │
│  │    │  │  │  - Project → (B,hidden_size)           │ │  ││ │
│  │    │  │  └──────────────────────────────────────┘ │  ││ │
│  │    │  │  - Combine with time_embeds                │  ││ │
│  │    │  │  - DiT blocks processing                   │  ││ │
│  │    │  │  - Generate motion parameters              │  ││ │
│  │    │  └──────────────────────────────────────────┘ │  ││ │
│  │    └──────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 4. motion_processer.driven_by_audio()                  │ │
│  │    - Convert motion to keypoints                        │ │
│  │    - LivePortrait rendering                             │ │
│  │    - Video assembly                                     │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Final Video Output                         │
│  - Enhanced facial expressions (DICE-Talk)                   │
│  - Superior lip sync (MoDA)                                   │
└──────────────────────────────────────────────────────────────┘
```

## Fallback Mechanisms

### 1. EmotionModel Unavailable

```
┌──────────────────────────────────────────────────────────────┐
│ Check: emo_model.pth exists?                                 │
│  ❌ Not found                                                 │
│  └─> use_enhanced_emotion = False                            │
│      └─> Use LabelEmbedder (simple emotion codes)           │
└──────────────────────────────────────────────────────────────┘
```

### 2. EmotionAdapter Fails

```
┌──────────────────────────────────────────────────────────────┐
│ Try: EmotionAdapter.forward(emo)                             │
│  ❌ Exception raised                                           │
│  └─> Catch exception                                          │
│      └─> Fallback to LabelEmbedder                            │
└──────────────────────────────────────────────────────────────┘
```

### 3. .npy File Not Found

```
┌──────────────────────────────────────────────────────────────┐
│ Input: int emotion code (e.g., 4 = happy)                    │
│  └─> Try: Find examples/emo/happy.npy                        │
│      ❌ Not found                                              │
│      └─> Return None from EmotionAdapter                     │
│          └─> Fallback to LabelEmbedder                       │
└──────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

### Enhanced Emotion Path (DICE-Talk)
- **EmotionModel forward**: ~5-10ms
- **Attention pooling**: ~1-2ms
- **Projection**: <1ms
- **Total overhead**: ~6-13ms per inference step
- **Memory**: +4.2 MB (emo_model.pth) + ~50 MB (runtime)

### Simple Emotion Path (Fallback)
- **LabelEmbedder lookup**: <1ms
- **Total overhead**: <1ms per inference step
- **Memory**: Minimal

## Configuration Flags

```yaml
# configs/audio2motion/model/config.yaml
use_enhanced_emotion: true  # Enable DICE-Talk emotions
emotion_model_path: null    # Auto-detect (recommended)
emotion_examples_dir: null  # Path to .npy files (optional)
```

## Summary

The integration flow ensures:
1. ✅ **Robustness**: Multiple fallback mechanisms
2. ✅ **Flexibility**: Supports int codes, .npy files, base64
3. ✅ **Performance**: Minimal overhead when enhanced emotions enabled
4. ✅ **Compatibility**: Works with or without emo_model.pth
5. ✅ **Quality**: Combines DICE-Talk expressions with MoDA lip sync

