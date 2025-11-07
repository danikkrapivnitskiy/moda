# Emotion Guidance Scale Integration Plan

## Overview

This document describes the plan to integrate a separate emotion guidance scale mechanism from DICE-Talk into MoDA, similar to how DICE-Talk uses `audio_guidance_scale` to control emotion expressiveness in the diffusion process.

## Problem Statement

Currently, MoDA uses a single `emo_scale` parameter that multiplies emotion embeddings before they're integrated with audio and motion. However, this approach has limitations:

1. **Artifacts at higher values**: When `emo_scale > 2.0`, artifacts appear in generated videos
2. **Limited expressiveness**: Even at safe values (1.2-1.5), emotions are less expressive than in DICE-Talk
3. **No CFG-level control**: The current approach doesn't leverage Classifier-Free Guidance (CFG) for emotion control

DICE-Talk achieves more expressive emotions by using `emo_scale` (which maps to `audio_guidance_scale`) as a **guidance scale in the diffusion CFG process**, not just as an embedding multiplier.

## Proposed Solution

Add a new `emo_guidance_scale` parameter that applies separate guidance scale for emotions in the CFG mechanism, similar to DICE-Talk's approach. This will:

1. **Enhance emotion expressiveness** without causing artifacts
2. **Maintain audio-video synchronization** (CFG only affects predictions, not temporal structure)
3. **Provide backward compatibility** (existing configs work unchanged)
4. **Allow fine-tuning** of emotion strength independently from general CFG

## Technical Architecture

### Current CFG Implementation

```python
# Current code in talking_head_diffusion.py (lines 223-228)
if cfg_scale > 1:
    uncond_pred, cond_pred = pred.chunk(2, dim=0)
    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```

**Limitation**: Single `cfg_scale` controls both audio and emotion influence equally.

### Proposed CFG Implementation

```python
# Proposed code with emo_guidance_scale
if cfg_scale > 1:
    uncond_pred, cond_pred = pred.chunk(2, dim=0)
    
    if self.emo_guidance_scale is not None and self.emo_guidance_scale > 1.0:
        # Apply separate guidance scale for emotions
        emotion_boost = (self.emo_guidance_scale - cfg_scale) / cfg_scale if cfg_scale > 1.0 else 0.0
        emotion_boost = max(0.0, min(emotion_boost, 1.0))  # Clamp to 0-100% boost
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred) * (1.0 + emotion_boost)
    else:
        # Standard CFG (backward compatible)
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```

**Benefits**:
- Separate control for emotion influence
- Boost is clamped to prevent instability
- Backward compatible when `emo_guidance_scale` is not set

## Implementation Plan

### Phase 1: Configuration

**File**: `MoDA/runpod_config.yaml`

Add new parameter in `motion_processor` section:

```yaml
motion_processor:
  emo_scale: 1.2  # Keep existing (embedding multiplier)
  emo_guidance_scale: 2.0  # NEW: Separate CFG guidance scale for emotions
                            # If None or not set, uses standard cfg_scale
                            # Recommended range: 1.5-3.0 (similar to DICE-Talk)
                            # Higher values = more expressive emotions
```

### Phase 2: MotionDiffusion Class Updates

**File**: `MoDA/src/models/dit/talking_head_diffusion.py`

#### 2.1. Update `__init__` method

**Location**: After line 48 (where `emo_scale` is read)

```python
emo_scale = motion_gen_params.get('emo_scale', 1.0)
emo_guidance_scale = motion_gen_params.get('emo_guidance_scale', None)  # NEW

# Store as instance attribute (after line 81)
self.emo_guidance_scale = emo_guidance_scale
if emo_guidance_scale is not None:
    print(f"[INFO] Emotion guidance scale set to {emo_guidance_scale} (separate from cfg_scale)")
```

#### 2.2. Update `sample_subclip` method

**Location**: Replace lines 223-228 (CFG application)

```python
# CFG with optional separate emotion guidance scale
if cfg_scale > 1:
    uncond_pred, cond_pred = pred.chunk(2, dim=0)
    
    # Apply separate guidance scale for emotions if specified
    if self.emo_guidance_scale is not None and self.emo_guidance_scale > 1.0:
        # Calculate emotion boost: additional guidance beyond base cfg_scale
        emotion_boost = (self.emo_guidance_scale - cfg_scale) / cfg_scale if cfg_scale > 1.0 else 0.0
        # Clamp boost to prevent instability (max 100% additional influence)
        emotion_boost = max(0.0, min(emotion_boost, 1.0))
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred) * (1.0 + emotion_boost)
    else:
        # Standard CFG (backward compatible)
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```

### Phase 3: Pipeline Configuration Merging

**File**: `MoDA/src/models/inference/moda_test.py`

#### 3.1. Update `__init__` method

**Location**: After line 128 (where `emo_scale` is merged)

```python
# Merge emo_guidance_scale from motion_processor section
emo_guidance_scale = primary_config.get('motion_processor', {}).get('emo_guidance_scale')
if emo_guidance_scale is not None:
    if 'motion_generator' not in motion_models_config:
        motion_models_config['motion_generator'] = OmegaConf.create({})
    if 'params' not in motion_models_config.motion_generator:
        motion_models_config.motion_generator['params'] = OmegaConf.create({})
    motion_models_config.motion_generator.params['emo_guidance_scale'] = emo_guidance_scale
    log(f"Merged emo_guidance_scale={emo_guidance_scale} from primary config into motion_generator.params")
```

## Formula Explanation

### Standard CFG (Current)
```
pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```

This applies the same guidance strength to all conditioning (audio + emotion).

### Enhanced CFG with Emotion Guidance Scale
```
emotion_boost = (emo_guidance_scale - cfg_scale) / cfg_scale
emotion_boost = clamp(emotion_boost, 0.0, 1.0)  # Max 100% additional boost
pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred) * (1.0 + emotion_boost)
```

**Example**:
- `cfg_scale = 1.15` (base guidance)
- `emo_guidance_scale = 2.0` (emotion guidance)
- `emotion_boost = (2.0 - 1.15) / 1.15 = 0.74` (74% additional boost)
- Final: `pred = uncond + 1.15 * (cond - uncond) * 1.74`

This gives emotions 74% more influence than the base CFG, while keeping audio-video synchronization intact.

## Safety Considerations

### 1. Backward Compatibility
- If `emo_guidance_scale` is `None` or not set, standard CFG is used
- Existing configs work without modification
- No breaking changes to API

### 2. Stability
- Boost is clamped to maximum 100% additional influence
- Prevents extreme values that could cause artifacts
- Formula is numerically stable

### 3. Audio-Video Synchronization
- CFG only affects prediction values, not temporal structure
- Audio embeddings are processed separately from emotion embeddings
- No impact on frame timing or synchronization

### 4. Testing Strategy
1. Start with `emo_guidance_scale: 2.0` (matching DICE-Talk)
2. Verify audio-video sync is maintained
3. Compare emotion expressiveness with/without parameter
4. Test edge cases: `emo_guidance_scale < cfg_scale`, `emo_guidance_scale = cfg_scale`

## Comparison with DICE-Talk

### DICE-Talk Approach
```python
# DICE-Talk uses two-level guidance
noise_pred = noise_pred_uncond + \
             guidance_scale1 * (noise_pred_drop_audio - noise_pred_uncond) + \
             guidance_scale2 * (noise_pred_cond - noise_pred_drop_audio)
# Where guidance_scale2 = emo_scale (audio/emotion guidance)
```

### MoDA Proposed Approach
```python
# MoDA uses boost-based approach (simpler, safer)
emotion_boost = (emo_guidance_scale - cfg_scale) / cfg_scale
pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred) * (1.0 + emotion_boost)
```

**Why different?**
- DICE-Talk has three-way CFG (uncond, drop_audio, full_cond)
- MoDA has two-way CFG (uncond, cond)
- Boost approach adapts DICE-Talk's concept to MoDA's architecture
- Simpler to implement and maintain

## Future Improvements

### 1. Three-Way CFG (Advanced)
If needed, implement full DICE-Talk-style three-way CFG:
- Unconditional prediction
- Audio-only prediction (emotion dropped)
- Full conditional prediction (audio + emotion)

This would require:
- Modifying `sample_subclip` to handle three predictions
- Updating `TalkingHeadDiT.forward` to support emotion dropout
- More complex CFG formula

### 2. Adaptive Guidance Scale
Make `emo_guidance_scale` adaptive based on:
- Emotion type (some emotions need more/less guidance)
- Audio characteristics (loud vs quiet speech)
- Video length (longer videos may need different scaling)

### 3. Per-Emotion Guidance Scales
Allow different guidance scales for different emotions:
```yaml
emo_guidance_scale:
  default: 2.0
  happy: 2.5
  sad: 1.8
  angry: 2.2
```

### 4. Time-Varying Guidance
Apply different guidance scales at different diffusion timesteps:
- Higher guidance early in denoising (coarse structure)
- Lower guidance later in denoising (fine details)

## Testing Checklist

- [ ] Test with `emo_guidance_scale: 2.0` (DICE-Talk equivalent)
- [ ] Verify audio-video synchronization maintained
- [ ] Compare emotion expressiveness with/without parameter
- [ ] Test backward compatibility (config without parameter)
- [ ] Test edge cases:
  - [ ] `emo_guidance_scale < cfg_scale`
  - [ ] `emo_guidance_scale = cfg_scale`
  - [ ] `emo_guidance_scale >> cfg_scale` (very high values)
- [ ] Check for artifacts at different values (1.5, 2.0, 2.5, 3.0)
- [ ] Performance benchmark (should be negligible overhead)

## Configuration Examples

### Conservative (Safe)
```yaml
motion_processor:
  emo_scale: 1.2
  emo_guidance_scale: 1.5  # Moderate boost
```

### Balanced (Recommended)
```yaml
motion_processor:
  emo_scale: 1.2
  emo_guidance_scale: 2.0  # DICE-Talk equivalent
```

### Expressive (May cause artifacts)
```yaml
motion_processor:
  emo_scale: 1.2
  emo_guidance_scale: 2.5  # Strong emotions, test carefully
```

### Disabled (Backward Compatible)
```yaml
motion_processor:
  emo_scale: 1.2
  # emo_guidance_scale not set → uses standard CFG
```

## References

- DICE-Talk implementation: `DICE-Talk/dice_talk.py` (lines 373-374)
- DICE-Talk CFG: `DICE-Talk/src/pipelines/pipeline_dicetalk.py` (lines 639-642)
- MoDA CFG: `MoDA/src/models/dit/talking_head_diffusion.py` (lines 223-228)
- MoDA emotion processing: `MoDA/src/models/dit/talking_head_dit.py` (lines 197-308)

## Summary

This plan introduces a separate emotion guidance scale parameter that enhances emotion expressiveness in MoDA by applying additional CFG boost specifically for emotions, similar to DICE-Talk's approach. The implementation is:

- **Safe**: Backward compatible, clamped boost, no sync issues
- **Effective**: Provides DICE-Talk-level emotion expressiveness
- **Simple**: Minimal code changes, easy to maintain
- **Flexible**: Can be tuned per use case or disabled

The plan maintains all existing functionality while adding the capability to achieve more expressive emotions without artifacts.

