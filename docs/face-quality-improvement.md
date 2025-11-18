# Face Quality Improvement Guide

## Problem Analysis

### Why Face Quality Suffers More Than Other Body Parts

The MoDA pipeline uses a **crop-and-paste-back** approach:

1. **Face is cropped** from original image (e.g., 1024x1024) → **256x256** for processing
2. **Face is generated** at 256x256 resolution (model's training resolution)
3. **Face is upscaled** to output resolution (512x512)
4. **Face is pasted back** into original image using blending
5. **Shoulders and body** remain untouched (original quality)

**Result:** Face quality is degraded due to:
- Low processing resolution (256x256)
- Upscaling artifacts (256 → 512)
- Blending artifacts at face boundaries
- Model limitations at higher resolutions

## Current Configuration

### Processing Pipeline

```yaml
# Current settings in runpod_config.yaml
motion_processor:
  source_max_dim: 512        # Source image max dimension
  input_height: 256          # Internal processing (model trained on this)
  input_width: 256            # Internal processing (model trained on this)
  output_height: 512          # Output video resolution
  output_width: 512           # Output video resolution
  flag_stitching: True        # Enable stitching for better blending
  flag_pasteback: True        # Paste face back to original image
```

### Paste-Back Process

The face is pasted back using one of two methods:

1. **Simple blending** (default):
   ```python
   result = mask * generated_face + (1 - mask) * original_image
   ```
   - Fast but can show seams
   - Visible boundaries between face and body

2. **Laplacian pyramid blending** (optional):
   ```python
   result = laplacian_pyramid_blending(generated_face, original_image, mask)
   ```
   - Better quality, smoother transitions
   - More computationally expensive
   - Currently **NOT enabled by default**

## Improvement Strategies

### Strategy 1: Enable Laplacian Blending (Recommended)

**What it does:**
- Uses multi-scale blending (Laplacian pyramid)
- Creates smoother transitions at face boundaries
- Reduces visible seams between face and body

**Implementation:**

Find where `paste_back` is called and enable `use_laplacian=True`:

```python
# In motion_processer.py or live_portrait_pipeline.py
# Current:
result = paste_back(generated_face, crop_info, original_image, mask)

# Improved:
result = paste_back(generated_face, crop_info, original_image, mask, use_laplacian=True)
```

**Or use face mask-based paste-back:**

```python
# Better quality with face mask detection
result = paste_back_by_face_mask(
    generated_face, 
    crop_info, 
    original_image, 
    crop_src_image,
    use_laplacian=True  # Enable Laplacian blending
)
```

**Performance impact:**
- +10-20% processing time
- Better visual quality
- Smoother face-body transitions

### Strategy 2: Increase Source Image Quality

**Current:**
```yaml
source_max_dim: 512  # Source image is downscaled to max 512px
```

**Improved:**
```yaml
source_max_dim: 1024  # Keep higher quality source image
```

**Benefits:**
- Better quality for non-face regions (shoulders, background)
- More detail preserved before face cropping
- Better paste-back blending (higher resolution mask)

**Trade-offs:**
- Slightly slower processing
- More memory usage
- Face still processed at 256x256 (model limitation)

### Strategy 3: Improve Stitching Quality

**Current:**
```yaml
flag_stitching: True
driving_smooth_observation_variance: 3e-7
```

**Improved:**
```yaml
flag_stitching: True
driving_smooth_observation_variance: 1e-7  # Less smoothing = more detail
```

**Benefits:**
- Preserves more facial details
- Better expression accuracy
- Less blur in generated face

**Trade-offs:**
- Slightly less smooth motion
- May show more artifacts if motion is too sharp

### Strategy 4: Use Better Face Masks

**Current:**
- Uses simple crop mask (`mask_template.png`)
- May not perfectly match face boundaries

**Improved:**
- Use face parser for precise face detection
- Better mask boundaries = better blending

**Implementation:**

The code already has `paste_back_by_face_mask` which uses face parser:

```python
# This is already available but may not be used by default
result = paste_back_by_face_mask(
    result, 
    crop_info, 
    src_img, 
    crop_src_image,
    use_laplacian=True  # Enable for best quality
)
```

### Strategy 5: Increase Output Resolution (Limited)

**Current:**
```yaml
output_height: 512
output_width: 512
```

**Improved (if model supports):**
```yaml
output_height: 768  # Higher output resolution
output_width: 768
```

**⚠️ Warning:**
- Model is trained on 256x256
- Upscaling to 768 may introduce artifacts
- Test carefully before using

**Better approach:**
- Keep output at 512x512
- Use better blending (Laplacian) instead

### Strategy 6: Post-Processing Enhancement

**Option A: Face-Specific Upscaling**

After paste-back, apply face-specific upscaling:

```python
# Use face detection to upscale only face region
face_region = detect_face_region(result)
upscaled_face = upscale_face(face_region, method='ESRGAN')  # or Real-ESRGAN
result = blend_upscaled_face(result, upscaled_face, face_mask)
```

**Option B: Denoising**

Apply denoising to face region only:

```python
face_mask = detect_face_mask(result)
denoised_face = cv2.fastNlMeansDenoisingColored(face_region, None, 10, 10, 7, 21)
result = blend_denoised_face(result, denoised_face, face_mask)
```

## Recommended Configuration

### High Quality (Best Face Quality)

```yaml
motion_processor:
  source_max_dim: 1024        # Higher source quality
  input_height: 256           # Keep model resolution
  input_width: 256            # Keep model resolution
  output_height: 512          # Output resolution
  output_width: 512           # Output resolution
  flag_stitching: True        # Enable stitching
  flag_pasteback: True        # Enable paste-back
  driving_smooth_observation_variance: 1e-7  # Less smoothing = more detail
  # Enable Laplacian blending in code (use_laplacian=True)
```

### Balanced (Good Quality + Performance)

```yaml
motion_processor:
  source_max_dim: 768         # Medium source quality
  input_height: 256           # Keep model resolution
  input_width: 256            # Keep model resolution
  output_height: 512          # Output resolution
  output_width: 512           # Output resolution
  flag_stitching: True        # Enable stitching
  flag_pasteback: True        # Enable paste-back
  driving_smooth_observation_variance: 2e-7  # Balanced smoothing
  # Enable Laplacian blending in code (use_laplacian=True)
```

### Performance (Current + Laplacian)

```yaml
motion_processor:
  source_max_dim: 512         # Current setting
  input_height: 256           # Keep model resolution
  input_width: 256            # Keep model resolution
  output_height: 512          # Output resolution
  output_width: 512           # Output resolution
  flag_stitching: True        # Enable stitching
  flag_pasteback: True        # Enable paste-back
  # Enable Laplacian blending in code (use_laplacian=True)
  # This alone will improve face quality significantly
```

## Code Changes Required

### 1. Enable Laplacian Blending

**File:** `src/models/inference/moda_test.py` or `src/datasets/preprocess/extract_features/motion_processer.py`

**Find:**
```python
results = [paste_back(results[i], crop_info['M_c2o'], src_img, mask_ori_float) for i in range(frames)]
```

**Replace with:**
```python
results = [paste_back(results[i], crop_info['M_c2o'], src_img, mask_ori_float, use_laplacian=True) for i in range(frames)]
```

**Or use face mask version:**
```python
# Better quality with face mask
for i in range(frames):
    result = paste_back_by_face_mask(
        results[i], 
        crop_info, 
        src_img, 
        crop_src_image,
        use_laplacian=True
    )
    results[i] = result
```

### 2. Add Configuration Parameter

**File:** `runpod_config.yaml`

```yaml
motion_processor:
  # ... existing parameters ...
  use_laplacian_blending: True  # Enable Laplacian pyramid blending for better quality
  source_max_dim: 1024          # Increase for better source quality
```

**File:** `src/utils/config_manager.py`

Add parameter merging:
```python
# Merge use_laplacian_blending into motion processor config
if 'use_laplacian_blending' in motion_processor_config:
    use_laplacian = motion_processor_config['use_laplacian_blending']
    # Pass to paste_back function
```

## Performance Impact

### Laplacian Blending

| Metric | Simple Blending | Laplacian Blending |
|--------|-----------------|-------------------|
| Processing Time | Baseline | +10-20% |
| Face Quality | Good | Excellent |
| Boundary Smoothness | Visible seams | Seamless |
| Memory Usage | Baseline | +5-10% |

### Source Max Dimension

| source_max_dim | Processing Time | Face Quality | Body Quality |
|----------------|----------------|--------------|--------------|
| 512 (current) | Baseline | Good | Good |
| 768 | +15% | Good | Better |
| 1024 | +30% | Good | Excellent |

## Testing Recommendations

### Test Sequence

1. **Start with Laplacian blending only:**
   - Enable `use_laplacian=True`
   - Keep other settings unchanged
   - Compare face quality

2. **Then increase source_max_dim:**
   - Try 768 first
   - Then 1024 if needed
   - Monitor performance

3. **Adjust smoothing:**
   - Reduce `driving_smooth_observation_variance` for more detail
   - Test for artifacts

### Quality Metrics

Compare:
- Face sharpness (especially eyes, mouth)
- Face-body boundary smoothness
- Overall video quality
- Processing time

## Why Shoulders Look Better

**Shoulders remain untouched:**
- Original image quality preserved
- No processing artifacts
- No upscaling artifacts
- No blending artifacts

**Face is processed:**
- Cropped to 256x256
- Generated at 256x256
- Upscaled to 512x512
- Blended back into image

**Solution:**
- Better blending (Laplacian) reduces visible difference
- Higher source quality helps preserve details
- Better masks improve boundary quality

## Summary

### Quick Win (Easiest, Biggest Impact)

**Enable Laplacian blending:**
- Change one line of code
- +10-20% processing time
- Significantly better face quality
- Smoother face-body transitions

### Best Quality

1. Enable Laplacian blending
2. Increase `source_max_dim` to 1024
3. Use face mask-based paste-back
4. Reduce smoothing for more detail

### Balanced Approach

1. Enable Laplacian blending
2. Keep `source_max_dim` at 512-768
3. Use face mask-based paste-back

## Quick Implementation Guide

### Step 1: Enable Laplacian Blending (Easiest, Biggest Impact)

**File:** `src/datasets/preprocess/extract_features/motion_processer.py`

**Line 1262:** Change from:
```python
results = [paste_back(results[i], crop_info['M_c2o'], src_img, mask_ori_float) for i in range(frames)]
```

**To:**
```python
results = [paste_back(results[i], crop_info['M_c2o'], src_img, mask_ori_float, use_laplacian=True) for i in range(frames)]
```

**Impact:**
- ✅ Better face quality
- ✅ Smoother face-body transitions
- ✅ Less visible seams
- ⚠️ +10-20% processing time

### Step 2: Use Face Mask-Based Paste-Back (Better Quality)

**File:** `src/datasets/preprocess/extract_features/motion_processer.py`

**Replace line 1262 with:**
```python
# Use face mask for better blending boundaries
crop_src_image = src_img_256x256  # Already available from line 1252
for i in range(frames):
    result = self.paste_back_by_face_mask(
        results[i], 
        crop_info, 
        src_img, 
        crop_src_image,
        use_laplacian=True  # Enable Laplacian blending
    )
    results[i] = result
```

**Impact:**
- ✅ Best face quality
- ✅ Precise face boundaries
- ✅ Better blending
- ⚠️ +15-25% processing time

### Step 3: Increase Source Quality (Optional)

**File:** `runpod_config.yaml`

**Change:**
```yaml
motion_processor:
  source_max_dim: 1024  # Increase from 512
```

**Impact:**
- ✅ Better body/background quality
- ✅ Better paste-back blending
- ⚠️ +15-30% processing time
- ⚠️ More memory usage

## Testing Checklist

After making changes:

- [ ] Test with different face images
- [ ] Compare face quality before/after
- [ ] Check face-body boundary smoothness
- [ ] Monitor processing time
- [ ] Verify no artifacts introduced
- [ ] Test with different emotions/expressions

## Expected Results

### Before (Simple Blending)
- Visible seams at face boundaries
- Face looks slightly "pasted on"
- Quality difference between face and body is noticeable

### After (Laplacian Blending)
- Seamless face-body transitions
- Face blends naturally with body
- Quality difference is minimized
- More professional appearance

## References

- `src/thirdparty/liveportrait/src/utils/crop.py` - Paste-back implementation (lines 540-565)
- `src/datasets/preprocess/extract_features/motion_processer.py` - Motion processing (line 1262)
- `runpod_config.yaml` - Configuration parameters

