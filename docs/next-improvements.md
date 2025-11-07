# Next Improvements - MoDA Roadmap

This document outlines planned improvements and enhancements for the MoDA project. Items are organized by priority and category.

## High Priority Improvements

### 1. SPADEDecoder State Dict Compatibility Fix

**Status**: Known Issue  
**Priority**: High  
**Estimated Effort**: 2-3 hours  
**Related**: CHANGELOG.md Known Issues section

**Problem**: Model loading fails with error:
```
Error(s) in loading state_dict for SPADEDecoder:
  Missing key(s) in state_dict: "conv_img.weight", "conv_img.bias"
  Unexpected key(s) in state_dict: "conv_img.0.weight", "conv_img.0.bias"
```

**Root Cause**: 
- Pre-trained models were saved with `upscale: 2` configuration, which uses `nn.Sequential` structure (keys: `conv_img.0.weight/bias`)
- When `upscale <= 1`, code creates `nn.Conv2d` directly (keys: `conv_img.weight/bias`)
- State dict keys don't match, causing loading failure

**Solution Options**:
1. **State dict transformation** (Recommended): Transform keys before loading
   - Detect if model has `conv_img.0.*` keys but code expects `conv_img.*`
   - Rename keys: `conv_img.0.weight` → `conv_img.weight`, `conv_img.0.bias` → `conv_img.bias`
   - Remove `conv_img.1` (Identity layer) if present
   - Implementation: Add transformation function in model loading code

2. **Always use Sequential structure**: Modify `spade_generator.py` to always use `nn.Sequential`
   - When `upscale <= 1`: Use `Sequential(Conv2d, Identity)` instead of direct `Conv2d`
   - Maintains compatibility with saved models
   - `nn.Identity()` has zero performance impact

3. **Model retraining**: Retrain models with `upscale: 1` configuration
   - Most time-consuming option
   - Requires full training pipeline

**Recommended Approach**: Option 1 (state dict transformation) - most flexible and backward compatible

**Implementation**:
```python
def transform_spade_state_dict(state_dict, target_upscale):
    """Transform SPADEDecoder state_dict keys for compatibility"""
    if target_upscale <= 1:
        # Model was saved with upscale=2 (Sequential), but we want upscale=1 (Conv2d)
        new_dict = {}
        for key, value in state_dict.items():
            if key.startswith('conv_img.0.'):
                # Rename conv_img.0.weight -> conv_img.weight
                new_key = key.replace('conv_img.0.', 'conv_img.')
                new_dict[new_key] = value
            elif key.startswith('conv_img.1.'):
                # Skip Identity layer (conv_img.1.*)
                continue
            else:
                new_dict[key] = value
        return new_dict
    return state_dict
```

**Testing**:
- Verify model loads correctly with `upscale: 1`
- Verify model loads correctly with `upscale: 2` (backward compatibility)
- Test inference quality with both configurations
- Verify no performance degradation

**Benefits**:
- Enables `upscale: 1` configuration without model retraining
- Maintains backward compatibility with existing models
- Flexible for future configuration changes

---

### 2. Emotion Guidance Scale Integration

**Status**: Planned  
**Priority**: High  
**Estimated Effort**: 2-4 hours  
**Related**: [emotion-guidance-scale-plan.md](./emotion-guidance-scale-plan.md)

**Problem**: Current `emo_scale` parameter causes artifacts when increased above 2.0, limiting emotion expressiveness compared to DICE-Talk.

**Solution**: Implement separate `emo_guidance_scale` parameter that applies guidance scale specifically for emotions in the CFG (Classifier-Free Guidance) mechanism, similar to DICE-Talk's approach.

**Benefits**:
- More expressive emotions without artifacts
- Maintains audio-video synchronization
- Backward compatible with existing configs
- Fine-grained control over emotion strength

**Implementation**:
- Add `emo_guidance_scale` parameter to `runpod_config.yaml`
- Modify CFG formula in `talking_head_diffusion.py`
- Update pipeline configuration merging

**Testing**:
- Verify audio-video sync maintained
- Compare emotion expressiveness with/without parameter
- Test edge cases and artifact thresholds

---

### 2. Adaptive Emotion Guidance Scale

**Status**: Future Enhancement  
**Priority**: Medium  
**Estimated Effort**: 4-6 hours  
**Dependencies**: Emotion Guidance Scale Integration (#1)

**Problem**: Fixed guidance scale may not be optimal for all scenarios (different emotions, audio characteristics, video lengths).

**Solution**: Make `emo_guidance_scale` adaptive based on:
- Emotion type (some emotions need more/less guidance)
- Audio characteristics (loud vs quiet speech)
- Video length (longer videos may need different scaling)

**Implementation Ideas**:
```python
def get_adaptive_emo_guidance_scale(base_scale, emotion_type, audio_level, video_length):
    # Adjust based on emotion type
    emotion_multipliers = {
        'happy': 1.1,
        'sad': 0.9,
        'angry': 1.2,
        'surprised': 1.0
    }
    
    # Adjust based on audio level
    if audio_level > threshold:
        audio_multiplier = 0.95  # Slightly reduce for loud audio
    else:
        audio_multiplier = 1.05  # Slightly increase for quiet audio
    
    return base_scale * emotion_multipliers.get(emotion_type, 1.0) * audio_multiplier
```

---

### 3. Per-Emotion Guidance Scales

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 2-3 hours  
**Dependencies**: Emotion Guidance Scale Integration (#1)

**Problem**: Different emotions may require different guidance strengths for optimal expressiveness.

**Solution**: Allow configuration of different guidance scales per emotion type.

**Configuration Example**:
```yaml
motion_processor:
  emo_guidance_scale:
    default: 2.0
    happy: 2.5
    sad: 1.8
    angry: 2.2
    surprised: 2.0
    neutral: 1.0
```

**Benefits**:
- Fine-tuned control per emotion
- Better expressiveness for each emotion type
- Can optimize based on user feedback

---

## Performance Improvements

### 4. Three-Way CFG Implementation

**Status**: Advanced Feature  
**Priority**: Medium  
**Estimated Effort**: 8-12 hours  
**Dependencies**: Emotion Guidance Scale Integration (#1)

**Problem**: Current two-way CFG (unconditional vs conditional) doesn't allow separate control of audio and emotion influence, like DICE-Talk's three-way approach.

**Solution**: Implement full DICE-Talk-style three-way CFG:
- Unconditional prediction
- Audio-only prediction (emotion dropped)
- Full conditional prediction (audio + emotion)

**Benefits**:
- More precise control over audio vs emotion influence
- Closer match to DICE-Talk's architecture
- Better separation of concerns

**Implementation Requirements**:
- Modify `sample_subclip` to handle three predictions
- Update `TalkingHeadDiT.forward` to support emotion dropout
- Implement three-way CFG formula:
  ```python
  pred = uncond_pred + \
         audio_guidance_scale * (audio_pred - uncond_pred) + \
         emo_guidance_scale * (full_pred - audio_pred)
  ```

**Challenges**:
- More complex implementation
- Requires emotion dropout mechanism
- Higher memory usage (3x predictions)

---

### 5. Time-Varying Guidance Scale

**Status**: Research Phase  
**Priority**: Low  
**Estimated Effort**: 6-8 hours

**Problem**: Fixed guidance scale throughout diffusion may not be optimal. Early steps (coarse structure) may benefit from different guidance than later steps (fine details).

**Solution**: Apply different guidance scales at different diffusion timesteps:
- Higher guidance early in denoising (coarse structure)
- Lower guidance later in denoising (fine details)

**Implementation Idea**:
```python
def get_timestep_guidance_scale(base_scale, timestep, total_steps):
    # Higher guidance early, lower guidance late
    progress = timestep / total_steps
    early_boost = 1.2  # 20% more guidance early
    late_reduction = 0.9  # 10% less guidance late
    
    if progress > 0.7:  # Last 30% of steps
        return base_scale * late_reduction
    else:  # First 70% of steps
        return base_scale * early_boost
```

**Benefits**:
- Better control over generation process
- Potentially better quality
- More stable training/inference

---

### 6. Batch Processing Optimization

**Status**: Future Enhancement  
**Priority**: Medium  
**Estimated Effort**: 4-6 hours

**Problem**: Current batch processing may not be optimal for all GPU memory sizes and video lengths.

**Solution**: Implement adaptive batch sizing based on:
- Available GPU memory
- Video length
- Model size
- Current memory usage

**Implementation**:
- Dynamic batch size calculation
- Memory-aware processing
- Automatic fallback on OOM errors

---

## Quality Improvements

### 7. Enhanced Face Quality with Laplacian Blending

**Status**: Documented, Not Implemented  
**Priority**: Medium  
**Estimated Effort**: 6-8 hours  
**Related**: [face-quality-improvement.md](./face-quality-improvement.md)

**Problem**: Face quality can be improved with better blending techniques.

**Solution**: Implement Laplacian pyramid blending for smoother face transitions and better quality.

**Benefits**:
- Smoother face transitions
- Better visual quality
- Reduced artifacts

---

### 8. Multi-Resolution Processing

**Status**: Research Phase  
**Priority**: Low  
**Estimated Effort**: 10-15 hours

**Problem**: Processing at single resolution may not be optimal for all face sizes and video qualities.

**Solution**: Implement multi-resolution processing:
- Process different face regions at different resolutions
- Adaptive resolution based on face size
- Quality-aware upscaling

**Benefits**:
- Better quality for different face sizes
- More efficient processing
- Adaptive quality

---

## Architecture Improvements

### 9. Modular Emotion Processing

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 4-6 hours

**Problem**: Emotion processing is tightly coupled with the main pipeline.

**Solution**: Create modular emotion processing system:
- Pluggable emotion adapters
- Easy to add new emotion models
- Support for multiple emotion systems

**Benefits**:
- Easier to experiment with different approaches
- Better code organization
- Easier maintenance

---

### 10. Emotion Model Fine-Tuning

**Status**: Research Phase  
**Priority**: Low  
**Estimated Effort**: 20-30 hours

**Problem**: Current emotion model is pre-trained and may not be optimal for all use cases.

**Solution**: Implement fine-tuning pipeline for emotion model:
- Collect emotion-labeled data
- Fine-tune on specific use cases
- Evaluate and iterate

**Benefits**:
- Better emotion expressiveness
- Customized for specific needs
- Continuous improvement

---

## Integration Improvements

### 11. Real-Time Processing Support

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 15-20 hours

**Problem**: Current implementation is optimized for batch processing, not real-time.

**Solution**: Implement real-time processing mode:
- Streaming audio processing
- Incremental video generation
- Low-latency pipeline

**Benefits**:
- Real-time applications
- Interactive use cases
- Lower latency

---

### 12. Multi-Language Emotion Support

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 8-12 hours

**Problem**: Emotion processing may be optimized for specific languages.

**Solution**: Add multi-language emotion support:
- Language-specific emotion models
- Cross-language emotion mapping
- Adaptive emotion processing

**Benefits**:
- Better results for different languages
- Broader applicability
- Improved user experience

---

## Monitoring and Observability

### 13. Emotion Expressiveness Metrics

**Status**: Future Enhancement  
**Priority**: Medium  
**Estimated Effort**: 4-6 hours

**Problem**: No quantitative metrics to measure emotion expressiveness.

**Solution**: Implement metrics for:
- Emotion intensity measurement
- Expression quality scoring
- User satisfaction tracking

**Benefits**:
- Data-driven improvements
- A/B testing capabilities
- Quality monitoring

---

### 14. Performance Profiling

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 3-4 hours

**Problem**: Limited visibility into performance bottlenecks.

**Solution**: Add comprehensive profiling:
- Emotion processing time
- CFG computation time
- Memory usage tracking
- GPU utilization

**Benefits**:
- Identify bottlenecks
- Optimize performance
- Better resource planning

---

## Testing Improvements

### 15. Automated Emotion Quality Tests

**Status**: Future Enhancement  
**Priority**: Medium  
**Estimated Effort**: 6-8 hours

**Problem**: Manual testing of emotion quality is time-consuming and subjective.

**Solution**: Implement automated tests:
- Emotion expressiveness scoring
- Artifact detection
- Synchronization verification
- Regression testing

**Benefits**:
- Faster iteration
- Consistent quality
- Early problem detection

---

### 16. Emotion Dataset Creation

**Status**: Research Phase  
**Priority**: Low  
**Estimated Effort**: 20-30 hours

**Problem**: Limited emotion-labeled data for evaluation and fine-tuning.

**Solution**: Create comprehensive emotion dataset:
- Collect diverse emotion samples
- Label with emotion types
- Create evaluation benchmarks

**Benefits**:
- Better evaluation
- Fine-tuning data
- Benchmarking

---

## Documentation Improvements

### 17. Emotion Tuning Guide

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 2-3 hours

**Problem**: Users may not know how to tune emotion parameters effectively.

**Solution**: Create comprehensive tuning guide:
- Parameter explanations
- Recommended values
- Use case examples
- Troubleshooting

**Benefits**:
- Better user experience
- Reduced support burden
- Faster adoption

---

### 18. API Documentation

**Status**: Future Enhancement  
**Priority**: Low  
**Estimated Effort**: 4-6 hours

**Problem**: Limited API documentation for emotion-related features.

**Solution**: Create comprehensive API docs:
- Parameter descriptions
- Example requests
- Response formats
- Error handling

**Benefits**:
- Easier integration
- Better developer experience
- Reduced errors

---

## Implementation Priority Matrix

| Improvement | Priority | Effort | Impact | Dependencies |
|------------|---------|--------|--------|--------------|
| SPADEDecoder Compatibility | High | 2-3h | High | None |
| Emotion Guidance Scale | High | 2-4h | High | None |
| Adaptive Guidance Scale | Medium | 4-6h | Medium | #1 |
| Per-Emotion Scales | Low | 2-3h | Medium | #1 |
| Three-Way CFG | Medium | 8-12h | High | #1 |
| Time-Varying Guidance | Low | 6-8h | Medium | #1 |
| Batch Optimization | Medium | 4-6h | Medium | None |
| Face Quality | Medium | 6-8h | High | None |
| Emotion Metrics | Medium | 4-6h | Medium | None |
| Automated Tests | Medium | 6-8h | High | None |

## Next Steps

1. **Immediate**: Fix SPADEDecoder state_dict compatibility (#1)
2. **Short-term**: Implement Emotion Guidance Scale Integration (#2)
3. **Short-term**: Add emotion expressiveness metrics (#13)
4. **Medium-term**: Implement adaptive guidance scale (#2)
5. **Long-term**: Research three-way CFG (#4) and time-varying guidance (#5)

## Contributing

If you'd like to contribute to any of these improvements:

1. Check the related documentation (if available)
2. Review existing code structure
3. Create a detailed implementation plan
4. Submit a pull request with tests

## Notes

- Priorities may change based on user feedback and requirements
- Estimated effort is approximate and may vary
- Some improvements may be combined or split based on implementation
- Dependencies should be considered when planning work

---

**Last Updated**: 2024  
**Maintainer**: MoDA Development Team

