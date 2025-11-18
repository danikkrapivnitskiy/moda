# MoDA Improvements - Implementation Guide

## Overview

This document describes planned and implemented improvements for the MoDA project, covering error handling, monitoring, performance optimizations, and production-ready features.

## Table of Contents

1. [OOM Prevention Tracking](#oom-prevention-tracking)
2. [Error Handling Improvements](#error-handling-improvements)
3. [Production Features](#production-features)
4. [Performance Optimizations](#performance-optimizations)
5. [Monitoring and Observability](#monitoring-and-observability)

---

## OOM Prevention Tracking

### Problem Statement

The system has automatic OOM prevention mechanisms:
- Automatic batch size reduction on CUDA OOM errors (with retries)
- Emergency batch size reduction due to high memory fragmentation
- Automatic cleanup and memory management

However, when these mechanisms activate successfully (preventing OOM without errors), the main service has no way to know that:
1. Batch size was automatically reduced
2. Performance may be impacted
3. Memory pressure was detected

### Solution

Add OOM prevention tracking that collects information about batch size adjustments and includes it in the successful response, allowing the main service to:
- Monitor memory pressure patterns
- Log performance impacts
- Make informed decisions about resource allocation
- Provide visibility into system health

### Implementation

#### 1. Track OOM Prevention in MotionProcesser

**File**: `src/datasets/preprocess/extract_features/motion_processer.py`

Add tracking structure at the beginning of `driven()` method:

```python
def driven(self, f_s, x_s_info, s_lmk, c_s_eyes_lst, kp_infos, c_d_eyes_lst=None, c_d_lip_lst=None, smooth=False):
    # ... existing code ...
    
    # Track OOM prevention mechanism activation
    oom_prevention_info = {
        "activated": False,
        "original_batch_size": None,
        "final_batch_size": None,
        "reductions_count": 0,
        "reduction_reasons": []  # List of reasons: "oom_retry", "fragmentation_emergency"
    }
    
    # Store original batch size
    original_batch_size = warp_batch_size
    oom_prevention_info["original_batch_size"] = original_batch_size
```

Track when batch size is reduced during OOM retry:

```python
# After successful retry (around line 1091)
if retry_count > 0 and batch_batch_size < warp_batch_size:
    warp_batch_size = batch_batch_size
    current_batch_size = batch_batch_size
    
    # Track OOM prevention
    if not oom_prevention_info["activated"]:
        oom_prevention_info["activated"] = True
    oom_prevention_info["final_batch_size"] = warp_batch_size
    oom_prevention_info["reductions_count"] += 1
    oom_prevention_info["reduction_reasons"].append({
        "reason": "oom_retry",
        "batch_range": f"{start}-{end}",
        "retry_count": retry_count,
        "new_batch_size": batch_batch_size
    })
```

Track emergency fragmentation-based reduction:

```python
# Emergency reduction due to fragmentation (around line 1139)
if fragmentation > 20:
    new_batch_size = max(16, int(warp_batch_size * reduction_factor))
    if new_batch_size < warp_batch_size:
        warp_batch_size = new_batch_size
        current_batch_size = warp_batch_size
        
        # Track fragmentation-based reduction
        if not oom_prevention_info["activated"]:
            oom_prevention_info["activated"] = True
        oom_prevention_info["final_batch_size"] = warp_batch_size
        oom_prevention_info["reductions_count"] += 1
        oom_prevention_info["reduction_reasons"].append({
            "reason": "fragmentation_emergency",
            "fragmentation_gb": fragmentation,
            "new_batch_size": warp_batch_size
        })
```

Store tracking info in class attribute at the end of method:

```python
# At the end of driven() method, before return
self.last_oom_prevention_info = oom_prevention_info
```

#### 2. Add Class Attribute

**File**: `src/datasets/preprocess/extract_features/motion_processer.py`

In `__init__` method:

```python
def __init__(self, cfg_path, device_id=0) -> None:
    # ... existing code ...
    self.last_oom_prevention_info = None  # Track OOM prevention info
```

#### 3. Pass Info Through Pipeline

**File**: `src/models/inference/moda_test.py`

In `driven_sample()` method, after calling `driven_by_audio()`:

```python
def driven_sample(self, image_path: str, audio_path: str, cfg_scale: float=1., emo=8, save_dir=None, smooth=False, silent_audio_path = None, silent_mode="post"):
    # ... existing code ...
    
    self.motion_processer.driven_by_audio(source_rgb_lst[0], kp_infos, save_path, ori_audio_path, smooth=smooth)
    
    # Get OOM prevention info if available
    oom_prevention_info = getattr(self.motion_processer, 'last_oom_prevention_info', None)
    
    # Store in pipeline for access from runpod_server
    if oom_prevention_info and oom_prevention_info.get("activated"):
        self.last_oom_prevention_info = oom_prevention_info
    
    return save_path
```

Add class attribute in `__init__`:

```python
def __init__(self, cfg_path: str, load_motion_generator: bool = True, motion_mean_std_path=None, primary_config=None):
    # ... existing code ...
    self.last_oom_prevention_info = None  # Track OOM prevention info
```

#### 4. Include in RunPod Response

**File**: `runpod_server.py`

After successful video generation:

```python
# Call driven_sample (returns path to generated video)
output_path = pipe.driven_sample(
    image_path=img_path,
    audio_path=aud_path,
    cfg_scale=cfg_scale,
    emo=emo,
    save_dir=save_dir,
    smooth=smooth
)
timing['inference'] = time.time() - start

print(f"   Generated video: {output_path}")

# Check for OOM prevention activation
oom_prevention_info = getattr(pipe, 'last_oom_prevention_info', None)
response_data = {
    "video": video_base64,
    "timing": timing
}

# Add OOM prevention info if mechanism was activated
if oom_prevention_info and oom_prevention_info.get("activated"):
    response_data["oom_prevention"] = {
        "activated": True,
        "original_batch_size": oom_prevention_info.get("original_batch_size"),
        "final_batch_size": oom_prevention_info.get("final_batch_size"),
        "reductions_count": oom_prevention_info.get("reductions_count"),
        "reduction_reasons": oom_prevention_info.get("reduction_reasons", []),
        "message": f"Batch size was automatically reduced from {oom_prevention_info.get('original_batch_size')} to {oom_prevention_info.get('final_batch_size')} to prevent OOM"
    }
    print(f"[INFO] OOM prevention mechanism activated: batch_size reduced from {oom_prevention_info.get('original_batch_size')} to {oom_prevention_info.get('final_batch_size')}")

return response_data
```

### Response Format

#### Success with OOM Prevention

```json
{
  "video": "base64_encoded_mp4",
  "timing": {
    "validation": 0.05,
    "decode": 0.12,
    "audio_conversion": 0.02,
    "inference": 15.34,
    "encode": 0.18,
    "total": 15.71
  },
  "oom_prevention": {
    "activated": true,
    "original_batch_size": 80,
    "final_batch_size": 40,
    "reductions_count": 2,
    "reduction_reasons": [
      {
        "reason": "oom_retry",
        "batch_range": "0-80",
        "retry_count": 1,
        "new_batch_size": 40
      },
      {
        "reason": "fragmentation_emergency",
        "fragmentation_gb": 22.5,
        "new_batch_size": 32
      }
    ],
    "message": "Batch size was automatically reduced from 80 to 32 to prevent OOM"
  }
}
```

#### Success without OOM Prevention

```json
{
  "video": "base64_encoded_mp4",
  "timing": {
    "validation": 0.05,
    "inference": 15.34,
    "total": 15.71
  }
}
```

### Client-Side Usage

#### Python Example

```python
import runpod

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "audio": audio_b64
    }
})

# Check for OOM prevention (not error, but warning)
if "oom_prevention" in result:
    oom_info = result["oom_prevention"]
    print(f"⚠️ OOM Prevention: {oom_info['message']}")
    print(f"   Batch size: {oom_info['original_batch_size']} → {oom_info['final_batch_size']}")
    print(f"   Reductions: {oom_info['reductions_count']}")
    
    # Log for monitoring
    logger.warning("OOM prevention activated", extra={
        "original_batch_size": oom_info['original_batch_size'],
        "final_batch_size": oom_info['final_batch_size'],
        "reductions": oom_info['reductions_count']
    })
    
    # Optional: Adjust future requests based on pattern
    if oom_info['reductions_count'] > 3:
        # Consider using smaller default batch_size for this user/video type
        pass

# Video is still available
video_data = base64.b64decode(result["video"])
```

### Configuration

OOM prevention behavior is controlled in `runpod_config.yaml`:

```yaml
compatibility:
  auto_adjust_batch_size_on_oom: True  # Enable automatic batch size reduction
  max_oom_retries: 3                   # Max retries with reduced batch_size
```

### Status

**Implementation Status**: Planned  
**Priority**: Medium  
**Estimated Effort**: 2-3 hours

---

## Error Handling Improvements

### Structured Error Responses

**Problem**: Inconsistent error formats made debugging difficult

**Solution**: Implement structured error responses with consistent format

**Implementation**:

```python
def create_error_response(error_type, message, details=None):
    """Create structured error response for consistent error handling"""
    response = {
        "error": message,
        "error_type": error_type,
        "status": "failed"
    }
    
    # Add OOM-specific details
    if "out of memory" in message.lower() or "oom" in message.lower() or error_type == "OOMError":
        response["error_type"] = "OOMError"
        response["is_oom"] = True
        if details is None:
            details = {}
        # Add memory stats if available
        if torch.cuda.is_available():
            details["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
    
    if details:
        response["details"] = details
    return response
```

**Usage**:

```python
# Validation error
return create_error_response(
    "ValidationError",
    "Missing required fields",
    {"required": ["image", "audio"], "provided": list(input_data.keys())}
)

# OOM error with memory stats
return create_error_response("OOMError", "CUDA out of memory", details)
```

**Benefits**:
- Consistent error format across all error types
- Better client debugging with structured details
- OOM errors include GPU memory statistics
- Easy to parse and handle programmatically

---

## Production Features

### 1. Graceful Shutdown Handling

**Problem**: Containers killed without cleanup could leave GPU memory allocated

**Solution**: Register signal handlers for graceful shutdown

```python
import signal
import sys

def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n[INFO] Received signal {signum}, shutting down gracefully...")
    
    # Clear GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[INFO] GPU memory cleared on shutdown")
    except Exception as e:
        print(f"[WARN] Failed to clear GPU on shutdown: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)
```

### 2. Disk Space Monitoring

**Problem**: Out of disk space errors during video generation

**Solution**: Check available disk space before operations

```python
def check_disk_space(path="/workspace", min_gb=10):
    """Check available disk space before operations"""
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)
        
        if available_gb < min_gb:
            raise RuntimeError(
                f"Insufficient disk space: {available_gb:.1f}GB available, "
                f"minimum {min_gb}GB required"
            )
        
        print(f"[INFO] Disk space: {available_gb:.1f}GB available")
        return available_gb
    except Exception as e:
        print(f"[WARN] Could not check disk space: {e}")
        return None
```

### 3. File Locking for Multi-Worker Safety

**Problem**: Multiple workers downloading models simultaneously could corrupt files

**Solution**: Use file locking to prevent concurrent access

```python
import fcntl

lock_file = f"/tmp/{model_name}.lock"
lock_fd = None
try:
    lock_fd = open(lock_file, 'w')
    print(f"[INFO] Acquiring lock for {model_name}...")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    print(f"[INFO] Lock acquired for {model_name}")
    
    # Download/copy models safely
    ...
    
finally:
    if lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            print(f"[INFO] Lock released for {model_name}")
        except Exception as e:
            print(f"[WARN] Failed to release lock: {e}")
```

---

## Performance Optimizations

### Memory Fragmentation Management

The system includes automatic memory fragmentation detection and cleanup:

- **Fragmentation monitoring**: Tracks reserved vs allocated GPU memory
- **Automatic cleanup**: Performs aggressive cleanup when fragmentation > 10GB
- **Emergency batch reduction**: Reduces batch size when fragmentation > 20GB
- **Batch-level cleanup**: Fast cleanup after each batch to prevent accumulation

### Batch Size Optimization

- **Dynamic adjustment**: Automatically reduces batch size on OOM
- **Persistence**: Reduced batch size persists for subsequent batches
- **Video length awareness**: Longer videos use smaller batch sizes
- **Retry mechanism**: Up to 3 retries with progressively smaller batches

### Parse Output Optimization (Current Implementation)

**Status**: Implemented and optimized (November 2024)

**Decision**: After testing chunked processing for `parse_output`, it was decided to keep the simpler single-tensor processing approach.

#### Current Implementation

The `parse_output` method processes the entire tensor at once on GPU before transferring to CPU:

1. **GPU Processing** (all operations on GPU):
   - Transpose: `[0, 2, 3, 1]` (1x3xHxW → 1xHxWx3)
   - Clip to 0~1: `torch.clamp(out, 0, 1)`
   - Convert to uint8: `torch.clamp(out * 255, 0, 255).to(torch.uint8)`

2. **Single Transfer**: Transfer only uint8 tensor to CPU (4x less data than float32)

3. **Performance**: 
   - For 1000 frames (2.93GB float32 → 0.73GB uint8): ~0.5-1.0 seconds
   - For 1450 frames (4.25GB float32 → 1.06GB uint8): ~0.7-1.5 seconds

#### Why Not Chunked Processing?

**Tested but not implemented**:

1. **Concatenation bottleneck**: Chunked processing required concatenating chunks on CPU, which was slow:
   - `np.concatenate`: ~17-20 seconds for 1000 frames
   - Pre-allocation + slicing: ~17-20 seconds (no improvement)
   - Memory view copying: Similar performance

2. **No significant VRAM benefit**: 
   - Current VRAM usage: ~26-31% after concatenation
   - Chunked processing didn't reduce peak VRAM usage significantly
   - Cleanup before `parse_output` already reduces VRAM pressure

3. **Simplicity**: Single-tensor processing is:
   - Simpler code (easier to maintain)
   - Faster for typical use cases
   - No concatenation overhead

#### Performance Metrics

**For 1000 frames (40 seconds video)**:
- GPU processing: ~0.1-0.2 seconds
- CPU transfer (uint8): ~0.3-0.8 seconds
- **Total parse_output**: ~0.5-1.0 seconds

**For 1450 frames (58 seconds video)**:
- GPU processing: ~0.15-0.3 seconds
- CPU transfer (uint8): ~0.5-1.2 seconds
- **Total parse_output**: ~0.7-1.5 seconds

#### Key Optimization

The main optimization is **processing on GPU and transferring only uint8**:
- **Before**: Transfer 2.93GB float32 → process on CPU → slow
- **After**: Process on GPU → transfer 0.73GB uint8 → fast (4x less data)

#### Future Considerations

If concatenation performance improves or VRAM usage becomes critical (>90%), chunked processing could be reconsidered. However, current implementation is optimal for:
- Videos up to 1500 frames (60 seconds)
- VRAM usage < 50%
- Fast single-tensor processing

---

## Monitoring and Observability

### OOM Prevention Monitoring

1. **Alert on Frequent Activations**: If OOM prevention activates > 10% of requests, consider:
   - Increasing GPU memory allocation
   - Reducing default batch sizes
   - Optimizing memory usage

2. **Track Reduction Patterns**: Monitor which reduction reasons are most common:
   - `oom_retry`: Actual OOM errors occurred (more serious)
   - `fragmentation_emergency`: Memory fragmentation issues (may need allocator tuning)

3. **Performance Impact**: Track inference time correlation with batch size reductions:
   - Smaller batch sizes = longer inference times
   - May need to adjust timeout expectations

### Error Monitoring

- Track error types and frequencies
- Monitor OOM error rates
- Alert on error spikes
- Track error recovery success rates

---

## Related Documentation

- `runpod_config.yaml` - Configuration for OOM prevention and other settings
- `src/datasets/preprocess/extract_features/motion_processer.py` - OOM retry logic
- [cuda-device-busy-fix.md](./cuda-device-busy-fix.md) - GPU memory management
- [performance-optimization.md](./performance-optimization.md) - Performance tuning
- [production-improvements.md](./production-improvements.md) - Additional production features

---

## Implementation Priority

1. **High Priority**:
   - OOM prevention tracking (visibility into system health)
   - Structured error responses (better debugging)
   - Graceful shutdown (production safety)

2. **Medium Priority**:
   - Disk space monitoring (prevent failures)
   - File locking (multi-worker safety)

3. **Low Priority**:
   - Enhanced monitoring dashboards
   - Performance metrics collection

---

## Summary

These improvements provide:
- **Visibility**: Know when memory pressure occurs, even without errors
- **Reliability**: Better error handling and graceful degradation
- **Production Readiness**: Proper cleanup, monitoring, and safety features
- **Performance**: Optimized memory management and batch processing

All improvements are designed to be backward compatible and safe for production deployment.

