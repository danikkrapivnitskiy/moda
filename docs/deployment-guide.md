# MoDA Docker Deployment Guide

Complete step-by-step guide for deploying MoDA to RunPod via Docker Hub.

## Prerequisites

- [x] Docker Desktop installed and running
- [x] HuggingFace account with access token
- [x] Docker Hub account (username: `your-dockerhub-username`)
- [x] Models downloaded locally

## Step 1: Upload Models to HuggingFace

### 1.1 Download Original Models

First, run MoDA locally to download models:

```bash
cd MoDA

# Install dependencies if needed
pip install -r requirements.txt

# Run inference to download models (they go to pretrain_weights/)
python src/models/inference/moda_test.py \
  --image_path src/examples/reference_images/6.jpg \
  --audio_path src/examples/driving_audios/5.wav
```

This will download models to `pretrain_weights/` directory.

### 1.2 Install HuggingFace CLI

```bash
pip install huggingface-hub
```

### 1.3 Login to HuggingFace

```bash
huggingface-cli login
# Enter your HF token: hf_xxxxxxxxxxxxx
```

### 1.4 Create Repository and Upload Models

```bash
# Create repository (do this once)
huggingface-cli repo create moda-pretrain-weights --type model

# Upload all pretrain weights
huggingface-cli upload your-huggingface-username/moda-pretrain-weights ./pretrain_weights --repo-type model
```

**Wait for upload to complete** (~5-10 GB, may take 10-30 minutes depending on connection)

### 1.5 Verify Upload

Visit: https://huggingface.co/your-huggingface-username/moda-pretrain-weights

You should see:
- `moda/` directory with checkpoint files
- `decode/v1/` directory with LivePortrait models

## Step 2: Build Docker Image with Pre-downloaded Models

### 2.1 Set Environment Variables

```bash
export DOCKER_USER="your-dockerhub-username"
export HUGGINGFACE_USERNAME="your-huggingface-username"
export HF_TOKEN="hf_xxxxxxxxxxxxx"  # Your HuggingFace token
```

### 2.2 Build the Image

**IMPORTANT**: This build will take 20-30 minutes and download ~5-10 GB of models.

```bash
cd MoDA

# Build for linux/amd64 (required for RunPod)
docker buildx build \
  --platform linux/amd64 \
  --build-arg HUGGINGFACE_USERNAME=your-huggingface-username \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  -t your-dockerhub-username/moda-runpod:latest \
  .
```

**Monitor the build**: Watch for these key stages:
1. ✅ System dependencies installed
2. ✅ PyTorch 2.4.1 verified
3. ✅ Python packages installed
4. ✅ Models pre-downloaded (~10-15 minutes)
5. ✅ Final verification passed

### 2.3 Verify the Build

Check image size (should be ~15-20 GB):

```bash
docker images your-dockerhub-username/moda-runpod:latest
```

Quick test (without GPU):

```bash
docker run --rm --platform linux/amd64 your-dockerhub-username/moda-runpod:latest python3 -c "
import sys
sys.path.insert(0, '/app/src')
from models.inference.moda_test import LiveVASAPipeline
print('✅ MoDA import successful')
"
```

## Step 3: Push to Docker Hub

### 3.1 Login to Docker Hub

```bash
docker login
# Username: your-dockerhub-username
# Password: [your Docker Hub password]
```

### 3.2 Push the Image

```bash
docker push your-dockerhub-username/moda-runpod:latest
```

**Wait for push to complete** (~15-20 GB, may take 20-60 minutes)

### 3.3 Verify on Docker Hub

Visit: https://hub.docker.com/r/your-dockerhub-username/moda-runpod

You should see:
- Tag: `latest`
- Size: ~15-20 GB
- Last pushed: just now

## Step 4: Deploy on RunPod

### 4.1 Create Serverless Endpoint

1. Go to https://www.runpod.io/serverless
2. Click "Create Endpoint"
3. Fill in details:
   - **Name**: `moda-runpod`
   - **Container Image**: `your-dockerhub-username/moda-runpod:latest`
   - **Container Disk**: 100 GB (for models and temporary files)
   - **GPU Types**: RTX 4090 or A100
   - **Min Workers**: 0 (auto-scale)
   - **Max Workers**: 10 (or your desired limit)

4. **Environment Variables**:
   ```
   HUGGINGFACE_USERNAME=your-huggingface-username
   ```
   
   Note: HF_TOKEN not needed if models are pre-built in image

5. Click "Create Endpoint"

### 4.2 Wait for Deployment

Wait 2-5 minutes for:
- Endpoint to be created
- First worker to start (cold start)
- Models to be verified

### 4.3 Test the Endpoint

Get your endpoint ID from RunPod dashboard, then:

```python
import runpod
import base64

# Set your RunPod API key
runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Initialize endpoint
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Test CUDA first
cuda_result = endpoint.run_sync({
    "input": {
        "action": "check_cuda"
    }
})
print("CUDA Status:", cuda_result["status"])
print("GPU:", cuda_result["cuda_info"]["gpu_details"])

# Load test image and audio
with open("src/examples/reference_images/6.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
with open("src/examples/driving_audios/5.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Generate video
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "audio": audio_b64,
        # MoDA-specific parameters (optional):
        "cfg_scale": 1.0,   # Guidance scale (0.5-2.0)
        "emo": 8,           # Emotion code (0-7 specific, 8=neutral)
        "smooth": False     # Smooth motion transitions
        # Note: output_fps and batch_size are configured in YAML files
    }
})

# Save result
if "video" in result:
    video_data = base64.b64decode(result["video"])
    with open("output.mp4", "wb") as f:
        f.write(video_data)
    print("✅ Video generated successfully!")
else:
    print("❌ Error:", result.get("error"))
```

## Step 5: Monitor and Scale

### 5.1 Monitor Performance

RunPod dashboard shows:
- Request count
- Average execution time (~10-30 seconds per video)
- GPU utilization
- Worker auto-scaling

### 5.2 Adjust Settings

If needed, adjust in RunPod:
- **Max Workers**: Increase for higher concurrency
- **Timeout**: Increase if videos take longer
- **Container Disk**: Increase if running out of space

### 5.3 Cost Optimization

- **Auto-scale**: Workers spin down when idle (save costs)
- **GPU selection**: RTX 4090 cheaper than A100 for most workloads
- **Timeout**: Set appropriate timeout to avoid hanging workers

## Troubleshooting

### Build Issues

**Issue**: `HUGGINGFACE_USERNAME build argument is not set`
**Solution**: Add `--build-arg HUGGINGFACE_USERNAME=your-huggingface-username`

**Issue**: `HF_TOKEN build argument is not set`
**Solution**: Add `--build-arg HF_TOKEN="${HF_TOKEN}"`

**Issue**: Model download fails during build
**Solution**: Check HF_TOKEN is valid and has access to repositories

### RunPod Issues

**Issue**: "CUDA not available"
**Solution**: Ensure GPU-enabled RunPod plan, check endpoint GPU settings

**Issue**: "Models not found"
**Solution**: 
- Check HUGGINGFACE_USERNAME environment variable in RunPod
- Or add HF_TOKEN if models not pre-built in image

**Issue**: Out of memory
**Solution**: 
- Use GPU with more VRAM (24GB+)
- Reduce batch_size in request

**Issue**: Timeout
**Solution**: Increase timeout in RunPod endpoint settings (60+ seconds)

## Quick Commands Reference

```bash
# Build image with models
docker buildx build --platform linux/amd64 \
  --build-arg HUGGINGFACE_USERNAME=your-huggingface-username \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  -t your-dockerhub-username/moda-runpod:latest .

# Push to Docker Hub
docker push your-dockerhub-username/moda-runpod:latest

# Test locally (without GPU)
docker run --rm your-dockerhub-username/moda-runpod:latest python3 -c "import sys; sys.path.insert(0, '/app/src'); from models.inference.moda_test import LiveVASAPipeline; print('OK')"

# Upload models to HuggingFace
huggingface-cli upload your-huggingface-username/moda-pretrain-weights ./pretrain_weights --repo-type model
```

## Next Steps

After successful deployment:
1. ✅ Test endpoint with multiple requests
2. ✅ Monitor performance and costs
3. ✅ Adjust worker settings as needed
4. ✅ Integrate into your application
5. ✅ Set up monitoring/alerting

## Post-Deployment Verification

After deploying fixes, verify all issues are resolved:

### Critical Issues
- [ ] Docker build completes without HUGGINGFACE_USERNAME error
- [ ] Models are downloaded during build (check build logs)
- [ ] Symlinks created successfully (no permission errors)

### Runtime Issues
- [ ] First request completes in <30 seconds
- [ ] Multiple concurrent requests work without race conditions
- [ ] Audio conversion handles all formats correctly
- [ ] Video quality matches configuration (crf=25, preset=faster)
- [ ] Disk space checks prevent out-of-space errors

### Cleanup & Resources
- [ ] No temporary files left after generation
- [ ] GPU memory cleared after each request
- [ ] Graceful shutdown on SIGTERM/SIGINT
- [ ] Error messages are clear and structured

### Configuration
- [ ] Device ID can be overridden via CUDA_DEVICE_ID env var
- [ ] Resource limits configurable via runpod_config.yaml
- [ ] All three video config files have matching values

## Support

- Docker issues: Check Docker Desktop logs
- HuggingFace issues: https://huggingface.co/docs
- RunPod issues: https://docs.runpod.io
- MoDA issues: https://github.com/lixinyyang/MoDA

