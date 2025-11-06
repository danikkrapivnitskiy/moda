# MoDA Docker Setup - Complete ✅

All files have been created following the DICE-Talk pattern. Your MoDA project is now ready for Docker deployment!

## ✅ What Was Done

### 1. Docker Configuration Files

- **`Dockerfile`** ✅
  - Base image: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`
  - PyTorch 2.4.1 + torchvision 0.19.1 + xformers 0.0.28.post1
  - All verification checks from DICE-Talk
  - Pre-downloads models from HuggingFace during build
  - Non-root user (appuser) for security
  - Expected image size: ~15-20 GB

- **`requirements.txt`** ✅
  - Synchronized with DICE-Talk versions
  - Compatible with PyTorch 2.4.1
  - All MoDA-specific dependencies included

- **`.dockerignore`** ✅
  - Optimized for faster builds
  - Excludes unnecessary files

- **`.gitignore`** ✅
  - Ignores models, cache, and build artifacts

- **`.gitattributes`** ✅
  - Proper line endings and binary file handling

### 2. RunPod Integration

- **`runpod_server.py`** ✅
  - Serverless handler for RunPod
  - 3-tier model loading: persistent storage → pre-built cache → download
  - CUDA diagnostics endpoint
  - Error handling and cleanup
  - Base64 image/audio input, video output

- **`runpod_config.yaml`** ✅
  - Inference parameters (fps, batch_size, etc.)
  - Model paths configuration
  - Easy customization

### 3. Build & Deploy Scripts

All scripts in `scripts/` directory are executable:

- **`build_docker.sh`** ✅ - Interactive build with push option
- **`quick_push.sh`** ✅ - Quick test and push
- **`deploy.sh`** ✅ - Interactive deployment menu
- **`auto_build_and_push.sh`** ✅ - Automated CI/CD
- **`build.sh`** ✅ - Simple build
- **`clean_docker_cache.sh`** ✅ - Clean Docker cache

### 4. Documentation

- **`README.md`** ✅ - Updated with full Docker deployment section
- **`DEPLOYMENT_GUIDE.md`** ✅ - Step-by-step deployment instructions

## 🚀 Quick Start

### Option 1: Automated Build & Push

```bash
cd /Users/daniilkrapiunitski/Projects/tg_bot/GPU/MoDA

# Set credentials
export DOCKER_USER="krapiunitski12"
export HUGGINGFACE_USERNAME="krapiunitski"
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Build and push
./scripts/auto_build_and_push.sh
```

### Option 2: Manual Build

```bash
cd /Users/daniilkrapiunitski/Projects/tg_bot/GPU/MoDA

# Build with models pre-downloaded
docker buildx build \
  --platform linux/amd64 \
  --build-arg HUGGINGFACE_USERNAME=krapiunitski \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  -t krapiunitski12/moda-runpod:latest \
  .

# Push to Docker Hub
docker push krapiunitski12/moda-runpod:latest
```

## 📋 Pre-Deployment Checklist

Before building, make sure you have:

- [ ] **HuggingFace Models Uploaded**
  - Repository: `krapiunitski/moda-pretrain-weights`
  - Contains: `moda/` and `decode/` directories
  - Upload command: `huggingface-cli upload krapiunitski/moda-pretrain-weights ./pretrain_weights --repo-type model`

- [ ] **HuggingFace Token**
  - Valid token with read access to your repositories
  - Set as `HF_TOKEN` environment variable

- [ ] **Docker Hub Access**
  - Logged in: `docker login`
  - Username: `krapiunitski12`

- [ ] **Docker Desktop**
  - Running and has sufficient resources
  - ~30 GB free space for build

## 🔧 Key Configuration Points

### HuggingFace Repositories

The Dockerfile expects models at:
- **Repository**: `krapiunitski/moda-pretrain-weights`
- **Structure**:
  ```
  moda-pretrain-weights/
  ├── moda/
  │   └── net-200.pth
  └── decode/
      └── v1/
          ├── first_stage/
          └── retargeting_models/
  ```

### Docker Hub Image

- **Image**: `krapiunitski12/moda-runpod:latest`
- **Platform**: `linux/amd64` (required for RunPod)
- **Size**: ~15-20 GB (with models)

### RunPod Configuration

When deploying to RunPod:
1. **Docker Image**: `krapiunitski12/moda-runpod:latest`
2. **Environment Variables**:
   ```
   HUGGINGFACE_USERNAME=krapiunitski
   ```
3. **GPU**: RTX 4090 or A100 recommended
4. **Disk**: 100GB+
5. **Max Workers**: 1 per GPU (or as needed)

## 📊 Expected Build Times

- **Full build** (first time): 20-30 minutes
  - System setup: 2-3 minutes
  - Python packages: 5-10 minutes
  - Model download: 10-15 minutes
  - Verification: 2-3 minutes

- **Rebuild** (with cache): 5-10 minutes
  - Only changed layers rebuild

- **Push to Docker Hub**: 20-60 minutes
  - Depends on upload speed (~15-20 GB)

## 🔍 Verification Steps

After building, verify:

```bash
# Check image exists
docker images krapiunitski12/moda-runpod:latest

# Test imports (no GPU needed)
docker run --rm --platform linux/amd64 \
  krapiunitski12/moda-runpod:latest \
  python3 -c "
import sys
sys.path.insert(0, '/app/src')
from models.inference.moda_test import LiveVASAPipeline
print('✅ MoDA imports successfully')
"

# Check image size (should be ~15-20 GB)
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep moda-runpod
```

## 📚 Key Differences from DICE-Talk

1. **Base Image**: PyTorch 2.4.1 (vs 2.4.1 in DICE-Talk - same now!)
2. **Models**: MoDA pretrain weights instead of DICE-Talk checkpoints
3. **Structure**: Uses `pretrain_weights/` instead of `checkpoints/`
4. **Pipeline**: LiveVASAPipeline instead of DICE_Talk
5. **Input**: No emotion parameter (handled internally by MoDA)

## 🐛 Common Issues & Solutions

### Build Issues

**Issue**: Models not downloading during build
```bash
# Solution: Check HF_TOKEN is valid
echo $HF_TOKEN
# Should output: hf_xxxxxxxxxxxxx
```

**Issue**: "platform does not match" on Mac
```bash
# Solution: Always use --platform linux/amd64
docker buildx build --platform linux/amd64 ...
```

### Push Issues

**Issue**: "denied: requested access to the resource is denied"
```bash
# Solution: Login to Docker Hub
docker login
# Username: krapiunitski12
```

### Runtime Issues

See `DEPLOYMENT_GUIDE.md` for complete troubleshooting.

## 📁 File Structure

```
MoDA/
├── Dockerfile                 # ✅ Main Docker image definition
├── requirements.txt           # ✅ Python dependencies
├── runpod_server.py          # ✅ RunPod serverless handler
├── runpod_config.yaml        # ✅ RunPod configuration
├── app.py                    # Gradio web interface
├── .dockerignore             # ✅ Docker build exclusions
├── .gitignore                # ✅ Git exclusions
├── .gitattributes            # ✅ Git file handling
├── README.md                 # ✅ Updated with Docker docs
├── DEPLOYMENT_GUIDE.md       # ✅ Step-by-step deployment
├── SETUP_COMPLETE.md         # ✅ This file
└── scripts/                  # ✅ Build & deploy scripts
    ├── build_docker.sh       # Interactive build
    ├── quick_push.sh         # Quick push
    ├── deploy.sh             # Deployment menu
    ├── auto_build_and_push.sh # Automated CI/CD
    ├── build.sh              # Simple build
    └── clean_docker_cache.sh # Cache cleanup
```

## 🎯 Next Steps

1. **Upload Models to HuggingFace** (if not done yet):
   ```bash
   huggingface-cli login
   huggingface-cli upload krapiunitski/moda-pretrain-weights ./pretrain_weights --repo-type model
   ```

2. **Build Docker Image**:
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxx"
   ./scripts/build_docker.sh
   ```

3. **Push to Docker Hub**:
   ```bash
   docker push krapiunitski12/moda-runpod:latest
   ```

4. **Deploy on RunPod**:
   - Follow instructions in `DEPLOYMENT_GUIDE.md`

5. **Test the Endpoint**:
   - Use example code in `README.md` or `DEPLOYMENT_GUIDE.md`

## ✨ Summary

Your MoDA project now has:
- ✅ Production-ready Dockerfile (PyTorch 2.4.1)
- ✅ RunPod serverless integration
- ✅ All build & deploy scripts
- ✅ Complete documentation
- ✅ Git configuration files
- ✅ Following DICE-Talk patterns

**Ready to build and deploy!** 🚀

For detailed instructions, see:
- **Quick start**: This file
- **Step-by-step deployment**: `DEPLOYMENT_GUIDE.md`
- **Docker usage**: `README.md` Docker section

