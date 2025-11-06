<h1 align='center'>MoDA: Multi-modal Diffusion Architecture for Talking Head Generation</h1>

<div align="center">

<strong>Authors</strong> <br><br>

Xinyang&nbsp;Li<sup>1,2</sup>,&nbsp;
Gen&nbsp;Li<sup>2</sup>,&nbsp;
Zhihui&nbsp;Lin<sup>1,3</sup>,&nbsp;
Yichen&nbsp;Qian<sup>1,3&nbsp;†</sup>,&nbsp;
Gongxin&nbsp;Yao<sup>2</sup>,&nbsp;
Weinan&nbsp;Jia<sup>1</sup>,&nbsp;
Aowen&nbsp;Wang<sup>1</sup>,&nbsp;
Weihua&nbsp;Chen<sup>1,3</sup>,&nbsp;
Fan&nbsp;Wang<sup>1,3</sup> <br><br>

<sup>1</sup>Xunguang&nbsp;Team,&nbsp;DAMO&nbsp;Academy,&nbsp;Alibaba&nbsp;Group&nbsp;&nbsp;&nbsp;
<sup>2</sup>Zhejiang&nbsp;University&nbsp;&nbsp;&nbsp;
<sup>3</sup>Hupan&nbsp;Lab <br><br>

<sup>†</sup>Corresponding authors: yichen.qyc@alibaba-inc.com,&nbsp;l_xyang@zju.edu.cn

</div>
<br>
<div align='center'>
    <a href='https://lixinyyang.github.io/MoDA.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://arxiv.org/abs/2507.03256'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

##  📂 Updates

* [2025.08.08] 🔥 We release our inference [codes](https://github.com/lixinyyang/MoDA/) and [models](https://huggingface.co/lixinyizju/moda/).

## ⚙️ Installation

**Create environment:**

```bash
# 1. Create base environment
conda create -n moda python=3.10 -y
conda activate moda 

# 2. Install requirements
pip install -r requirements.txt

# 3. Install ffmpeg
sudo apt-get update  
sudo apt-get install ffmpeg -y
```
## &#x1F680; Inference

### Local Inference
```python
python src/models/inference/moda_test.py  --image_path src/examples/reference_images/6.jpg  --audio_path src/examples/driving_audios/5.wav 
```

### Gradio Web Interface
```bash
python app.py
```

## 🐳 Docker Deployment (RunPod)

### Building Docker Image

**Prerequisites:**
- Docker Desktop installed and running
- HuggingFace account with access token
- Docker Hub account

**Build Steps:**

1. **Set environment variables:**
```bash
export DOCKER_USER="krapiunitski12"
export HUGGINGFACE_USERNAME="krapiunitski"
export HF_TOKEN="hf_xxxxxxxxxxxxx"  # Your HuggingFace token
```

2. **Build the image:**
```bash
cd MoDA
./scripts/build_docker.sh
```

Or build with models pre-downloaded (recommended for production):
```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg HUGGINGFACE_USERNAME=krapiunitski \
  --build-arg HF_TOKEN=hf_xxxxxxxxxxxxx \
  -t krapiunitski12/moda-runpod:latest \
  .
```

3. **Push to Docker Hub:**
```bash
./scripts/quick_push.sh
```

Or manually:
```bash
docker push krapiunitski12/moda-runpod:latest
```

### RunPod Deployment

1. **Create RunPod Endpoint:**
   - Go to [RunPod Serverless](https://www.runpod.io/serverless)
   - Create new endpoint
   - Docker image: `krapiunitski12/moda-runpod:latest`
   - GPU: RTX 4090 or A100 recommended
   - Disk space: 100GB+

2. **Set Environment Variables:**
   ```
   HUGGINGFACE_USERNAME=krapiunitski
   ```
   
   (Optional, only if models not pre-built in image):
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

3. **Test the endpoint:**
```python
import runpod
import base64

# Initialize RunPod
runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

# Load image and audio
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Run inference
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
video_data = base64.b64decode(result["video"])
with open("output.mp4", "wb") as f:
    f.write(video_data)
```

### CUDA Diagnostics

Check if GPU and CUDA are working correctly:
```python
result = endpoint.run_sync({
    "input": {
        "action": "check_cuda"
    }
})
print(result["cuda_info"])
```

### Model Upload to HuggingFace

To upload MoDA models to your HuggingFace account:

1. **Download original models:**
```bash
# The models will be downloaded during first inference run
python src/models/inference/moda_test.py --image_path src/examples/reference_images/6.jpg --audio_path src/examples/driving_audios/5.wav
```

2. **Upload to HuggingFace:**
```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Upload pretrain weights
huggingface-cli upload krapiunitski/moda-pretrain-weights ./pretrain_weights --repo-type model
```

3. **Verify upload:**
   - Check: https://huggingface.co/krapiunitski/moda-pretrain-weights

### Docker Scripts

Available scripts in `scripts/` directory:

- `build_docker.sh` - Interactive build with push option
- `quick_push.sh` - Quick test and push to Docker Hub
- `deploy.sh` - Interactive deployment menu
- `auto_build_and_push.sh` - Automated CI/CD build and push
- `build.sh` - Simple build without options
- `clean_docker_cache.sh` - Clean Docker build cache

### Configuration

Edit `runpod_config.yaml` to customize inference parameters:
- `output_fps` - Output video frame rate (default: 25)
- `batch_size` - Batch size for processing (default: 100)
- `device_id` - GPU device ID (default: 0)

### Image Size

- Base image: ~8-10 GB
- With pre-downloaded models: ~15-20 GB
- Cold start time (without models): ~2-5 minutes
- Cold start time (with models): ~30 seconds

### Performance

- GPU: RTX 4090 recommended
- Memory: 24GB+ VRAM
- Processing time: ~10-30 seconds per video (depends on audio length)
- Concurrent workers: 1 per GPU (recommended)
## ⚖️ Disclaimer
This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## 🙏🏻 Acknowledgements

We would like to thank the contributors to the [LivePortrait](https://github.com/KwaiVGI/LivePortrait), and [echomimic](https://github.com/antgroup/echomimic),[JoyVasa](https://github.com/jdh-algo/JoyVASA/),[Ditto](https://github.com/antgroup/ditto-talkinghead/), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [InsightFace](https://github.com/deepinsight/insightface), [X-Pose](https://github.com/IDEA-Research/X-Pose), [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk), [Hallo](https://github.com/fudan-generative-vision/hallo), [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain), [Q-Align](https://github.com/Q-Future/Q-Align), [Syncnet](https://github.com/joonson/syncnet_python), and [VBench](https://github.com/Vchitect/VBench) repositories, for their open research and extraordinary work.
If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.
## 📑 Citation

If you use MoDA in your research, please cite:

```bibtex
@article{li2025moda,
  title={MoDA: Multi-modal Diffusion Architecture for Talking Head Generation},
  author={Li, Xinyang and Li, Gen and Lin, Zhihui and Qian, Yichen and Yao, GongXin and Jia, Weinan and Chen, Weihua and Wang, Fan},
  journal={arXiv preprint arXiv:2507.03256},
  year={2025}
}
```
