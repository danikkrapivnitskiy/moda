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
    <a href='https://huggingface.co/lixinyizju/moda/'><img src='https://img.shields.io/badge/Models-Official-yellow'></a>
    <a href='https://huggingface.co/krapiunitski/moda-pretrain-weights'><img src='https://img.shields.io/badge/Weights-Fork-orange'></a>
</div>

**Generate a lip-synced talking-head video from a portrait image and driving audio — with optional emotion control and a RunPod serverless handler for GPU deployment.**

This fork adds a RunPod serverless handler, Docker build pipeline, and an optional emotion adapter. Local inference, Gradio (`app.py`), and YAML-driven configuration are included.

## Updates

- [2025.08.08] Original inference code and [pretrained weights](https://huggingface.co/lixinyizju/moda/) (Li et al.).
- Fork: RunPod deployment, emotion integration, pre-built Docker images — see `CHANGELOG.md`.

## Model weights

| Repository | Contents |
|------------|----------|
| [lixinyizju/moda](https://huggingface.co/lixinyizju/moda/) | Official MoDA checkpoints (paper authors) |
| [krapiunitski/moda-pretrain-weights](https://huggingface.co/krapiunitski/moda-pretrain-weights) | Pretrain weights mirror for RunPod Docker builds |
| [krapiunitski/moda-emotion-model](https://huggingface.co/krapiunitski/moda-emotion-model) | Lightweight `emo_model.pth` (~286 MB) for emotion adapter |

Set `HUGGINGFACE_USERNAME` to your namespace when building images that download weights at build time (see `.env.example`).

## Examples

### Demo output

Sample talking-head clip (512×512, English, generated with this pipeline):

<video src="https://github.com/danikkrapivnitskiy/moda/releases/download/readme-demo/welcome-video-en-female.mp4" controls width="512" playsinline></video>

### Local inference (bundled samples)

```bash
conda create -n moda python=3.10 -y && conda activate moda
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install ffmpeg -y   # Linux

python src/models/inference/moda_test.py \
  --image_path src/examples/reference_images/6.jpg \
  --audio_path src/examples/driving_audios/5.wav
```

### RunPod request (image + audio as base64)

```python
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "audio": audio_b64,
        "cfg_scale": 1.0,
        "emo": 8,       # 0–7 emotion codes, 8 = neutral
        "smooth": False,
    }
})
```

### Emotion codes

| Code | Emotion |
|------|---------|
| 0 | Anger |
| 1 | Contempt |
| 2 | Disgust |
| 3 | Fear |
| 4 | Happiness |
| 5 | Neutral |
| 6 | Sadness |
| 7 | Surprise |
| 8 | None / default |

See `emo_map` in `src/models/inference/moda_test.py`.

## Try it (Gradio)

```bash
python app.py
```

Weights download automatically on first run from [lixinyizju/moda](https://huggingface.co/lixinyizju/moda/) unless pre-baked in a Docker image.

## How it works

```
Reference image + driving audio
        │
        ▼
  AudioProcessor / MotionProcessor
        │
        ▼
  MotionDiffusion (DiT) ──► motion latents
        │
        ▼
  LivePortrait pipeline ──► talking-head video (.mp4)
        │
        └─ (optional) emotion adapter ──► emotion-conditioned motion
```

Key paths:

| Component | Location |
|-----------|----------|
| Inference pipeline | `src/models/inference/moda_test.py` |
| Diffusion model | `src/models/dit/` |
| Emotion adapter | `src/models/emotion/` |
| Configs | `configs/audio2motion/` |
| RunPod handler | `runpod_server.py` |
| RunPod tuning | `runpod_config.yaml` |

Emotion codes (`emo` parameter): see [Examples](#examples) and `emo_map` in `moda_test.py`.

## RunPod deployment

**Prerequisites:** Docker, a HuggingFace account/token, Docker Hub account (for pushing images).

1. **Configure secrets locally** (never commit `.env`):

```bash
cp .env.example .env
# edit: HUGGINGFACE_USERNAME, HF_TOKEN, DOCKER_USER, DOCKER_TOKEN
```

2. **Build and push:**

```bash
export DOCKER_USER="your-dockerhub-username"
export HUGGINGFACE_USERNAME="your-huggingface-username"
export HF_TOKEN="hf_your_token_here"

./scripts/build_docker.sh
# or: docker buildx build --platform linux/amd64 \
#   --build-arg HUGGINGFACE_USERNAME=$HUGGINGFACE_USERNAME \
#   --build-arg HF_TOKEN=$HF_TOKEN \
#   -t $DOCKER_USER/moda-runpod:latest .
```

3. **Create a RunPod Serverless endpoint** with image `$DOCKER_USER/moda-runpod:latest`, GPU RTX 4090 / A100 (24 GB+ VRAM), 100 GB+ disk.

4. **Environment on RunPod** (if models are baked into the image, `HF_TOKEN` is optional at runtime):

```
HUGGINGFACE_USERNAME=your-huggingface-username
```

5. **Call the endpoint:**

```python
import runpod, base64

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "audio": audio_b64,
        "cfg_scale": 1.0,
        "emo": 8,
        "smooth": False,
    }
})

video_data = base64.b64decode(result["video"])
open("output.mp4", "wb").write(video_data)
```

CUDA diagnostics:

```python
endpoint.run_sync({"input": {"action": "check_cuda"}})
```

Container path checks (uses env vars — see `.env.example`):

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python test_check_paths.py
```

### Docker scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_docker.sh` | Interactive build with optional push |
| `scripts/quick_push.sh` | Test image and push |
| `scripts/auto_build_and_push.sh` | CI-style build + push |
| `scripts/download_emotion_model.sh` | Fetch `emo_model.pth` for local dev |
| `scripts/clean_docker_cache.sh` | Prune build cache |

Tune inference defaults in `runpod_config.yaml` (`output_fps`, `batch_size`, emotion block).

### Image size & performance

| Metric | Typical value |
|--------|---------------|
| Base image | ~8–10 GB |
| With pre-downloaded weights | ~15–20 GB |
| Cold start (weights in image) | ~30 s |
| Cold start (download at runtime) | ~2–5 min |
| Inference | ~10–30 s per clip (audio-length dependent) |
| Recommended GPU | RTX 4090 / A6000 / A40, 24 GB+ VRAM |
| Workers per GPU | 1 (multiple workers risk OOM) |

## Emotion model (optional)

Enhanced expressions use `emo_model.pth` (~286 MB). Setup:

```bash
./scripts/download_emotion_model.sh
# or upload your own copy to HuggingFace: scripts/upload_emotion_model.sh
```

See `docs/emotion-model-setup.md` and `examples/emo/` for `.npy` emotion priors.

## Configuration

Runtime behavior is driven by YAML under `configs/audio2motion/` and `runpod_config.yaml`. The RunPod handler merges `runpod_config.yaml` over nested OmegaConf configs at startup.

Environment variables used across scripts and `runpod_server.py`:

| Variable | Purpose |
|----------|---------|
| `HUGGINGFACE_USERNAME` | HF repo namespace for weight fallback download |
| `HF_TOKEN` | HuggingFace token (build time or Tier-3 fallback) |
| `DOCKER_USER` / `DOCKER_TOKEN` | Docker Hub push scripts |
| `RUNPOD_API_KEY` / `RUNPOD_ENDPOINT_ID` | `test_check_paths.py` diagnostics |

## Project layout

```
MoDA/
├── src/models/           # DiT, inference, emotion adapter
├── src/datasets/         # audio/motion preprocessing
├── configs/audio2motion/ # model + inference YAML
├── runpod_server.py      # RunPod serverless entry
├── runpod_config.yaml    # production tuning
├── scripts/              # Docker and model utilities
├── docs/                 # deployment and optimization guides
└── examples/             # sample images, audio, emotion .npy
```

## Additional docs

- `docs/deployment-guide.md` — step-by-step RunPod setup
- `docs/load-balancing-guide.md` — scaling multiple endpoints
- `docs/performance-optimization.md` — GPU and batch tuning
- `CHANGELOG.md` — fork-specific changes (emotion integration, RunPod handler)

## Disclaimer

This project is intended for academic research. Users are solely responsible for generated content.

## Citation

```bibtex
@article{li2025moda,
  title={MoDA: Multi-modal Diffusion Architecture for Talking Head Generation},
  author={Li, Xinyang and Li, Gen and Lin, Zhihui and Qian, Yichen and Yao, GongXin and Jia, Weinan and Chen, Weihua and Wang, Fan},
  journal={arXiv preprint arXiv:2507.03256},
  year={2025}
}
```

## License

Follow the license terms of the upstream MoDA release and bundled third-party components.
