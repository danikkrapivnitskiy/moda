#!/usr/bin/env python3
"""
RunPod Serverless handler for MoDA
Generates multi-modal talking head videos

IMPORTANT RunPod Configuration:
- GPU: A6000/A40 48GB @ $0.00034/sec ($1.22/hr)
- Max Workers: Set to 1 (each video generation uses significant GPU memory)
  Multiple workers may cause CUDA Out of Memory errors
- Batch size: 50 (optimized for A6000/A40 48GB memory)
- For production scaling, use separate GPU instances (not multiple workers per GPU)
"""

# ============================================================================
# Critical: Set cache paths BEFORE importing any HuggingFace libraries
# This ensures models are cached to persistent storage (/workspace)
# Without this, models re-download on every request (overhead)
# ============================================================================
import os
import sys
import signal
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] Loaded environment variables from .env file")
    else:
        print(f"[INFO] .env file not found, using system environment variables")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Configure HuggingFace cache to use persistent storage
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['XDG_CACHE_HOME'] = '/workspace/.cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

# CPU thread limits - prevents 100% CPU usage during GPU operations
# Override in RunPod if you need different values (e.g., ENV vars)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('TORCH_NUM_THREADS', '2')

# PyTorch CUDA memory allocator configuration will be set from runpod_config.yaml
# via ConfigManager after primary config is loaded

# Create cache directory if it doesn't exist
os.makedirs('/workspace/.cache/huggingface', exist_ok=True)

# Signal handlers for graceful shutdown
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

# Now safe to import other libraries
import runpod
import base64
import tempfile
import subprocess
import json
import time
import fcntl
import shutil
from omegaconf import OmegaConf

# Lazy initialization to avoid import conflicts
pipe = None
runpod_config = None

def convert_audio_to_wav(audio_path):
    """
    Converts audio file to WAV format if necessary.
    Also resamples WAV files to 16kHz mono if needed (e.g., OpenAI TTS outputs 24kHz).
    Supports: MP3, M4A, AAC, OGG, FLAC, WAV via ffmpeg (installed in Docker)
    Uses direct ffmpeg call for better performance and correct audio parameters.
    
    Args:
        audio_path: Path to input audio file
    
    Returns:
        Path to WAV file in optimal format (16kHz, mono, 16-bit PCM)
    
    Raises:
        RuntimeError: If conversion fails or format is unsupported
    """
    from pathlib import Path
    import subprocess
    
    # Check if already WAV - but verify it's in optimal format (16kHz mono)
    if audio_path.lower().endswith('.wav'):
        # Check WAV parameters using ffprobe
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate,channels',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            if len(lines) >= 2:
                sample_rate = int(lines[0])
                channels = int(lines[1])
                
                print(f"[INFO] WAV format detected: {sample_rate}Hz, {channels} channel(s)")
                
                # Check if already optimal (16kHz mono)
                if sample_rate == 16000 and channels == 1:
                    print(f"[INFO] Audio is already optimal format (16kHz mono) - no conversion needed")
                    return audio_path
                
                # Need to resample/convert
                print(f"[INFO] WAV needs resampling to 16kHz mono (from {sample_rate}Hz, {channels}ch)")
        
        except Exception as e:
            print(f"[WARN] Could not check WAV parameters: {e}")
            print(f"[INFO] Will convert to ensure optimal format")
    
    # Detect format from file
    file_ext = Path(audio_path).suffix.lower()
    supported_formats = ['.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wav', '.dat']
    
    if file_ext not in supported_formats:
        raise RuntimeError(
            f"Unsupported audio format: {file_ext}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    # Generate output path
    if audio_path.lower().endswith('.wav'):
        # If already .wav, create new path to avoid overwriting original
        wav_path = str(Path(audio_path).parent / f"{Path(audio_path).stem}_16k.wav")
    else:
        wav_path = str(Path(audio_path).with_suffix('.wav'))
    
    try:
        # Convert/resample to optimal format: 16kHz mono PCM WAV
        # -y: overwrite output file
        # -i: input file
        # -acodec pcm_s16le: 16-bit PCM WAV format
        # -ar 16000: sample rate (optimized for speech processing, MoDA requirement)
        # -ac 1: mono audio (required by MoDA)
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', audio_path,
            '-acodec', 'pcm_s16le',  # WAV codec
            '-ar', '16000',  # 16kHz sample rate for speech
            '-ac', '1',  # Mono channel
            wav_path
        ]
        
        print(f"[INFO] Converting to optimal WAV format (16kHz, mono)...")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        if os.path.exists(wav_path):
            print(f"[OK] Audio converted successfully")
            return wav_path
        else:
            raise RuntimeError("Conversion completed but output file not found")
            
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg conversion failed: {e.stderr}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Audio conversion error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)

def validate_input_sizes(image_base64, audio_base64):
    """Validate input file sizes to prevent memory issues"""
    # Approximate size in MB (base64 is ~33% larger than binary)
    image_size_mb = len(image_base64) * 0.75 / (1024 * 1024)
    audio_size_mb = len(audio_base64) * 0.75 / (1024 * 1024)
    
    # Load limits from config (with defaults)
    try:
        config = load_runpod_config()
        MAX_IMAGE_SIZE_MB = config.get('resource_limits', {}).get('max_image_size_mb', 10)
        MAX_AUDIO_SIZE_MB = config.get('resource_limits', {}).get('max_audio_size_mb', 50)
    except Exception:
        # Fallback to defaults if config not available
        MAX_IMAGE_SIZE_MB = 10
        MAX_AUDIO_SIZE_MB = 50
    
    if image_size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(
            f"Image too large: {image_size_mb:.1f} MB (max: {MAX_IMAGE_SIZE_MB} MB)"
        )
    
    if audio_size_mb > MAX_AUDIO_SIZE_MB:
        raise ValueError(
            f"Audio too large: {audio_size_mb:.1f} MB (max: {MAX_AUDIO_SIZE_MB} MB)"
        )
    
    print(f"[INFO] Input sizes validated: image={image_size_mb:.1f}MB, audio={audio_size_mb:.1f}MB")
    return True

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

def create_error_response(error_type, message, details=None):
    """Create structured error response for consistent error handling"""
    response = {
        "error": message,
        "error_type": error_type,
        "status": "failed"
    }
    if details:
        response["details"] = details
    return response

def safe_base64_decode(data, field_name):
    """Safely decode base64 with validation"""
    try:
        if not isinstance(data, str):
            raise ValueError(f"{field_name} must be a string")
        
        # Decode with validation
        decoded = base64.b64decode(data, validate=True)
        
        if len(decoded) == 0:
            raise ValueError(f"{field_name} is empty after decoding")
        
        return decoded
    except Exception as e:
        raise ValueError(f"Invalid base64 for {field_name}: {str(e)}")

def process_emotion_input(emotion_input, temp_dir):
    """
    Process emotion input from RunPod request.
    
    Supports two input modes:
    1. Integer code (0-8): Basic emotions
       - 0: Anger, 1: Contempt, 2: Disgust, 3: Fear
       - 4: Happiness, 5: Neutral, 6: Sadness, 7: Surprise
       - 8: None (neutral, default)
    2. Base64 encoded .npy file: Enhanced DICE-Talk emotion features
       - Upload .npy file from DICE-Talk examples/emo/ directory
       - Provides 64-code emotion control with attention-based retrieval
    
    Args:
        emotion_input: int, str (base64 .npy), or None
        temp_dir: temporary directory for files
    
    Returns:
        Processed emotion (int or str path to .npy)
    """
    if emotion_input is None:
        return 8  # Default neutral
    
    if isinstance(emotion_input, int):
        # Validate emotion code
        if 0 <= emotion_input <= 8:
            return emotion_input
        else:
            print(f"[WARN] Invalid emotion code {emotion_input}, using neutral (8)")
            return 8
    
    if isinstance(emotion_input, str):
        # Check if it's base64 encoded .npy file
        try:
            # Try to decode base64
            emotion_data = base64.b64decode(emotion_input)
            
            # Verify it's a valid .npy file (starts with magic bytes)
            if emotion_data[:6] == b'\x93NUMPY':
                # Save to temp file
                emotion_path = os.path.join(temp_dir, "emotion_features.npy")
                with open(emotion_path, 'wb') as f:
                    f.write(emotion_data)
                print(f"[INFO] Saved emotion features to {emotion_path}")
                return emotion_path
            else:
                # Not a .npy file, might be emotion name string
                emotion_name = emotion_input.lower()
                emotion_code_map = {
                    'anger': 0, 'angry': 0,
                    'contempt': 1,
                    'disgust': 2, 'disgusted': 2,
                    'fear': 3,
                    'happiness': 4, 'happy': 4,
                    'neutral': 5,
                    'sadness': 6, 'sad': 6,
                    'surprise': 7, 'surprised': 7,
                    'none': 8
                }
                if emotion_name in emotion_code_map:
                    return emotion_code_map[emotion_name]
                else:
                    print(f"[WARN] Unknown emotion string: {emotion_input}, using neutral")
                    return 8
        except Exception as e:
            # Not base64, might be file path or emotion name
            if emotion_input.endswith('.npy'):
                # File path
                if os.path.exists(emotion_input):
                    return emotion_input
                else:
                    print(f"[WARN] Emotion file not found: {emotion_input}, using neutral")
                    return 8
            else:
                # Try as emotion name
                emotion_name = emotion_input.lower()
                emotion_code_map = {
                    'anger': 0, 'angry': 0,
                    'contempt': 1,
                    'disgust': 2, 'disgusted': 2,
                    'fear': 3,
                    'happiness': 4, 'happy': 4,
                    'neutral': 5,
                    'sadness': 6, 'sad': 6,
                    'surprise': 7, 'surprised': 7,
                    'none': 8
                }
                if emotion_name in emotion_code_map:
                    return emotion_code_map[emotion_name]
                else:
                    print(f"[WARN] Failed to process emotion input: {e}, using neutral")
                    return 8
    
    return 8

def load_runpod_config():
    """Load RunPod configuration from YAML file"""
    global runpod_config
    if runpod_config is None:
        config_path = Path(__file__).parent / "runpod_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"RunPod config file not found: {config_path}")
        runpod_config = OmegaConf.load(config_path)
    return runpod_config

def ensure_models_downloaded():
    """Ensure required models are available (use pre-built or download)
    
    Priority:
    1. Check persistent storage (/workspace/pretrain_weights) - fastest if cached
    2. Copy from pre-built image cache (/app/models_cache) - fast cold start
    3. Download from HuggingFace - slowest fallback
    """
    import os
    from huggingface_hub import snapshot_download
    import subprocess
    import shutil
    
    # Use persistent disk in RunPod (/workspace) to avoid re-downloading on worker restart
    # Also create symlink so code using BASE_DIR still works
    persistent_pretrain = "/workspace/pretrain_weights"
    app_pretrain = "/app/pretrain_weights"
    prebuilt_cache = "/app/models_cache"  # Pre-built models from Docker image
    
    # Create persistent directory
    os.makedirs(persistent_pretrain, exist_ok=True)
    
    # Setup symlink for /app/pretrain_weights → /workspace/pretrain_weights
    if os.path.exists(app_pretrain):
        if os.path.islink(app_pretrain):
            real_path = os.readlink(app_pretrain)
            if real_path != persistent_pretrain:
                os.unlink(app_pretrain)
                os.symlink(persistent_pretrain, app_pretrain)
                print(f"[INFO] Updated symlink {app_pretrain} -> {persistent_pretrain}")
            else:
                print(f"[INFO] Symlink {app_pretrain} -> {persistent_pretrain} already exists")
        elif os.path.isdir(app_pretrain):
            # If it's an empty directory (from Dockerfile mkdir), remove it and create symlink
            try:
                if not os.listdir(app_pretrain):  # Check if empty
                    os.rmdir(app_pretrain)
                    os.symlink(persistent_pretrain, app_pretrain)
                    print(f"[INFO] Removed empty directory and created symlink {app_pretrain} -> {persistent_pretrain}")
                else:
                    # If directory has content (from Docker COPY), keep it
                    print(f"[INFO] {app_pretrain} exists as non-empty directory, skipping symlink")
            except Exception as e:
                print(f"[ERROR] Failed to setup symlink: {e}")
        else:
            print(f"[WARN] {app_pretrain} exists but is not a directory or symlink")
    else:
        try:
            os.symlink(persistent_pretrain, app_pretrain)
            print(f"[INFO] Created symlink {app_pretrain} -> {persistent_pretrain}")
        except OSError as e:
            print(f"[ERROR] Failed to create symlink: {e}")
            # Fallback: use persistent_pretrain directly
    
    pretrain_dir = persistent_pretrain
    
    # Helper function to ensure model availability with fallback logic
    def ensure_model(model_name, repo_id, target_path, required_subdirs=None):
        """
        Ensure model is available using 3-tier fallback with file locking for multi-worker safety:
        1. Check persistent storage (fastest)
        2. Copy from pre-built cache (fast cold start)
        3. Download from HuggingFace (slow fallback)
        """
        lock_file = f"/tmp/{model_name}.lock"
        prebuilt_path = os.path.join(prebuilt_cache, model_name)
        
        # Try to acquire lock (wait max 300 seconds)
        lock_fd = None
        try:
            lock_fd = open(lock_file, 'w')
            print(f"[INFO] Acquiring lock for {model_name}...")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            print(f"[INFO] Lock acquired for {model_name}")
            
            # Tier 1: Check if model exists in persistent storage
            if os.path.exists(target_path):
                if required_subdirs:
                    # Check specific subdirectories
                    all_exist = all(os.path.exists(os.path.join(target_path, d)) for d in required_subdirs)
                    if all_exist:
                        print(f"[OK] {model_name} found in persistent storage")
                        return
                elif os.listdir(target_path):  # Directory exists and not empty
                    print(f"[OK] {model_name} found in persistent storage")
                    return
            
            # Tier 2: Check if pre-built model exists in image cache
            if os.path.exists(prebuilt_path) and os.listdir(prebuilt_path):
                print(f"[INFO] Copying {model_name} from pre-built image cache...")
                print(f"       This is much faster than downloading from HuggingFace")
                try:
                    # Copy the pre-built models
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copytree(prebuilt_path, target_path, dirs_exist_ok=True)
                    
                    print(f"[OK] {model_name} copied from pre-built cache (~30 seconds)")
                    
                    # Verify required subdirectories if specified
                    if required_subdirs:
                        missing = [d for d in required_subdirs if not os.path.exists(os.path.join(target_path, d))]
                        if missing:
                            print(f"[WARN] Some subdirectories missing after copy: {missing}")
                            raise FileNotFoundError(f"Missing subdirectories: {missing}")
                    
                    return
                except Exception as e:
                    print(f"[WARN] Failed to copy from pre-built cache: {e}")
                    print(f"       Will try downloading from HuggingFace...")
            else:
                print(f"[INFO] Pre-built cache for {model_name} not found at {prebuilt_path}")
                print(f"       This is normal if not using pre-built Docker image")
            
            # Tier 3: Download from HuggingFace (slowest fallback)
            # Only reached if models are not in persistent storage or pre-built cache
            print(f"[INFO] Downloading {model_name} from HuggingFace...")
            print(f"       This may take several minutes...")
            print(f"       Note: HF_TOKEN should be set for private repositories")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False,
                    token=os.getenv('HF_TOKEN')  # Explicitly pass token for private repos
                )
                print(f"[OK] {model_name} downloaded successfully")
                
                # Verify required subdirectories if specified
                if required_subdirs:
                    missing = [d for d in required_subdirs if not os.path.exists(os.path.join(target_path, d))]
                    if missing:
                        raise FileNotFoundError(f"Missing required subdirectories after download: {missing}")
            except Exception as e:
                print(f"[ERROR] Failed to download {model_name}: {e}")
                raise
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                    print(f"[INFO] Lock released for {model_name}")
                except Exception as e:
                    print(f"[WARN] Failed to release lock: {e}")
    
    # Check disk space before model download
    check_disk_space(path="/workspace", min_gb=15)
    
    # Ensure all required models are available
    print("[INFO] Checking required models...")
    
    # Load model repositories from environment variables - REQUIRED
    HUGGINGFACE_USERNAME = os.getenv('HUGGINGFACE_USERNAME')
    
    if not HUGGINGFACE_USERNAME:
        error_msg = (
            "HUGGINGFACE_USERNAME environment variable is not set!\n"
            "Please set it to your HuggingFace username:\n"
            "  export HUGGINGFACE_USERNAME='your-username'\n"
            "Or in RunPod: Add HUGGINGFACE_USERNAME to environment variables"
        )
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    # Build repository IDs - use only custom repositories
    MODA_PRETRAIN_REPO = f'{HUGGINGFACE_USERNAME}/moda-pretrain-weights'
    
    print(f"[INFO] Using HuggingFace repositories for user: {HUGGINGFACE_USERNAME}")
    print(f"[INFO] Model repository:")
    print(f"  - MoDA pretrain weights: {MODA_PRETRAIN_REPO}")
    
    # Check if HF_TOKEN is set (only needed for Tier 3 fallback download)
    # If models are pre-built in Docker image (Tier 2), token is NOT needed in RunPod
    # Token is only needed during Docker build (--build-arg HF_TOKEN) for pre-downloading models
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print("[INFO] HF_TOKEN found - will use if models need to be downloaded (Tier 3 fallback)")
        os.environ['HF_TOKEN'] = hf_token
    else:
        print("[INFO] No HF_TOKEN in RunPod - OK if models are pre-built in Docker image")
        print("       Models will be copied from /app/models_cache (Tier 2) or downloaded if needed")
    
    # Model: MoDA pretrain weights (~5-10 GB)
    # Includes LivePortrait models, audio processing models, and MoDA checkpoint
    required_subdirs = ["moda", "decode"]  # Expected subdirectories
    ensure_model(
        model_name="pretrain_weights",
        repo_id=MODA_PRETRAIN_REPO,
        target_path=pretrain_dir,
        required_subdirs=required_subdirs
    )
    
    # Emotion model: DICE-Talk emotion adapter (optional, ~286 MB)
    # Auto-download if not found in standard locations
    emotion_checkpoints_dir = "/workspace/checkpoints/DICE-Talk"
    emotion_model_path = os.path.join(emotion_checkpoints_dir, "emo_model.pth")
    emotion_repo = f'{HUGGINGFACE_USERNAME}/moda-emotion-model'
    
    if not os.path.exists(emotion_model_path):
        # Check pre-built cache first
        prebuilt_emotion_path = "/app/models_cache/checkpoints/DICE-Talk/emo_model.pth"
        if os.path.exists(prebuilt_emotion_path):
            print("[INFO] Copying emotion model from pre-built cache...")
            os.makedirs(emotion_checkpoints_dir, exist_ok=True)
            shutil.copy2(prebuilt_emotion_path, emotion_model_path)
            print("[OK] Emotion model copied from pre-built cache")
        else:
            # Try to download from HuggingFace (optional, non-fatal)
            try:
                from huggingface_hub import hf_hub_download
                print("[INFO] Downloading emotion model from HuggingFace (optional)...")
                print(f"       Repository: {emotion_repo} (~286 MB)")
                os.makedirs(emotion_checkpoints_dir, exist_ok=True)
                hf_hub_download(
                    repo_id=emotion_repo,
                    filename="emo_model.pth",  # Direct file in root, no subdirectory
                    local_dir=emotion_checkpoints_dir,
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                print("[OK] Emotion model downloaded")
            except Exception as e:
                print(f"[WARN] Failed to download emotion model: {e}")
                print("       Enhanced emotions will be disabled, using simple emotion codes")
                print(f"       You can manually download from: {emotion_repo}/emo_model.pth")
    else:
        print("[OK] Emotion model found in persistent storage")
    
    print("[OK] All models are ready")

def get_pipe():
    """Lazy initialization of MoDA LiveVASAPipeline with model auto-download"""
    global pipe
    if pipe is None:
        # Ensure models are downloaded first
        try:
            ensure_models_downloaded()
        except Exception as e:
            print(f"[ERROR] Model download failed: {e}")
            raise
        
        # Load primary configuration using ConfigManager
        import sys
        # Add src to path for imports (both /app/src for Docker and relative for local)
        src_paths = ['/app/src', str(Path(__file__).parent / 'src')]
        for src_path in src_paths:
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
        from utils.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        primary_config = config_manager.load_primary_config()
        
        # Apply memory optimization environment variables
        memory_env = config_manager.get_memory_optimization_env(primary_config)
        for env_key, env_value in memory_env.items():
            os.environ[env_key] = env_value
            print(f"[INFO] Set {env_key}={env_value} from primary config")
        
        # Get inference config path from primary config
        config = load_runpod_config()  # Keep for backward compatibility
        cfg_path = primary_config.get('inference_config_path', config.inference_config_path)
        
        print(f"[INFO] Initializing MoDA pipeline with config: {cfg_path}")
        print(f"[INFO] Using enterprise configuration manager for automatic config merging")
        
        # Merge primary config into inference config
        inference_cfg = config_manager.merge_configs(
            primary_config=primary_config,
            nested_config_path=cfg_path,
            section='inference'
        )
        
        # Allow device_id override from environment (backward compatibility)
        if os.getenv('CUDA_DEVICE_ID'):
            device_id_override = int(os.getenv('CUDA_DEVICE_ID'))
            inference_cfg.device_id = device_id_override
            print(f"[INFO] Device ID overridden from CUDA_DEVICE_ID env var: {device_id_override}")
        
        # Save merged config to temp file
        import tempfile
        from omegaconf import OmegaConf
        temp_cfg_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        OmegaConf.save(inference_cfg, temp_cfg_path.name)
        temp_cfg_path.close()
        merged_cfg_path = temp_cfg_path.name
        
        # Import MoDA pipeline
        from models.inference.moda_test import LiveVASAPipeline
        
        # Initialize pipeline with merged config and primary config for further merging
        pipe = LiveVASAPipeline(
            cfg_path=merged_cfg_path,
            load_motion_generator=True,
            motion_mean_std_path=primary_config.get('motion_mean_std_path', config.get('motion_mean_std_path', None)),
            primary_config=primary_config  # Pass primary config for motion processor merging
        )
        print("[OK] MoDA pipeline initialized with enterprise configuration system")
    return pipe

def check_cuda_info():
    """Comprehensive CUDA and GPU diagnostics"""
    info = {
        "cuda_available": False,
        "pytorch_cuda_version": None,
        "nvidia_smi": None,
        "gpu_count": 0,
        "gpu_details": [],
        "cudnn_available": False,
        "cudnn_version": None,
        "warnings": [],
        "errors": []
    }
    
    # PyTorch CUDA info
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["pytorch_cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["cudnn_available"] = torch.backends.cudnn.is_available()
            
            if info["cudnn_available"]:
                info["cudnn_version"] = torch.backends.cudnn.version()
            
            # Get GPU details
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory": f"{props.total_memory / (1024**3):.2f} GB",
                    "multiprocessor_count": props.multi_processor_count
                }
                
                # Check current memory usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_info["memory_allocated"] = f"{allocated:.2f} GB"
                gpu_info["memory_reserved"] = f"{reserved:.2f} GB"
                
                # Warnings
                if props.major < 7:
                    info["warnings"].append(f"GPU {i} has compute capability {props.major}.{props.minor} - may not support all features")
                
                info["gpu_details"].append(gpu_info)
        else:
            info["errors"].append("CUDA not available in PyTorch")
            
    except ImportError:
        info["errors"].append("PyTorch not installed")
    except Exception as e:
        info["errors"].append(f"PyTorch CUDA check failed: {e}")
    
    # nvidia-smi info (system CUDA/runtime)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,cuda_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info["nvidia_smi"] = result.stdout.strip()
        else:
            info["warnings"].append("nvidia-smi not available or failed")
    except FileNotFoundError:
        info["warnings"].append("nvidia-smi command not found")
    except Exception as e:
        info["warnings"].append(f"nvidia-smi check failed: {e}")
    
    # Check torchvision::nms operator
    # Import torch first to avoid circular import issues
    try:
        import torch
        # Wait a bit to ensure torchvision is fully initialized
        import sys
        if 'torchvision' in sys.modules:
            # Clear cached import if exists
            del sys.modules['torchvision']
        
        # Import torchvision after torch is loaded
        from torchvision import ops
        
        # Check if nms is available
        if hasattr(ops, 'nms') and callable(ops.nms):
            info["torchvision_nms"] = "available"
            
            # Try actual usage
            try:
                boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
                scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
                result = ops.nms(boxes, scores, iou_threshold=0.5)
                info["torchvision_nms"] = "available and working"
            except Exception as e:
                error_msg = str(e)
                if 'operator torchvision::nms does not exist' in error_msg:
                    info["torchvision_nms"] = "operator does not exist"
                    info["errors"].append("torchvision::nms operator does not exist - CUDA compatibility issue!")
                else:
                    info["torchvision_nms"] = f"available but test failed: {error_msg}"
                    info["warnings"].append(f"torchvision::nms test failed: {e}")
        else:
            info["torchvision_nms"] = "not callable"
            info["errors"].append("torchvision::nms operator not callable")
    except ImportError as e:
        info["torchvision_nms"] = "import failed"
        info["errors"].append(f"torchvision import failed: {e}")
    except Exception as e:
        error_msg = str(e)
        if 'operator torchvision::nms does not exist' in error_msg:
            info["torchvision_nms"] = "operator does not exist"
            info["errors"].append("torchvision::nms operator does not exist - CUDA compatibility issue!")
        elif 'circular import' in error_msg.lower() or 'partially initialized' in error_msg.lower():
            # This might be a runtime issue, try lazy import
            info["torchvision_nms"] = "import issue (may work at runtime)"
            info["warnings"].append(f"torchvision::nms import issue: {error_msg[:100]}")
        else:
            info["torchvision_nms"] = f"error: {error_msg[:100]}"
            info["warnings"].append(f"torchvision::nms check: {e}")
    
    return info

def handler(event):
    """
    RunPod handler
    Input: {
        "image": "base64_encoded_image",
        "audio": "base64_encoded_audio",
        "output_fps": 25 (optional, default: 25)
        "batch_size": 100 (optional, default: 100)
        "action": "check_cuda" (optional, for CUDA diagnostics)
    }
    """
    # Special diagnostic endpoint
    input_data = event.get("input", {})
    
    # Debug: log what we received
    print(f"[DEBUG] Received event input keys: {list(input_data.keys())}")
    print(f"[DEBUG] Action value: {input_data.get('action')}")
    
    if input_data.get("action") == "check_cuda":
        cuda_info = check_cuda_info()
        
        # Pretty print for logs
        print("=" * 60)
        print("CUDA DIAGNOSTICS")
        print("=" * 60)
        print(f"PyTorch version: {cuda_info.get('pytorch_version', 'N/A')}")
        print(f"CUDA available: {cuda_info['cuda_available']}")
        if cuda_info['cuda_available']:
            print(f"PyTorch CUDA version: {cuda_info['pytorch_cuda_version']}")
            print(f"cuDNN available: {cuda_info['cudnn_available']}")
            if cuda_info['cudnn_available']:
                print(f"cuDNN version: {cuda_info['cudnn_version']}")
            print(f"GPU count: {cuda_info['gpu_count']}")
            for gpu in cuda_info['gpu_details']:
                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(f"    Compute: {gpu['compute_capability']}")
                print(f"    Memory: {gpu['total_memory']} (allocated: {gpu['memory_allocated']}, reserved: {gpu['memory_reserved']})")
        print(f"torchvision::nms: {cuda_info.get('torchvision_nms', 'N/A')}")
        if cuda_info.get('nvidia_smi'):
            print(f"\nnvidia-smi output:\n{cuda_info['nvidia_smi']}")
        if cuda_info.get('warnings'):
            print(f"\nWarnings: {len(cuda_info['warnings'])}")
            for warn in cuda_info['warnings']:
                print(f"  - {warn}")
        if cuda_info.get('errors'):
            print(f"\nErrors: {len(cuda_info['errors'])}")
            for err in cuda_info['errors']:
                print(f"  - {err}")
        print("=" * 60)
        
        return {
            "cuda_info": cuda_info,
            "status": "success" if not cuda_info.get('errors') else "warning"
        }
    
    # Health check endpoint
    if input_data.get("action") == "health":
        return {
            "status": "healthy",
            "models_loaded": pipe is not None,
            "timestamp": time.time()
        }
    
    if input_data.get("action") == "check_paths":
        print("[INFO] Path diagnostics requested")
        paths_info = {"status": "ok", "paths": {}}
        
        check_paths = [
            "/app/pretrain_weights",
            "/workspace/pretrain_weights",
            "/workspace/pretrain_weights/audio",
            "/workspace/pretrain_weights/audio/chinese-hubert-base",
            "/app/models_cache/pretrain_weights",
            "/app/models_cache/pretrain_weights/audio",
            "/app/pretrain_weights/audio/chinese-hubert-base"
        ]
        
        for path in check_paths:
            exists = os.path.exists(path)
            is_link = os.path.islink(path)
            info = {"exists": exists, "is_link": is_link}
            
            if is_link:
                try:
                    info["link_target"] = os.readlink(path)
                except:
                    pass
            
            if exists and os.path.isdir(path):
                try:
                    items = os.listdir(path)
                    info["contents"] = items[:10]
                    info["total_items"] = len(items)
                except Exception as e:
                    info["error"] = str(e)
            
            paths_info["paths"][path] = info
        
        return paths_info
    
    # Normal video generation endpoint
    try:
        timing = {}
        start_total = time.time()
        
        input_data = event.get("input", {})
        
        # Validate required fields
        if "image" not in input_data or "audio" not in input_data:
            return create_error_response(
                "ValidationError",
                "Missing required fields: 'image' and 'audio' are required for video generation",
                {"required": ["image", "audio"], "provided": list(input_data.keys())}
            )
        
        image_base64 = input_data["image"]
        audio_base64 = input_data["audio"]
        
        # Check disk space before video generation
        check_disk_space(path="/tmp", min_gb=1)
        
        # Validate input sizes
        start = time.time()
        validate_input_sizes(image_base64, audio_base64)
        timing['validation'] = time.time() - start
        
        # Note: Video encoding parameters (fps, crf, codec, preset) are configured in
        # liveportrait_config.yaml and loaded by the pipeline during initialization.
        # These settings are applied automatically during video generation.
        
        # Decode base64
        start = time.time()
        image_data = safe_base64_decode(image_base64, "image")
        audio_data = safe_base64_decode(audio_base64, "audio")
        timing['decode'] = time.time() - start
        
        # Create temporary files
        img_path = None
        aud_path_temp = None
        aud_path = None
        output_path = None
        
        try:
            img_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
            aud_path_temp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
            
            # Write input files
            with open(img_path, 'wb') as f:
                f.write(image_data)
            with open(aud_path_temp, 'wb') as f:
                f.write(audio_data)
            
            # Convert audio to WAV if needed (supports MP3, M4A, AAC, OGG, FLAC)
            start = time.time()
            print(f"[INFO] Processing audio file...")
            aud_path = convert_audio_to_wav(aud_path_temp)
            timing['audio_conversion'] = time.time() - start
            
            # Clean up temp file only if conversion created new file
            if aud_path and aud_path != aud_path_temp:
                try:
                    os.unlink(aud_path_temp)
                    aud_path_temp = None  # Mark as cleaned
                    print(f"[INFO] Cleaned up temporary audio file")
                except Exception as e:
                    print(f"[WARN] Failed to cleanup temp audio: {e}")
            
            print(f"🚀 Generating video (Telegram optimized)")
            print(f"   Image: {img_path}")
            print(f"   Audio: {aud_path}")
            print(f"   Note: Video quality (fps, crf, codec, preset) is configured in liveportrait_config.yaml")
            print(f"         Current settings: fps=25, crf=25, codec=libx264, preset=faster")
            
            # Get pipe (lazy init on first request)
            pipe = get_pipe()
            
            # Clear GPU cache before inference to maximize available memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[INFO] GPU memory before inference: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            except Exception as e:
                print(f"[WARN] Failed to clear GPU cache: {e}")
            
            # Process video using driven_sample
            start = time.time()
            
            # Create temporary directory for processing (needed for emotion .npy files)
            temp_dir = tempfile.mkdtemp()
            
            # driven_sample parameters (method signature from moda_test.py line 213):
            # - image_path, audio_path, cfg_scale=1., emo=8, save_dir=None, 
            #   smooth=False, silent_audio_path=None, silent_mode="post"
            # Get optional parameters from input
            cfg_scale = input_data.get("cfg_scale", 1.0)  # Guidance scale for generation
            emo_input = input_data.get("emo", 8)  # Emotion: 0-8 for codes, or base64 .npy string
            emo = process_emotion_input(emo_input, temp_dir)  # Process emotion (int code or .npy file)
            smooth = input_data.get("smooth", False)  # Smooth motion transitions
            
            # Create temporary directory for output (driven_sample needs save_dir, not file path)
            save_dir = tempfile.mkdtemp()
            
            print(f"   Save dir: {save_dir}")
            print(f"   Parameters: cfg_scale={cfg_scale}, emo={emo}, smooth={smooth}")
            
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
            
            # Clear GPU memory after inference
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("[INFO] GPU memory cleared")
            except Exception as e:
                print(f"[WARN] Failed to clear GPU memory: {e}")
            
            # Read and encode output video
            start = time.time()
            with open(output_path, 'rb') as f:
                video_data = f.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            timing['encode'] = time.time() - start
            
            # Calculate total time
            timing['total'] = time.time() - start_total
            
            print(f"✅ Video generated: {len(video_data)} bytes")
            print(f"⏱️  Performance metrics:")
            print(f"   Validation: {timing['validation']:.2f}s")
            print(f"   Decode: {timing['decode']:.2f}s")
            print(f"   Audio conversion: {timing['audio_conversion']:.2f}s")
            print(f"   Inference: {timing['inference']:.2f}s")
            print(f"   Encode: {timing['encode']:.2f}s")
            print(f"   Total: {timing['total']:.2f}s")
            
            return {
                "video": video_base64,
                "timing": timing
            }
                
        finally:
            # Cleanup temporary files
            cleanup_paths = [
                ("image", img_path),
                ("audio", aud_path),
                ("temp audio", aud_path_temp),
                ("output", output_path)
            ]
            
            for name, path in cleanup_paths:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                        print(f"[DEBUG] Cleaned up {name}: {path}")
                    except Exception as e:
                        print(f"[WARN] Failed to cleanup {name} at {path}: {e}")
            
            # Cleanup temporary directories
            if 'save_dir' in locals() and save_dir and os.path.exists(save_dir):
                try:
                    shutil.rmtree(save_dir)
                    print(f"[DEBUG] Cleaned up save directory: {save_dir}")
                except Exception as e:
                    print(f"[WARN] Failed to cleanup save_dir at {save_dir}: {e}")
            
            if 'temp_dir' in locals() and temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"[DEBUG] Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    print(f"[WARN] Failed to cleanup temp_dir at {temp_dir}: {e}")
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"❌ Error [{error_type}]: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Structured error response for better client debugging
        return create_error_response(error_type, error_msg)

# ============================================================================
# WORKER STARTUP: Pre-download models and initialize pipeline
# ============================================================================
# This eliminates the 86-second IN_QUEUE delay on first request by doing
# initialization at container startup instead of on first request
# ============================================================================

print("[INFO] ========================================")
print("[INFO] RunPod Worker Starting...")
print("[INFO] ========================================")

# Step 1: Ensure models are downloaded to disk
print("[INFO] Step 1/2: Checking models on disk...")
try:
    ensure_models_downloaded()
    print("[OK] Models are ready on disk")
except Exception as e:
    print(f"[WARN] Model download failed: {e}")
    print("       Models will be downloaded on first request instead")

# Step 2: CONDITIONAL INITIALIZATION
# Use EAGER INIT for always-on workers (min_workers=1), LAZY INIT for cold start (min_workers=0)
# Controlled by USE_EAGER_INIT environment variable (runtime infrastructure setting, not model config)
# 
# Architecture: Model configuration (batch_size, performance) → runpod_config.yaml
#               Runtime behavior (worker initialization) → Environment variables
# 
# Set in RunPod: Environment Variables → USE_EAGER_INIT=true/false
# Or in .env file for local development: USE_EAGER_INIT=false
USE_EAGER_INIT = os.getenv('USE_EAGER_INIT', 'false').lower() == 'true'

if USE_EAGER_INIT:
    # EAGER INIT - for always-on workers (min_workers=1)
    # Worker becomes "Ready" slower (80-260 sec), but first request is fast (20 sec)
    print("[INFO] Step 2/2: Initializing MoDA pipeline (EAGER INIT for A6000/A40 48GB)...")
    print("[INFO] USE_EAGER_INIT=true - pipeline will be loaded to GPU at startup")
    print("[INFO] This takes 60-80 seconds but happens ONCE at startup")
    print("[INFO] After this, first request will be instant!")
    
    try:
        startup_start = time.time()
        
        # Pre-initialize pipeline (loads models to GPU)
        print("[INFO] Loading LiveVASAPipeline to GPU...")
        pipe = get_pipe()
        
        # Additional optimizations for A6000/A40 48GB
        try:
            import torch
            if torch.cuda.is_available():
                # Set optimal memory allocator for A6000/A40
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
                
                # Enable optimizations for A6000/A40
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')  # Faster matmul on A6000/A40
                    print("[OK] High precision matmul enabled for A6000/A40")
                
                # Clear cache after initialization
                torch.cuda.empty_cache()
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[INFO] GPU memory after init: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        except Exception as e:
            print(f"[WARN] GPU optimization failed: {e}")
        
        startup_time = time.time() - startup_start
        print(f"[OK] ✅ MoDA pipeline initialized in {startup_time:.1f} seconds")
        print("[OK] ✅ Worker is TRULY ready - optimized for A6000/A40 48GB!")
        print("[OK] ✅ Batch size: 50, torch.compile: enabled")
        print("[INFO] ========================================")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to pre-initialize pipeline: {e}")
        print("[WARN] Pipeline will initialize on first request (slower)")
        print(f"[WARN] Error details: {str(e)[:500]}")
        import traceback
        traceback.print_exc()
else:
    # LAZY INIT - for cold start workers (min_workers=0)
    # Worker becomes "Ready" faster (20-50 sec), but first request is slower (80-100 sec)
    print("[INFO] Step 2/2: Skipping eager initialization (LAZY INIT mode)")
    print("[INFO] USE_EAGER_INIT=false or not set - pipeline will initialize on first request")
    print("[INFO] Worker becomes Ready faster (~20-50 sec), but first request will take 60-80 seconds")
    print("[INFO] This is optimal for cold start strategy (min_workers=0)")
    print("[INFO] To enable eager init for always-on workers, set USE_EAGER_INIT=true in RunPod environment variables")
    print("[INFO] ========================================")

# Start server
print("[INFO] Starting RunPod serverless handler...")
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

