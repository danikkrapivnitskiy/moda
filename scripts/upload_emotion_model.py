#!/usr/bin/env python3
"""
Script to upload emo_model.pth to HuggingFace repository {HUGGINGFACE_USERNAME}/moda-emotion-model
"""

import os
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

def main():
    hf_username = os.getenv("HUGGINGFACE_USERNAME")
    if not hf_username:
        print("[ERROR] HUGGINGFACE_USERNAME environment variable is not set")
        sys.exit(1)

    repo_id = f"{hf_username}/moda-emotion-model"
    emo_model_path = project_root / "emo_model.pth"
    
    print("=" * 60)
    print("Uploading DICE-Talk Emotion Model to HuggingFace")
    print("=" * 60)
    print()
    
    # Check if file exists
    if not emo_model_path.exists():
        print(f"[ERROR] emo_model.pth not found at: {emo_model_path}")
        print()
        print("Please ensure emo_model.pth is in the MoDA root directory")
        sys.exit(1)
    
    # Check file size
    file_size_mb = emo_model_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Found emo_model.pth")
    print(f"       Location: {emo_model_path}")
    print(f"       Size: {file_size_mb:.1f} MB")
    print()
    
    # Check for HF_TOKEN
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN environment variable is not set")
        print()
        print("Please set your HuggingFace token:")
        print("  export HF_TOKEN='your_token_here'")
        print()
        print("Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[ERROR] huggingface_hub not installed")
        print("        Install with: pip install 'huggingface_hub'")
        sys.exit(1)
    
    print(f"[INFO] Repository: {repo_id}")
    print(f"[INFO] This will create the repository if it doesn't exist")
    print()
    
    response = input("Continue with upload? (y/N): ")
    if response.lower() != 'y':
        print("[INFO] Upload cancelled")
        sys.exit(0)
    
    print()
    print("[INFO] Uploading emo_model.pth to HuggingFace...")
    print(f"       This may take a few minutes (~{file_size_mb:.0f} MB)...")
    print()
    
    try:
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,
                token=hf_token,
                exist_ok=True
            )
            print(f"[INFO] Repository {repo_id} ready")
        except Exception as e:
            print(f"[INFO] Repository already exists or creation skipped: {e}")
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=str(emo_model_path),
            path_in_repo="emo_model.pth",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message="Add emo_model.pth for MoDA emotion integration"
        )
        
        print()
        print(f"[OK] Successfully uploaded emo_model.pth to {repo_id}")
        print()
        print(f"Repository URL: https://huggingface.co/{repo_id}")
        print()
        print("The model will be automatically downloaded by:")
        print("  - Dockerfile (during Docker build)")
        print("  - runpod_server.py (during RunPod initialization)")
        print("  - download_emotion_model.sh (for local development)")
        print()
        
    except Exception as e:
        print()
        print(f"[ERROR] Failed to upload emo_model.pth: {e}")
        print()
        print("Possible issues:")
        print("  1. Invalid HF_TOKEN")
        print("  2. Network connection problem")
        print("  3. Repository permissions")
        sys.exit(1)

if __name__ == "__main__":
    main()

