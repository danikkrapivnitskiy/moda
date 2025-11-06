"""
Emotion adapter for integrating DICE-Talk EmotionModel into MoDA.

This adapter handles:
- Format conversion: int emotion codes → .npy format → EmotionModel input
- Shape transformation: EmotionModel output (32×1024) → hidden_size vector
- Fallback to simple LabelEmbedder if EmotionModel unavailable
"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Optional

from .emotion_model import EmotionModel


# Emotion code to .npy file mapping
# Maps MoDA emotion codes (0-8) to DICE-Talk emotion names
EMOTION_CODE_TO_NAME = {
    0: 'angry',
    1: 'contempt', 
    2: 'disgusted',
    3: 'fear',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised',
    8: 'neutral'  # None -> neutral
}


class EmotionAdapter(nn.Module):
    """
    Adapter that wraps DICE-Talk EmotionModel for MoDA integration.
    
    Handles:
    - Loading emotion features from .npy files
    - Converting int emotion codes to .npy format
    - Projecting EmotionModel output (32×1024) to hidden_size
    - Fallback to simple embedding if EmotionModel unavailable
    """
    
    def __init__(
        self,
        hidden_size: int = 1152,
        emotion_model_path: Optional[str] = None,
        emotion_examples_dir: Optional[str] = None,
        use_enhanced_emotion: bool = True
    ):
        """
        Initialize emotion adapter.
        
        Args:
            hidden_size: Output dimension (DiT hidden size, typically 1152)
            emotion_model_path: Path to emo_model.pth checkpoint (optional, auto-detected if None)
            emotion_examples_dir: Directory containing emotion .npy files
            use_enhanced_emotion: Whether to use DICE-Talk EmotionModel (default: True)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.emotion_examples_dir = emotion_examples_dir
        self.use_enhanced_emotion = use_enhanced_emotion
        self.emotion_model_available = False
        
        # Initialize EmotionModel if enabled
        if use_enhanced_emotion:
            try:
                self.emotion_model = EmotionModel()
                self.emotion_model_available = True
                
                # Auto-detect emotion model path if not provided
                if emotion_model_path is None:
                    emotion_model_path = self._find_emotion_model()
                
                # Load checkpoint if found
                if emotion_model_path and osp.exists(emotion_model_path):
                    try:
                        state_dict = torch.load(emotion_model_path, map_location='cpu', weights_only=False)
                        self.emotion_model.load_state_dict(state_dict, strict=False)
                        print(f"[INFO] Loaded DICE-Talk emotion model from {emotion_model_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to load emotion checkpoint: {e}")
                        print(f"[WARN] Using randomly initialized EmotionModel")
                else:
                    print(f"[WARN] Emotion checkpoint not found, using randomly initialized EmotionModel")
                    print(f"[WARN] Searched in standard locations, but emo_model.pth not found")
                    print(f"[WARN] For best results, ensure emo_model.pth is available")
                
            except Exception as e:
                print(f"[WARN] Failed to initialize EmotionModel: {e}")
                print(f"[WARN] Falling back to simple emotion embedding")
                self.emotion_model_available = False
                self.use_enhanced_emotion = False
        
        # Projection layer: 32 tokens × 1024 dim → hidden_size
        # Use attention pooling to combine 32 tokens into single vector
        self.attention_pool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1)  # Attention weights for each token
        )
        self.projection = nn.Linear(1024, hidden_size)
    
    def _find_emotion_model(self) -> Optional[str]:
        """
        Auto-detect emotion model path in standard locations.
        
        Priority:
        1. /workspace/checkpoints/DICE-Talk/emo_model.pth (persistent storage)
        2. /app/models_cache/checkpoints/DICE-Talk/emo_model.pth (pre-built cache)
        3. checkpoints/DICE-Talk/emo_model.pth (local development)
        4. Relative to current file
        
        Returns:
            Path to emo_model.pth or None if not found
        """
        # Standard search paths (same pattern as other MoDA models)
        search_paths = [
            # Persistent storage (RunPod)
            "/workspace/checkpoints/DICE-Talk/emo_model.pth",
            # Pre-built cache (Docker image)
            "/app/models_cache/checkpoints/DICE-Talk/emo_model.pth",
            # Local development
            "checkpoints/DICE-Talk/emo_model.pth",
            # Relative to project root
            osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), 
                    "checkpoints", "DICE-Talk", "emo_model.pth"),
        ]
        
        for path in search_paths:
            if osp.exists(path):
                return path
        
        return None
        
    def load_emotion_from_npy(self, npy_path: str) -> torch.Tensor:
        """
        Load emotion features from .npy file.
        
        Args:
            npy_path: Path to .npy file
        
        Returns:
            Emotion tensor: (1, 1, 1, 1, 256)
        """
        try:
            data = np.load(npy_path, allow_pickle=True)
            
            # Handle both formats:
            # 1. Direct array (old format)
            # 2. Dict with 'mu' key (DICE-Talk format)
            if isinstance(data, np.ndarray):
                emo_tensor = torch.from_numpy(data).float()
            else:
                # Dict format
                emo_tensor = torch.from_numpy(data.item()['mu']).float()
            
            # Ensure correct shape: (1, 1, 1, 1, 256)
            if emo_tensor.dim() == 5:
                return emo_tensor
            elif emo_tensor.dim() == 1:
                return emo_tensor.view(1, 1, 1, 1, -1)
            else:
                # Reshape to expected format
                return emo_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                
        except Exception as e:
            print(f"[ERROR] Failed to load emotion from {npy_path}: {e}")
            raise
    
    def get_emotion_npy_path(self, emotion_code: int) -> Optional[str]:
        """
        Get path to .npy file for given emotion code.
        
        Args:
            emotion_code: Emotion code (0-8)
        
        Returns:
            Path to .npy file or None if not found
        """
        if self.emotion_examples_dir is None:
            return None
        
        emotion_name = EMOTION_CODE_TO_NAME.get(emotion_code, 'neutral')
        npy_path = osp.join(self.emotion_examples_dir, f"{emotion_name}.npy")
        
        if osp.exists(npy_path):
            return npy_path
        else:
            # Try neutral as fallback
            neutral_path = osp.join(self.emotion_examples_dir, "neutral.npy")
            if osp.exists(neutral_path):
                return neutral_path
            return None
    
    def forward(
        self,
        emotion_input: Union[int, torch.Tensor, str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Process emotion input and return embedding.
        
        Args:
            emotion_input: Can be:
                - int: Emotion code (0-8)
                - str: Path to .npy file
                - torch.Tensor: Direct emotion features (1, 1, 1, 1, 256)
            device: Target device for tensors
        
        Returns:
            Emotion embedding: (B, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Handle different input types
        if isinstance(emotion_input, int):
            # Emotion code: try to load .npy file, otherwise use simple embedding
            npy_path = self.get_emotion_npy_path(emotion_input)
            
            if npy_path and self.emotion_model_available:
                # Load from .npy and use EmotionModel
                emo_features = self.load_emotion_from_npy(npy_path).to(device)
                return self._process_with_emotion_model(emo_features)
            else:
                # Fallback: return simple embedding (will be handled by LabelEmbedder)
                # Return None to signal fallback
                return None
                
        elif isinstance(emotion_input, str):
            # .npy file path
            if self.emotion_model_available:
                emo_features = self.load_emotion_from_npy(emotion_input).to(device)
                return self._process_with_emotion_model(emo_features)
            else:
                print(f"[WARN] EmotionModel not available, cannot process .npy file")
                return None
                
        elif isinstance(emotion_input, torch.Tensor):
            # Direct tensor input
            if self.emotion_model_available:
                emo_features = emotion_input.to(device)
                # Ensure correct shape
                if emo_features.dim() == 5:
                    return self._process_with_emotion_model(emo_features)
                else:
                    emo_features = emo_features.view(1, 1, 1, 1, -1)
                    return self._process_with_emotion_model(emo_features)
            else:
                return None
        else:
            raise ValueError(f"Unsupported emotion input type: {type(emotion_input)}")
    
    def _process_with_emotion_model(self, emo_features: torch.Tensor) -> torch.Tensor:
        """
        Process emotion features through EmotionModel and project to hidden_size.
        
        Args:
            emo_features: (B, L, D1, D2, 256) emotion features
        
        Returns:
            Emotion embedding: (B, hidden_size)
        """
        # Forward through EmotionModel: (B, L, D1, D2, 256) → (B, L, 32, 1024)
        emo_tokens, _ = self.emotion_model(emo_features, retrieval=True)
        
        # emo_tokens: (B, L, 32, 1024)
        # We need to pool 32 tokens into single vector
        B, L, num_tokens, token_dim = emo_tokens.shape
        
        # Reshape for attention pooling: (B*L, 32, 1024)
        emo_tokens_flat = emo_tokens.view(B * L, num_tokens, token_dim)
        
        # Attention pooling: compute weights for each token
        attn_weights = self.attention_pool(emo_tokens_flat)  # (B*L, 32, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize
        
        # Weighted sum: (B*L, 1024)
        pooled = (emo_tokens_flat * attn_weights).sum(dim=1)
        
        # Project to hidden_size: (B*L, hidden_size)
        projected = self.projection(pooled)
        
        # Reshape back: (B, L, hidden_size) → (B, hidden_size) if L=1
        if L == 1:
            return projected.squeeze(1)
        else:
            return projected.view(B, L, self.hidden_size)

