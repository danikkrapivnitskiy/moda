"""
Emotion module for MoDA with DICE-Talk integration support.

This module provides enhanced emotion processing capabilities by integrating
DICE-Talk's sophisticated emotion adapter with VQ-VAE codebook and attention mechanisms.
"""

from .emotion_model import EmotionModel, QKVAttention, emotion_bank, Classifier
from .emotion_adapter import EmotionAdapter

__all__ = [
    'EmotionModel',
    'QKVAttention', 
    'emotion_bank',
    'Classifier',
    'EmotionAdapter'
]

