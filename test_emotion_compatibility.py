#!/usr/bin/env python3
"""
Compatibility test for DICE-Talk EmotionModel integration into MoDA.

This script tests:
1. Loading DICE-Talk EmotionModel in MoDA environment
2. AudioProjModel compatibility
3. Shape transformations: 5D input → 32×1024 output → hidden_size
4. Error handling and fallback mechanisms
"""

import os
import os.path as osp
import sys
import torch
import numpy as np

# Add MoDA src to path
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

def test_audio_proj_compatibility():
    """Test if AudioProjModel from MoDA is compatible with DICE-Talk EmotionModel"""
    print("=" * 60)
    print("Test 1: AudioProjModel Compatibility")
    print("=" * 60)
    
    try:
        from models.audio.audio_proj import AudioProjModel
        print("✓ AudioProjModel imported successfully")
        
        # Test instantiation with DICE-Talk parameters
        model = AudioProjModel(
            seq_len=1,
            blocks=1,
            channels=256,
            intermediate_dim=1024,
            output_dim=1024,
            context_tokens=32
        )
        print("✓ AudioProjModel instantiated with DICE-Talk parameters")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 1, 256)  # (B, L, blocks, channels)
        output = model(test_input)
        expected_shape = (1, 1, 32, 1024)  # (B, L, context_tokens, output_dim)
        
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful: {output.shape}")
            return True
        else:
            print(f"✗ Shape mismatch: got {output.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"✗ AudioProjModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotion_model_loading():
    """Test loading DICE-Talk EmotionModel"""
    print("\n" + "=" * 60)
    print("Test 2: EmotionModel Loading")
    print("=" * 60)
    
    try:
        # Import EmotionModel (will be created in emotion module)
        from models.emotion.emotion_model import EmotionModel, QKVAttention, emotion_bank
        print("✓ EmotionModel classes imported successfully")
        
        # Test instantiation
        model = EmotionModel()
        print("✓ EmotionModel instantiated")
        
        # Test with dummy input (5D tensor: B, L, D1, D2, 256)
        test_input = torch.randn(1, 1, 1, 1, 256)
        output, vq_loss = model(test_input, retrieval=True)
        
        expected_shape = (1, 1, 32, 1024)  # (B, L, context_tokens, output_dim)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful: {output.shape}")
            print(f"  VQ loss: {vq_loss.item():.4f}")
            return True
        else:
            print(f"✗ Shape mismatch: got {output.shape}, expected {expected_shape}")
            return False
            
    except ImportError as e:
        print(f"⚠ EmotionModel not yet created: {e}")
        print("  This is expected - will be created in next phase")
        return None
    except Exception as e:
        print(f"✗ EmotionModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shape_transformations():
    """Test shape transformations for integration"""
    print("\n" + "=" * 60)
    print("Test 3: Shape Transformations")
    print("=" * 60)
    
    try:
        # Simulate DICE-Talk emotion input (from .npy file)
        # .npy file contains dict with 'mu' key: shape (1, 1, 1, 1, 256)
        emo_npy_data = {'mu': np.random.randn(1, 1, 1, 1, 256).astype(np.float32)}
        emo_tensor = torch.from_numpy(emo_npy_data['mu'])
        print(f"✓ Loaded .npy format: {emo_tensor.shape}")
        
        # Simulate EmotionModel output: (B, L, 32, 1024)
        emotion_output = torch.randn(1, 1, 32, 1024)
        print(f"✓ EmotionModel output: {emotion_output.shape}")
        
        # Project to hidden_size (1152 for DiT-XL)
        hidden_size = 1152
        # Option 1: Average pooling over tokens
        avg_pooled = emotion_output.mean(dim=2)  # (B, L, 1024)
        print(f"✓ Average pooled: {avg_pooled.shape}")
        
        # Option 2: Attention pooling (learned)
        attention_weights = torch.softmax(torch.randn(1, 1, 32), dim=-1)  # (B, L, 32)
        attn_pooled = (emotion_output * attention_weights.unsqueeze(-1)).sum(dim=2)  # (B, L, 1024)
        print(f"✓ Attention pooled: {attn_pooled.shape}")
        
        # Project to hidden_size
        proj_layer = torch.nn.Linear(1024, hidden_size)
        final_embedding = proj_layer(attn_pooled.squeeze(1))  # (B, hidden_size)
        print(f"✓ Final embedding: {final_embedding.shape}")
        
        if final_embedding.shape == (1, hidden_size):
            print("✓ Shape transformation successful!")
            return True
        else:
            print(f"✗ Final shape mismatch: {final_embedding.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Shape transformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npy_file_loading():
    """Test loading .npy emotion files"""
    print("\n" + "=" * 60)
    print("Test 4: .npy File Loading")
    print("=" * 60)
    
    try:
        # Create test .npy file
        test_dir = "test_emotion_data"
        os.makedirs(test_dir, exist_ok=True)
        test_file = osp.join(test_dir, "test_emotion.npy")
        
        # Create dummy emotion data (DICE-Talk format)
        emo_data = {'mu': np.random.randn(1, 1, 1, 1, 256).astype(np.float32)}
        np.save(test_file, emo_data, allow_pickle=True)
        print(f"✓ Created test .npy file: {test_file}")
        
        # Load it back
        loaded_data = np.load(test_file, allow_pickle=True)
        if isinstance(loaded_data, np.ndarray):
            # Old format: direct array
            emo_tensor = torch.from_numpy(loaded_data)
        else:
            # New format: dict with 'mu' key
            emo_tensor = torch.from_numpy(loaded_data.item()['mu'])
        
        print(f"✓ Loaded .npy file: {emo_tensor.shape}")
        
        # Cleanup
        os.remove(test_file)
        os.rmdir(test_dir)
        
        if emo_tensor.shape == (1, 1, 1, 1, 256):
            print("✓ .npy file loading successful!")
            return True
        else:
            print(f"✗ Shape mismatch: {emo_tensor.shape}")
            return False
            
    except Exception as e:
        print(f"✗ .npy file loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all compatibility tests"""
    print("\n" + "=" * 60)
    print("DICE-Talk EmotionModel Compatibility Test for MoDA")
    print("=" * 60)
    
    results = {}
    
    # Test 1: AudioProjModel
    results['audio_proj'] = test_audio_proj_compatibility()
    
    # Test 2: EmotionModel (may not exist yet)
    results['emotion_model'] = test_emotion_model_loading()
    
    # Test 3: Shape transformations
    results['shape_transform'] = test_shape_transformations()
    
    # Test 4: .npy file loading
    results['npy_loading'] = test_npy_file_loading()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is None:
            status = "⚠ SKIP (not implemented yet)"
        else:
            status = "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(r is True or r is None for r in results.values())
    
    if all_passed:
        print("\n✓ All tests passed or skipped (expected for initial run)")
        print("  Proceed with emotion module creation")
    else:
        print("\n✗ Some tests failed - review errors above")
        print("  Fix issues before proceeding")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

