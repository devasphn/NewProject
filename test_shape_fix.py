#!/usr/bin/env python3
"""
Quick test to verify shape mismatch fix works
Run this BEFORE restarting the full 6-8 hour training
"""

import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_shape_fix():
    """Test if decoder shape fix works"""
    print("=" * 60)
    print("TESTING SHAPE MISMATCH FIX")
    print("=" * 60)
    print()
    
    try:
        from telugu_codec import TeluCodec
        
        print("✓ Successfully imported TeluCodec")
        print()
        
        # Create codec
        codec = TeluCodec()
        print(f"✓ TeluCodec initialized")
        print(f"  Parameters: {sum(p.numel() for p in codec.parameters())/1e6:.2f}M")
        print()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        codec = codec.to(device)
        print(f"✓ Device: {device}")
        print()
        
        # Test with the problematic length
        print("Testing with batch_size=32, length=32000 (the failing case)...")
        audio = torch.randn(32, 1, 32000).to(device)
        
        with torch.no_grad():
            output = codec(audio)
        
        print(f"✓ Input shape: {audio.shape}")
        print(f"✓ Output shape: {output['audio'].shape}")
        print()
        
        # Verify shapes match
        if output["audio"].shape == audio.shape:
            print("✅ SHAPES MATCH! Fix works correctly!")
        else:
            print(f"❌ SHAPES DON'T MATCH!")
            print(f"   Expected: {audio.shape}")
            print(f"   Got: {output['audio'].shape}")
            return False
        
        print()
        
        # Test with various lengths
        print("Testing with various audio lengths...")
        for length in [16000, 24000, 32000, 40000, 48000]:
            audio = torch.randn(4, 1, length).to(device)
            
            with torch.no_grad():
                output = codec(audio)
            
            if output["audio"].shape == audio.shape:
                print(f"  ✓ Length {length:5d}: {audio.shape} → {output['audio'].shape}")
            else:
                print(f"  ❌ Length {length:5d}: MISMATCH!")
                return False
        
        print()
        
        # Test losses are computed
        print("Testing loss computation...")
        audio = torch.randn(2, 1, 32000).to(device)
        output = codec(audio)
        
        print(f"  ✓ Total loss: {output['loss'].item():.4f}")
        print(f"  ✓ Recon loss: {output['recon_loss'].item():.4f}")
        print(f"  ✓ VQ loss: {output['vq_loss'].item():.4f}")
        print(f"  ✓ Perceptual loss: {output['perceptual_loss'].item():.4f}")
        
        print()
        print("=" * 60)
        print("✅ ALL SHAPE TESTS PASSED!")
        print("=" * 60)
        print()
        print("The shape mismatch fix works correctly!")
        print("You can now safely start the full training.")
        print()
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_shape_fix()
    sys.exit(0 if success else 1)
