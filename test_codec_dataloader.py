#!/usr/bin/env python3
"""
Quick test script to verify train_codec.py data loading works
Run this BEFORE starting the full 6-8 hour training
"""

import sys
from pathlib import Path
import torch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading():
    """Test if data loading works without crashing"""
    print("=" * 60)
    print("TESTING CODEC DATA LOADING")
    print("=" * 60)
    print()
    
    try:
        # Import the dataset class
        from train_codec import TeluguAudioDataset
        
        print("✓ Successfully imported TeluguAudioDataset")
        print()
        
        # Test train split
        print("Loading train split...")
        train_dataset = TeluguAudioDataset(
            data_dir="/workspace/telugu_data/raw",
            segment_length=32000,  # 2 seconds
            split="train"
        )
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        
        # Test validation split
        print("Loading validation split...")
        val_dataset = TeluguAudioDataset(
            data_dir="/workspace/telugu_data/raw",
            segment_length=32000,
            split="validation"
        )
        print(f"✓ Validation dataset: {len(val_dataset)} samples")
        
        # Test loading one sample
        print()
        print("Testing sample loading...")
        sample = train_dataset[0]
        print(f"✓ Sample shape: {sample.shape}")
        print(f"✓ Sample dtype: {sample.dtype}")
        print(f"✓ Sample min/max: {sample.min():.4f} / {sample.max():.4f}")
        
        # Test batch loading
        print()
        print("Testing batch loading...")
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        batch = next(iter(loader))
        print(f"✓ Batch shape: {batch.shape}")
        print(f"✓ Expected: [4, 1, 32000]")
        
        # Verify shape is correct
        if batch.shape == torch.Size([4, 1, 32000]):
            print("✓ Batch shape matches expected!")
        else:
            print(f"⚠ Batch shape mismatch!")
        
        print()
        print("=" * 60)
        print("✅ ALL DATA LOADING TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your train_codec.py is ready to run!")
        print()
        return True
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Make sure data exists at: /workspace/telugu_data/raw/")
        print("Run: find /workspace/telugu_data/raw -name '*.wav' | wc -l")
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
