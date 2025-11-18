"""
Debug script to check validation data quality
Run this to see what your validation samples actually look like
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np

# Settings (match your training config)
data_dir = Path("/workspace/telugu_data/raw")
segment_length = 32000  # 2 seconds at 16kHz
val_ratio = 0.1
test_ratio = 0.1

# Get all audio files
all_audio_files = list(data_dir.rglob("*.wav"))
all_audio_files.sort()

# Create validation split (same logic as training)
total_files = len(all_audio_files)
n_test = int(total_files * test_ratio)
n_val = int(total_files * val_ratio)
n_train = total_files - n_test - n_val

val_files = all_audio_files[n_train:n_train + n_val]

print(f"=== VALIDATION DATA ANALYSIS ===\n")
print(f"Total files: {total_files}")
print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}\n")

print(f"Validation files ({len(val_files)}):")
for i, f in enumerate(val_files):
    print(f"  {i+1}. {f.name}")

print("\n=== DETAILED ANALYSIS ===\n")

for idx, audio_path in enumerate(val_files):
    print(f"\n--- File {idx+1}: {audio_path.name} ---")
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        print(f"Original:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {waveform.shape[0]}")
        print(f"  Samples: {waveform.shape[1]:,}")
        print(f"  Duration: {waveform.shape[1]/sample_rate:.2f} seconds")
        print(f"  Range: [{waveform.min():.6f}, {waveform.max():.6f}]")
        print(f"  Mean: {waveform.mean():.6f}")
        print(f"  Std: {waveform.std():.6f}")
        print(f"  RMS: {(waveform**2).mean().sqrt():.6f}")
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            print(f"\nAfter resampling to 16kHz:")
            print(f"  Samples: {waveform.shape[1]:,}")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print(f"  Converted to mono")
        
        # Crop or pad
        original_len = waveform.shape[1]
        if waveform.shape[1] > segment_length:
            waveform = waveform[:, :segment_length]
            print(f"\nAfter cropping to {segment_length} samples:")
            print(f"  Cropped {original_len - segment_length:,} samples")
        elif waveform.shape[1] < segment_length:
            padding = segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            print(f"\nAfter padding to {segment_length} samples:")
            print(f"  Added {padding:,} zero samples ({padding/segment_length*100:.1f}% padding)")
        
        # Final stats
        print(f"\nFinal processed sample:")
        print(f"  Shape: {waveform.shape}")
        print(f"  Range: [{waveform.min():.6f}, {waveform.max():.6f}]")
        print(f"  Mean: {waveform.mean():.6f}")
        print(f"  Std: {waveform.std():.6f}")
        print(f"  RMS: {(waveform**2).mean().sqrt():.6f}")
        
        # Check for problems
        num_zeros = (waveform.abs() < 1e-6).sum().item()
        zero_pct = num_zeros / segment_length * 100
        print(f"\n  Zero/silent samples: {num_zeros:,} ({zero_pct:.1f}%)")
        
        if zero_pct > 50:
            print(f"  ⚠️  WARNING: >50% zeros/silence!")
        
        if waveform.std() < 0.001:
            print(f"  ⚠️  WARNING: Very low variance (mostly silent)")
        
        if waveform.abs().max() < 0.01:
            print(f"  ⚠️  WARNING: Very quiet audio")
            
    except Exception as e:
        print(f"  ❌ ERROR loading file: {e}")

print("\n\n=== SUMMARY ===")
print("If you see:")
print("  - >50% padding: Validation files too short")
print("  - Very low variance: Files are mostly silent")
print("  - Very quiet audio: Files need normalization")
print("\nThese issues would cause negative SNR!")
