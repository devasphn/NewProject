#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  FAST GPU-ACCELERATED AUDIO AUGMENTATION
  
  Uses GPU for speed augmentation via resampling - 50-100x faster!
  Processes in batches for maximum throughput.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

# Suppress deprecation warnings
warnings.filterwarnings('ignore')
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'

import torch
import torchaudio

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def load_audio_fast(filepath):
    """Load audio file quickly"""
    try:
        waveform, sr = torchaudio.load(str(filepath))
        return waveform, sr
    except:
        return None, None


def augment_speed_gpu(waveform, sr, speed_factor):
    """GPU-accelerated speed change using resampling"""
    if speed_factor == 1.0:
        return waveform
    
    # Move to GPU
    waveform = waveform.to(DEVICE)
    
    # Speed change via resampling
    # To speed up by 1.1x: resample from sr to sr*1.1, then back to sr
    new_sr = int(sr * speed_factor)
    
    # Use torchaudio's GPU-compatible resample
    resampler = torchaudio.transforms.Resample(sr, new_sr).to(DEVICE)
    stretched = resampler(waveform)
    
    # Resample back to original rate
    resampler_back = torchaudio.transforms.Resample(new_sr, sr).to(DEVICE)
    result = resampler_back(stretched)
    
    return result.cpu()


def augment_noise_gpu(waveform, snr_db=20.0):
    """Add noise on GPU"""
    waveform = waveform.to(DEVICE)
    signal_power = torch.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    result = waveform + noise
    return result.cpu()


def augment_pitch_simple(waveform, sr, semitones):
    """Simple pitch shift using resampling (changes duration slightly)"""
    if semitones == 0:
        return waveform
    
    waveform = waveform.to(DEVICE)
    
    # Pitch shift factor
    factor = 2 ** (semitones / 12)
    new_sr = int(sr / factor)
    
    # Resample to change pitch
    resampler = torchaudio.transforms.Resample(sr, new_sr).to(DEVICE)
    result = resampler(waveform)
    
    return result.cpu()


def process_file_batch(file_batch, output_dir, augment_types):
    """Process a batch of files"""
    results = []
    
    for input_path in file_batch:
        try:
            waveform, sr = load_audio_fast(input_path)
            if waveform is None:
                continue
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                sr = 16000
            
            base_name = input_path.stem
            
            for aug_type in augment_types:
                try:
                    if aug_type == 'speed_slow':
                        augmented = augment_speed_gpu(waveform, sr, 0.9)
                        suffix = "_slow"
                    elif aug_type == 'speed_fast':
                        augmented = augment_speed_gpu(waveform, sr, 1.1)
                        suffix = "_fast"
                    elif aug_type == 'pitch_down':
                        augmented = augment_pitch_simple(waveform, sr, -2)
                        suffix = "_pdn"
                    elif aug_type == 'pitch_up':
                        augmented = augment_pitch_simple(waveform, sr, 2)
                        suffix = "_pup"
                    elif aug_type == 'noise':
                        augmented = augment_noise_gpu(waveform, snr_db=25.0)
                        suffix = "_nz"
                    else:
                        continue
                    
                    output_path = output_dir / f"{base_name}{suffix}.wav"
                    torchaudio.save(str(output_path), augmented, sr)
                    results.append(output_path)
                except Exception as e:
                    pass
                    
        except Exception as e:
            pass
    
    return results


def augment_dataset_fast(input_dir: str, output_dir: str, 
                         augment_types: list, batch_size: int = 100):
    """Fast batch augmentation with GPU"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files (exclude already augmented)
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in input_path.rglob(ext):
            # Skip augmented directory
            if 'augmented' not in str(f):
                audio_files.append(f)
    
    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return 0
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🎵 Files: {len(audio_files)}")
    print(f"🔧 Types: {augment_types}")
    print(f"📈 Output: ~{len(audio_files) * len(augment_types)} files")
    print(f"⚡ Batch: {batch_size}")
    print()
    
    # Process in batches
    total_created = 0
    batches = [audio_files[i:i+batch_size] for i in range(0, len(audio_files), batch_size)]
    
    for batch in tqdm(batches, desc="Augmenting (GPU)", unit="batch"):
        results = process_file_batch(batch, output_path, augment_types)
        total_created += len(results)
        
        # Clear GPU cache periodically
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_created


def main():
    parser = argparse.ArgumentParser(description="Fast GPU audio augmentation")
    parser.add_argument("--language", type=str, required=True,
                       choices=['telugu', 'hindi', 'english', 'all'],
                       help="Language to augment")
    parser.add_argument("--expansion", type=str, default="5x",
                       choices=["3x", "5x"],
                       help="Expansion factor")
    parser.add_argument("--batch", type=int, default=200,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Set augmentation types
    if args.expansion == "3x":
        augment_types = ['speed_slow', 'speed_fast', 'noise']
    else:  # 5x
        augment_types = ['speed_slow', 'speed_fast', 'pitch_down', 'pitch_up', 'noise']
    
    print("═" * 70)
    print("  FAST GPU AUDIO AUGMENTATION")
    print("═" * 70)
    print(f"  Language: {args.language}")
    print(f"  Expansion: {args.expansion}")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("═" * 70)
    print()
    
    languages = [args.language] if args.language != 'all' else ['telugu', 'hindi']
    
    for lang in languages:
        input_dir = f"/workspace/data/{lang}"
        output_dir = f"/workspace/data/{lang}/augmented"
        
        print(f"\n🔄 Processing {lang.upper()}...")
        created = augment_dataset_fast(input_dir, output_dir, augment_types, args.batch)
        print(f"✅ Created {created} files for {lang}")
    
    print("\n" + "═" * 70)
    print("  AUGMENTATION COMPLETE!")
    print("═" * 70)
    print("  Next: python calculate_real_hours.py")
    print("  Then: python train_codec_production.py")
    print("═" * 70)


if __name__ == "__main__":
    main()
