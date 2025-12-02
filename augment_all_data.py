#!/usr/bin/env python3
"""
Comprehensive Data Augmentation for All Languages

Applies augmentation to expand training data:
- Speed perturbation (0.9x, 1.0x, 1.1x) = 3x data
- Volume normalization
- Optional noise injection
- Optional reverb

Run on RunPod: python augment_all_data.py

Expected output: 2-3x more training data
"""

import os
import sys
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")
SAMPLE_RATE = 16000
NUM_WORKERS = 8  # Adjust based on CPU cores

# Augmentation settings
SPEED_FACTORS = [0.9, 1.1]  # Original + these = 3x data
ADD_NOISE = True
NOISE_SNR_RANGE = (15, 30)  # dB
ADD_REVERB = False  # Slower, optional

# Languages to augment
LANGUAGES = ['english', 'hindi', 'telugu']

# Subdirectories to augment
SUBDIRS = ['librispeech', 'gramvaani', 'openslr103', 'openslr66', 'kathbath', 'indicvoices', 'commonvoice']


def get_audio_files(directory: Path) -> list:
    """Get all audio files in directory."""
    extensions = ['.wav', '.flac', '.mp3', '.ogg']
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}'))
    return files


def load_audio(path: Path) -> tuple:
    """Load audio file."""
    try:
        waveform, sr = torchaudio.load(str(path))
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, SAMPLE_RATE
    except Exception as e:
        return None, None


def speed_perturb(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    """Apply speed perturbation via resampling."""
    if abs(factor - 1.0) < 0.01:
        return waveform
    
    orig_sr = SAMPLE_RATE
    new_sr = int(SAMPLE_RATE * factor)
    
    # Resample to change speed
    waveform = torchaudio.functional.resample(waveform, orig_sr, new_sr)
    # Resample back to original sample rate
    waveform = torchaudio.functional.resample(waveform, new_sr, orig_sr)
    
    return waveform


def add_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise at specified SNR."""
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


def normalize_volume(waveform: torch.Tensor, target_db: float = -20) -> torch.Tensor:
    """Normalize audio to target dB level."""
    rms = torch.sqrt(waveform.pow(2).mean())
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        waveform = waveform * (target_rms / rms)
    
    # Clip to prevent clipping
    waveform = torch.clamp(waveform, -1.0, 1.0)
    return waveform


def process_file(args):
    """Process a single file with augmentation."""
    input_path, output_dir, speed_factor, add_noise_flag = args
    
    try:
        # Load audio
        waveform, sr = load_audio(input_path)
        if waveform is None:
            return None
        
        # Apply speed perturbation
        if speed_factor != 1.0:
            waveform = speed_perturb(waveform, speed_factor)
        
        # Add noise occasionally
        if add_noise_flag and random.random() < 0.5:
            snr = random.uniform(*NOISE_SNR_RANGE)
            waveform = add_noise(waveform, snr)
        
        # Normalize volume
        waveform = normalize_volume(waveform)
        
        # Generate output filename
        stem = input_path.stem
        suffix = f"_sp{int(speed_factor*100)}"
        if add_noise_flag:
            suffix += "_aug"
        output_path = output_dir / f"{stem}{suffix}.wav"
        
        # Save
        torchaudio.save(str(output_path), waveform, SAMPLE_RATE)
        
        return output_path
        
    except Exception as e:
        return None


def augment_directory(input_dir: Path, output_dir: Path, max_files: int = None):
    """Augment all files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_files = get_audio_files(input_dir)
    if not input_files:
        return 0
    
    if max_files:
        input_files = input_files[:max_files]
    
    print(f"  Found {len(input_files)} files to augment")
    
    # Prepare tasks
    tasks = []
    for input_path in input_files:
        for speed in SPEED_FACTORS:
            tasks.append((input_path, output_dir, speed, ADD_NOISE))
    
    print(f"  Creating {len(tasks)} augmented files...")
    
    # Process in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
            result = future.result()
            if result:
                completed += 1
    
    return completed


def main():
    print("=" * 70)
    print("  COMPREHENSIVE DATA AUGMENTATION")
    print("=" * 70)
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Speed factors: {SPEED_FACTORS}")
    print(f"  Add noise: {ADD_NOISE} (SNR: {NOISE_SNR_RANGE} dB)")
    print(f"  Workers: {NUM_WORKERS}")
    print("=" * 70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU")
    
    total_created = 0
    
    for lang in LANGUAGES:
        lang_dir = BASE_DIR / lang
        if not lang_dir.exists():
            print(f"\n⚠️ {lang} directory not found, skipping...")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"  {lang.upper()}")
        print(f"{'=' * 70}")
        
        augmented_dir = lang_dir / "augmented"
        
        for subdir in SUBDIRS:
            source_dir = lang_dir / subdir
            
            if not source_dir.exists():
                continue
            
            # Skip if already augmented subdir
            if subdir == "augmented":
                continue
            
            # Check file count
            files = get_audio_files(source_dir)
            if not files:
                continue
            
            print(f"\n📁 {subdir}: {len(files)} source files")
            
            # Create augmented output directory
            output_subdir = augmented_dir / subdir
            
            # Check if already augmented
            existing = len(get_audio_files(output_subdir)) if output_subdir.exists() else 0
            if existing > len(files):
                print(f"  ✅ Already augmented ({existing} files)")
                continue
            
            # Augment
            created = augment_directory(source_dir, output_subdir)
            total_created += created
            print(f"  ✅ Created {created} augmented files")
    
    # Summary
    print("\n" + "=" * 70)
    print("  AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"  Total augmented files created: {total_created}")
    
    # Count all files
    print("\n  Final file counts:")
    for lang in LANGUAGES:
        lang_dir = BASE_DIR / lang
        if lang_dir.exists():
            total = len(get_audio_files(lang_dir))
            print(f"    {lang}: {total} files")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
