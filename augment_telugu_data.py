#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  TELUGU DATA AUGMENTATION SCRIPT
  
  Purpose: Expand ~800h Telugu data to 2000h+ through augmentation
  
  Augmentation techniques:
  1. Speed perturbation (0.9x, 1.0x, 1.1x) → 3x data
  2. Pitch shifting (-2, 0, +2 semitones) → 3x data  
  3. Background noise injection → 2x data
  4. Reverb simulation → 2x data
  5. Combined augmentations → 5x+ total expansion
  
  Input: 800h original Telugu audio
  Output: 4000h+ augmented Telugu audio
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F


def speed_augment(waveform: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
    """Change playback speed without changing pitch"""
    if speed == 1.0:
        return waveform
    
    # Resample to change speed
    effects = [
        ["speed", str(speed)],
        ["rate", str(sr)]
    ]
    try:
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return augmented
    except:
        # Fallback: simple interpolation
        new_length = int(waveform.shape[-1] / speed)
        return F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear').squeeze(0)


def pitch_augment(waveform: torch.Tensor, sr: int, semitones: int) -> torch.Tensor:
    """Shift pitch by semitones"""
    if semitones == 0:
        return waveform
    
    effects = [
        ["pitch", str(semitones * 100)],  # cents
        ["rate", str(sr)]
    ]
    try:
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return augmented
    except:
        return waveform  # Return original if pitch shift fails


def add_noise(waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """Add white noise at specified SNR"""
    signal_power = torch.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


def add_reverb(waveform: torch.Tensor, sr: int, room_size: float = 0.5) -> torch.Tensor:
    """Add simple reverb effect"""
    try:
        effects = [
            ["reverb", str(int(room_size * 100))]
        ]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return augmented
    except:
        # Fallback: simple echo
        delay_samples = int(sr * 0.03 * room_size)  # 30ms max delay
        decay = 0.3 * room_size
        delayed = F.pad(waveform, (delay_samples, 0))[:, :waveform.shape[-1]]
        return waveform + delayed * decay


def process_single_file(args):
    """Process a single audio file with augmentations"""
    input_path, output_dir, augment_types = args
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(input_path))
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        
        stem = input_path.stem
        results = []
        
        # Generate augmented versions
        for aug_type in augment_types:
            if aug_type == 'speed_slow':
                aug_waveform = speed_augment(waveform, sr, 0.9)
                suffix = '_speed09'
            elif aug_type == 'speed_fast':
                aug_waveform = speed_augment(waveform, sr, 1.1)
                suffix = '_speed11'
            elif aug_type == 'pitch_down':
                aug_waveform = pitch_augment(waveform, sr, -2)
                suffix = '_pitch-2'
            elif aug_type == 'pitch_up':
                aug_waveform = pitch_augment(waveform, sr, 2)
                suffix = '_pitch+2'
            elif aug_type == 'noise_light':
                aug_waveform = add_noise(waveform, snr_db=25)
                suffix = '_noise25'
            elif aug_type == 'noise_medium':
                aug_waveform = add_noise(waveform, snr_db=15)
                suffix = '_noise15'
            elif aug_type == 'reverb_small':
                aug_waveform = add_reverb(waveform, sr, room_size=0.3)
                suffix = '_reverb03'
            elif aug_type == 'reverb_large':
                aug_waveform = add_reverb(waveform, sr, room_size=0.7)
                suffix = '_reverb07'
            else:
                continue
            
            # Normalize
            aug_waveform = aug_waveform / (aug_waveform.abs().max() + 1e-8) * 0.95
            
            # Save
            output_path = Path(output_dir) / f"{stem}{suffix}.wav"
            torchaudio.save(str(output_path), aug_waveform, sr)
            results.append(str(output_path))
        
        return results
        
    except Exception as e:
        return []


def augment_dataset(input_dir: str, output_dir: str, num_workers: int = 8,
                   augment_types: list = None):
    """Augment all audio files in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default augmentation types (5x expansion)
    if augment_types is None:
        augment_types = [
            'speed_slow',    # 0.9x speed
            'speed_fast',    # 1.1x speed
            'pitch_down',    # -2 semitones
            'pitch_up',      # +2 semitones
            'noise_light',   # SNR 25dB
        ]
    
    # Find all audio files
    audio_files = list(input_path.rglob("*.wav")) + \
                  list(input_path.rglob("*.flac")) + \
                  list(input_path.rglob("*.mp3"))
    
    print(f"\n📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎵 Found {len(audio_files)} audio files")
    print(f"🔧 Augmentation types: {augment_types}")
    print(f"📈 Expected output: {len(audio_files) * len(augment_types)} files")
    print(f"⏱️  Expansion factor: {len(augment_types)}x")
    print()
    
    # Prepare arguments for parallel processing
    args_list = [(f, output_dir, augment_types) for f in audio_files]
    
    # Process with progress bar
    total_created = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args[0] 
                  for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Augmenting"):
            results = future.result()
            total_created += len(results)
    
    print(f"\n✅ Created {total_created} augmented files")
    return total_created


def main():
    parser = argparse.ArgumentParser(
        description="Augment Telugu audio data to expand dataset size"
    )
    parser.add_argument("--input_dir", type=str, 
                       default="/workspace/data/telugu",
                       help="Input directory with Telugu audio")
    parser.add_argument("--output_dir", type=str,
                       default="/workspace/data/telugu/augmented",
                       help="Output directory for augmented audio")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--expansion", type=str, default="5x",
                       choices=["3x", "5x", "8x"],
                       help="Expansion factor")
    
    args = parser.parse_args()
    
    # Set augmentation types based on expansion factor
    if args.expansion == "3x":
        augment_types = ['speed_slow', 'speed_fast', 'noise_light']
    elif args.expansion == "5x":
        augment_types = ['speed_slow', 'speed_fast', 'pitch_down', 
                        'pitch_up', 'noise_light']
    elif args.expansion == "8x":
        augment_types = ['speed_slow', 'speed_fast', 'pitch_down', 
                        'pitch_up', 'noise_light', 'noise_medium',
                        'reverb_small', 'reverb_large']
    
    print("═" * 70)
    print("  TELUGU DATA AUGMENTATION")
    print("═" * 70)
    print(f"  Expansion: {args.expansion}")
    print(f"  Workers: {args.workers}")
    print("═" * 70)
    
    augment_dataset(
        args.input_dir,
        args.output_dir,
        num_workers=args.workers,
        augment_types=augment_types
    )
    
    # Calculate hours
    output_path = Path(args.output_dir)
    total_duration = 0
    for f in output_path.rglob("*.wav"):
        try:
            info = torchaudio.info(str(f))
            total_duration += info.num_frames / info.sample_rate
        except:
            pass
    
    hours = total_duration / 3600
    print(f"\n📊 Total augmented duration: {hours:.1f} hours")
    print("═" * 70)


if __name__ == "__main__":
    main()
