#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  HINDI DATA AUGMENTATION SCRIPT
  
  Purpose: Expand Hindi data through augmentation for balanced training
  
  Augmentation techniques:
  1. Speed perturbation (0.9x, 1.1x) 
  2. Pitch shifting (-2, +2 semitones)
  3. Light noise injection
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
    
    effects = [
        ["speed", str(speed)],
        ["rate", str(sr)]
    ]
    try:
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return augmented
    except:
        new_length = int(waveform.shape[-1] / speed)
        return F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear').squeeze(0)


def pitch_augment(waveform: torch.Tensor, sr: int, semitones: int) -> torch.Tensor:
    """Shift pitch by semitones"""
    if semitones == 0:
        return waveform
    
    effects = [
        ["pitch", str(semitones * 100)],
        ["rate", str(sr)]
    ]
    try:
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return augmented
    except:
        return waveform


def add_noise(waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """Add white noise at specified SNR"""
    signal_power = torch.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


def process_single_file(args):
    """Process a single audio file with augmentations"""
    input_path, output_dir, augment_types = args
    
    try:
        waveform, sr = torchaudio.load(str(input_path))
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        
        base_name = input_path.stem
        results = []
        
        for aug_type in augment_types:
            if aug_type == 'speed_slow':
                augmented = speed_augment(waveform, sr, 0.9)
                suffix = "_slow"
            elif aug_type == 'speed_fast':
                augmented = speed_augment(waveform, sr, 1.1)
                suffix = "_fast"
            elif aug_type == 'pitch_down':
                augmented = pitch_augment(waveform, sr, -2)
                suffix = "_pitchdown"
            elif aug_type == 'pitch_up':
                augmented = pitch_augment(waveform, sr, 2)
                suffix = "_pitchup"
            elif aug_type == 'noise_light':
                augmented = add_noise(waveform, snr_db=25.0)
                suffix = "_noise"
            else:
                continue
            
            output_path = output_dir / f"{base_name}{suffix}.wav"
            torchaudio.save(str(output_path), augmented, sr)
            results.append(output_path)
        
        return results
        
    except Exception as e:
        return []


def augment_dataset(input_dir: str, output_dir: str, num_workers: int = 8,
                   augment_types: list = None):
    """Augment entire dataset"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if augment_types is None:
        augment_types = ['speed_slow', 'speed_fast', 'noise_light']
    
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        audio_files.extend(list(input_path.rglob(ext)))
    
    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return 0
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎵 Found {len(audio_files)} audio files")
    print(f"🔧 Augmentation types: {augment_types}")
    print(f"📈 Expected output: {len(audio_files) * len(augment_types)} files")
    print(f"⏱️  Expansion factor: {len(augment_types)}x")
    print()
    
    args_list = [(f, output_path, augment_types) for f in audio_files]
    
    total_created = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args[0] 
                  for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Augmenting Hindi"):
            results = future.result()
            total_created += len(results)
    
    print(f"\n✅ Created {total_created} augmented files")
    return total_created


def main():
    parser = argparse.ArgumentParser(
        description="Augment Hindi audio data to expand dataset size"
    )
    parser.add_argument("--input_dir", type=str, 
                       default="/workspace/data/hindi",
                       help="Input directory with Hindi audio")
    parser.add_argument("--output_dir", type=str,
                       default="/workspace/data/hindi/augmented",
                       help="Output directory for augmented audio")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--expansion", type=str, default="3x",
                       choices=["2x", "3x", "5x"],
                       help="Expansion factor")
    
    args = parser.parse_args()
    
    if args.expansion == "2x":
        augment_types = ['speed_slow', 'speed_fast']
    elif args.expansion == "3x":
        augment_types = ['speed_slow', 'speed_fast', 'noise_light']
    elif args.expansion == "5x":
        augment_types = ['speed_slow', 'speed_fast', 'pitch_down', 
                        'pitch_up', 'noise_light']
    
    print("═" * 70)
    print("  HINDI DATA AUGMENTATION")
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
