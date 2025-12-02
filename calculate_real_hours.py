#!/usr/bin/env python3
"""
Calculate REAL audio hours by actually reading file durations.
Much more accurate than estimating from file counts.
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

DATA_DIR = Path("/workspace/data")

def get_duration_ffprobe(filepath):
    """Get duration using ffprobe (fast and accurate)."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(filepath)],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except:
        return 0.0

def sample_directory_duration(directory, sample_size=100):
    """Sample files to estimate average duration, then extrapolate."""
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3', '*.m4a']:
        audio_files.extend(list(Path(directory).rglob(ext)))
    
    if not audio_files:
        return 0, 0, 0
    
    total_files = len(audio_files)
    
    # Sample random files
    sample = random.sample(audio_files, min(sample_size, total_files))
    
    durations = []
    for f in sample:
        dur = get_duration_ffprobe(f)
        if dur > 0:
            durations.append(dur)
    
    if not durations:
        return total_files, 0, 0
    
    avg_duration = sum(durations) / len(durations)
    estimated_total_seconds = avg_duration * total_files
    estimated_hours = estimated_total_seconds / 3600
    
    return total_files, avg_duration, estimated_hours

def main():
    print("=" * 70)
    print("  CALCULATING REAL AUDIO HOURS (Sampling Method)")
    print("=" * 70)
    print("  Sampling 100 files from each directory to estimate duration...")
    print("=" * 70)
    print()
    
    results = {}
    
    for lang in ['english', 'hindi', 'telugu']:
        lang_dir = DATA_DIR / lang
        if not lang_dir.exists():
            continue
        
        print(f"📁 {lang.upper()}")
        print("-" * 50)
        
        lang_total_files = 0
        lang_total_hours = 0
        
        # Check subdirectories
        subdirs = [d for d in lang_dir.iterdir() if d.is_dir()]
        if not subdirs:
            subdirs = [lang_dir]
        
        for subdir in sorted(subdirs):
            files, avg_dur, hours = sample_directory_duration(subdir)
            if files > 0:
                print(f"  {subdir.name:20s} | {files:8d} files | avg {avg_dur:5.1f}s | ~{hours:7.1f}h")
                lang_total_files += files
                lang_total_hours += hours
        
        print(f"  {'TOTAL':20s} | {lang_total_files:8d} files |           | ~{lang_total_hours:7.1f}h")
        print()
        
        results[lang] = {
            'files': lang_total_files,
            'hours': lang_total_hours
        }
    
    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    total_hours = 0
    for lang, data in results.items():
        print(f"  {lang.upper():10s}: ~{data['hours']:7.1f} hours ({data['files']:,} files)")
        total_hours += data['hours']
    
    print(f"  {'TOTAL':10s}: ~{total_hours:7.1f} hours")
    print()
    
    # Recommendations
    print("=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    
    telugu_hours = results.get('telugu', {}).get('hours', 0)
    hindi_hours = results.get('hindi', {}).get('hours', 0)
    english_hours = results.get('english', {}).get('hours', 0)
    
    print()
    if telugu_hours < 200:
        print(f"  ⚠️  Telugu ({telugu_hours:.0f}h) is low - recommend 5x augmentation → ~{telugu_hours*5:.0f}h")
        print("      Command: python augment_telugu_data.py")
    else:
        print(f"  ✅ Telugu ({telugu_hours:.0f}h) is good!")
    
    print()
    if total_hours < 1000:
        print("  ⚠️  Total data is low - recommend augmentation for all languages")
        print("      Command: python augment_all_data.py")
    elif total_hours < 2000:
        print("  ✅ Data is sufficient for training - augmentation optional")
    else:
        print("  ✅ Data is excellent! Ready for training")
    
    print()
    print("=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    if telugu_hours < english_hours / 5:
        print("  1. Augment Telugu: python augment_telugu_data.py")
        print("  2. Then train:     python train_codec_production.py")
    else:
        print("  1. Train directly: python train_codec_production.py \\")
        print("       --data_dirs /workspace/data/english /workspace/data/hindi /workspace/data/telugu")
    print("=" * 70)

if __name__ == "__main__":
    main()
