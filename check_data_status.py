#!/usr/bin/env python3
"""
Data Status Checker - Check what's downloaded and what's missing
Run on RunPod: python check_data_status.py
"""

import os
import subprocess
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.opus'}

# Expected data sources and approximate hours
EXPECTED_DATA = {
    'english': {
        'librispeech': {'expected_hours': 960, 'source': 'OpenSLR 12', 'type': 'wget'},
        'librilight': {'expected_hours': 600, 'source': 'OpenSLR (subset)', 'type': 'wget'},
    },
    'hindi': {
        'gramvaani': {'expected_hours': 1100, 'source': 'OpenSLR 118', 'type': 'wget'},
        'openslr103': {'expected_hours': 95, 'source': 'OpenSLR 103', 'type': 'wget'},
        'kathbath': {'expected_hours': 140, 'source': 'HuggingFace', 'type': 'hf'},
        'indicvoices': {'expected_hours': 800, 'source': 'HuggingFace', 'type': 'hf'},
        'commonvoice': {'expected_hours': 20, 'source': 'HuggingFace', 'type': 'hf'},
    },
    'telugu': {
        'openslr66': {'expected_hours': 10, 'source': 'OpenSLR 66', 'type': 'wget'},
        'kathbath': {'expected_hours': 140, 'source': 'HuggingFace', 'type': 'hf'},
        'indicvoices': {'expected_hours': 300, 'source': 'HuggingFace', 'type': 'hf'},
        'commonvoice': {'expected_hours': 20, 'source': 'HuggingFace', 'type': 'hf'},
    }
}


def count_audio_files(directory: Path) -> tuple:
    """Count audio files and estimate total duration."""
    if not directory.exists():
        return 0, 0, 0
    
    total_files = 0
    total_size = 0
    
    for ext in AUDIO_EXTENSIONS:
        try:
            result = subprocess.run(
                ['find', str(directory), '-type', 'f', '-name', f'*{ext}'],
                capture_output=True, text=True, timeout=60
            )
            files = [f for f in result.stdout.strip().split('\n') if f]
            total_files += len(files)
            
            # Get size of files
            for f in files[:100]:  # Sample first 100 for size estimation
                try:
                    total_size += os.path.getsize(f)
                except:
                    pass
        except:
            pass
    
    # Estimate hours (16kHz mono = ~115KB/min, ~7MB/hour)
    estimated_hours = (total_size / (7 * 1024 * 1024)) * (total_files / min(total_files, 100)) if total_files > 0 else 0
    
    return total_files, total_size, estimated_hours


def get_disk_usage(directory: Path) -> str:
    """Get disk usage of directory."""
    if not directory.exists():
        return "0"
    try:
        result = subprocess.run(
            ['du', '-sh', str(directory)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.split()[0] if result.stdout else "0"
    except:
        return "?"


def check_hf_login() -> bool:
    """Check if logged into HuggingFace."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        return token is not None
    except:
        return False


def main():
    print("=" * 80)
    print("  DATA STATUS CHECKER")
    print("=" * 80)
    
    # Check disk space
    print("\n📊 DISK SPACE:")
    try:
        result = subprocess.run(['df', '-h', '/workspace'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("  Could not check disk space")
    
    # Check HuggingFace login
    hf_logged_in = check_hf_login()
    print(f"🔐 HuggingFace Login: {'✅ Logged in' if hf_logged_in else '❌ Not logged in'}")
    if not hf_logged_in:
        print("   Run: huggingface-cli login")
    
    # Results storage
    results = defaultdict(dict)
    
    print("\n" + "=" * 80)
    print("  CHECKING DOWNLOADED DATA")
    print("=" * 80)
    
    total_hours_downloaded = 0
    total_hours_expected = 0
    
    for lang, sources in EXPECTED_DATA.items():
        lang_dir = BASE_DIR / lang
        print(f"\n{'─' * 40}")
        print(f"  {lang.upper()}")
        print(f"{'─' * 40}")
        
        lang_hours = 0
        
        for source_name, info in sources.items():
            source_dir = lang_dir / source_name
            
            # Check alternative paths
            alt_paths = [
                source_dir,
                lang_dir / source_name.replace('openslr', 'openslr'),
                lang_dir / 'openslr' / source_name,
            ]
            
            found_dir = None
            for p in alt_paths:
                if p.exists():
                    found_dir = p
                    break
            
            if found_dir:
                files, size, hours = count_audio_files(found_dir)
                disk_usage = get_disk_usage(found_dir)
                
                status = "✅" if files > 0 else "⚠️ EMPTY"
                expected = info['expected_hours']
                
                # Color coding based on completeness
                if hours >= expected * 0.8:
                    completeness = "✅ COMPLETE"
                elif hours >= expected * 0.3:
                    completeness = "⚠️ PARTIAL"
                elif files > 0:
                    completeness = "⚠️ LOW"
                else:
                    completeness = "❌ EMPTY"
                
                print(f"  {source_name:15} | {files:>8} files | {disk_usage:>8} | ~{hours:>6.1f}h / {expected}h | {completeness}")
                
                lang_hours += hours
                total_hours_downloaded += hours
                
                results[lang][source_name] = {
                    'status': 'downloaded',
                    'files': files,
                    'hours': hours,
                    'expected': expected,
                    'path': str(found_dir)
                }
            else:
                expected = info['expected_hours']
                download_type = info['type']
                
                print(f"  {source_name:15} | {'❌ NOT FOUND':>8} | {'-':>8} | ~{0:>6.1f}h / {expected}h | ❌ MISSING ({download_type})")
                
                results[lang][source_name] = {
                    'status': 'missing',
                    'type': download_type,
                    'expected': expected
                }
            
            total_hours_expected += info['expected_hours']
        
        print(f"  {'─' * 60}")
        print(f"  {lang.upper()} TOTAL: ~{lang_hours:.1f} hours")
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Total Downloaded: ~{total_hours_downloaded:.1f} hours")
    print(f"  Total Expected:   ~{total_hours_expected:.1f} hours")
    print(f"  Completion:       {total_hours_downloaded/total_hours_expected*100:.1f}%")
    
    # Missing data
    print("\n" + "=" * 80)
    print("  MISSING DATA & FIX COMMANDS")
    print("=" * 80)
    
    missing_wget = []
    missing_hf = []
    
    for lang, sources in results.items():
        for source_name, info in sources.items():
            if info['status'] == 'missing':
                if info['type'] == 'wget':
                    missing_wget.append((lang, source_name, info['expected']))
                else:
                    missing_hf.append((lang, source_name, info['expected']))
    
    if missing_wget:
        print("\n📥 WGET DOWNLOADS (run download_6000h_data.sh again or manually):")
        for lang, source, hours in missing_wget:
            print(f"  - {lang}/{source}: ~{hours}h")
            
        print("\n  Commands to retry wget downloads:")
        print("  bash download_6000h_data.sh")
    
    if missing_hf:
        print("\n📥 HUGGINGFACE DOWNLOADS (need batch processing):")
        for lang, source, hours in missing_hf:
            print(f"  - {lang}/{source}: ~{hours}h")
        
        print("\n  ⚠️ The std::bad_alloc error means OOM when loading full dataset")
        print("  ✅ Solution: Use the fixed batch download script:")
        print("  python download_huggingface_fixed.py")
    
    # Check for parquet files (partially downloaded HF data)
    print("\n" + "=" * 80)
    print("  CHECKING FOR CACHED HUGGINGFACE DATA")
    print("=" * 80)
    
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    if hf_cache.exists():
        print(f"\n  HuggingFace cache: {hf_cache}")
        try:
            result = subprocess.run(['du', '-sh', str(hf_cache)], capture_output=True, text=True)
            print(f"  Cache size: {result.stdout.strip()}")
            
            # Check for specific datasets
            for ds in ['ai4bharat___kathbath', 'ai4bharat___indic_voices', 'mozilla-foundation___common_voice']:
                ds_path = hf_cache / ds
                if ds_path.exists():
                    result = subprocess.run(['du', '-sh', str(ds_path)], capture_output=True, text=True)
                    print(f"    {ds}: {result.stdout.split()[0] if result.stdout else 'exists'}")
        except:
            pass
    
    print("\n" + "=" * 80)
    print("  RECOMMENDED NEXT STEPS")
    print("=" * 80)
    
    if total_hours_downloaded >= 2000:
        print("\n  ✅ You have enough data to start training!")
        print("     Minimum for production: 1000h per language (3000h total)")
        print(f"     Current: ~{total_hours_downloaded:.0f}h")
    else:
        print(f"\n  ⚠️ Need more data. Current: ~{total_hours_downloaded:.0f}h, Target: 3000h")
    
    print("\n  1. Run the fixed HuggingFace downloader (processes in batches):")
    print("     python download_huggingface_fixed.py")
    print("\n  2. If Gramvaani 1000h failed, download manually:")
    print("     wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_3h.tar.gz")
    print("     wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Train_100h.tar.gz")
    print("\n  3. Start training with available data:")
    print("     python train_codec_production.py --data_dirs /workspace/data/english /workspace/data/hindi /workspace/data/telugu")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
