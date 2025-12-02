#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MASTER DATA DOWNLOAD SCRIPT
  
  Orchestrates ALL data downloads for the production codec:
  1. Checks what's already downloaded
  2. Downloads missing wget data
  3. Extracts cached HuggingFace parquet files (low-memory)
  4. Verifies final data status
  
  Run on RunPod: python download_all_data_master.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")
HF_CACHE = Path("/root/.cache/huggingface/datasets")

# Minimum requirements per language (hours)
MIN_HOURS = {
    'english': 500,
    'hindi': 500,
    'telugu': 200
}


def run_cmd(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=True, text=True, timeout=3600)
        return result.returncode == 0
    except:
        return False


def count_audio_files(directory: Path) -> int:
    """Count audio files in directory."""
    if not directory.exists():
        return 0
    
    count = 0
    for ext in ['.wav', '.flac', '.mp3', '.ogg']:
        result = subprocess.run(
            f'find {directory} -name "*{ext}" | wc -l',
            shell=True, capture_output=True, text=True
        )
        try:
            count += int(result.stdout.strip())
        except:
            pass
    return count


def estimate_hours(file_count: int, avg_duration_sec: float = 5.0) -> float:
    """Estimate hours from file count."""
    return file_count * avg_duration_sec / 3600


def check_data_status() -> dict:
    """Check current data status."""
    print("\n" + "═" * 70)
    print("  CHECKING CURRENT DATA STATUS")
    print("═" * 70)
    
    status = {}
    
    for lang in ['english', 'hindi', 'telugu']:
        lang_dir = BASE_DIR / lang
        files = count_audio_files(lang_dir)
        hours = estimate_hours(files)
        
        status[lang] = {
            'files': files,
            'hours': hours,
            'complete': hours >= MIN_HOURS[lang]
        }
        
        icon = "✅" if status[lang]['complete'] else "⚠️"
        print(f"  {icon} {lang}: {files} files (~{hours:.0f}h / {MIN_HOURS[lang]}h min)")
    
    return status


def download_wget_data():
    """Download data using wget (OpenSLR, Gramvaani)."""
    print("\n" + "═" * 70)
    print("  DOWNLOADING WGET DATA")
    print("═" * 70)
    
    # Check if bash script exists
    script_path = Path("/workspace/NewProject/download_6000h_data.sh")
    if not script_path.exists():
        print("  ⚠️ download_6000h_data.sh not found")
        return
    
    # Run the wget downloads
    print("  Running download_6000h_data.sh...")
    print("  (This may take a while...)")
    
    # Run in background and monitor
    result = subprocess.run(
        f"bash {script_path}",
        shell=True,
        cwd="/workspace/NewProject",
        timeout=7200  # 2 hour timeout
    )
    
    if result.returncode == 0:
        print("  ✅ wget downloads complete")
    else:
        print("  ⚠️ Some wget downloads may have failed")


def extract_hf_parquet_data():
    """Extract audio from cached HuggingFace parquet files."""
    print("\n" + "═" * 70)
    print("  EXTRACTING HUGGINGFACE CACHED DATA")
    print("═" * 70)
    
    import io
    import gc
    
    try:
        import pyarrow.parquet as pq
        import soundfile as sf
    except ImportError:
        print("  Installing dependencies...")
        os.system("pip install pyarrow soundfile")
        import pyarrow.parquet as pq
        import soundfile as sf
    
    if not HF_CACHE.exists():
        print("  ⚠️ No HuggingFace cache found")
        return
    
    # Find Kathbath cache
    kathbath_cache = None
    for subdir in HF_CACHE.iterdir():
        if 'kathbath' in subdir.name.lower():
            kathbath_cache = subdir
            break
    
    if not kathbath_cache:
        print("  ⚠️ No Kathbath cache found")
        return
    
    print(f"  Found cache: {kathbath_cache.name}")
    
    # Find parquet files for each language
    for lang in ['hindi', 'telugu']:
        print(f"\n  Processing {lang.upper()}...")
        
        output_dir = BASE_DIR / lang / 'kathbath'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing
        existing = count_audio_files(output_dir)
        if existing > 50000:
            print(f"    ✅ Already has {existing} files, skipping")
            continue
        
        # Find parquet files
        parquet_files = [f for f in kathbath_cache.rglob("*.parquet") 
                        if lang in str(f).lower()]
        
        if not parquet_files:
            print(f"    ⚠️ No parquet files found for {lang}")
            continue
        
        print(f"    Found {len(parquet_files)} parquet files")
        
        extracted = existing
        current_idx = existing
        
        for pf in tqdm(parquet_files, desc=f"Kathbath {lang}"):
            try:
                # Open parquet file
                parquet_file = pq.ParquetFile(pf)
                
                # Process each row group
                for rg_idx in range(parquet_file.num_row_groups):
                    try:
                        table = parquet_file.read_row_group(rg_idx)
                        
                        # Find audio column
                        audio_col = None
                        for col in ['audio', 'Audio', 'speech']:
                            if col in table.column_names:
                                audio_col = col
                                break
                        
                        if not audio_col:
                            continue
                        
                        audio_data = table.column(audio_col).to_pylist()
                        
                        for item in audio_data:
                            try:
                                if isinstance(item, dict):
                                    audio_bytes = item.get('bytes')
                                    sr = item.get('sampling_rate', 16000)
                                    
                                    if audio_bytes:
                                        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                                        out_path = output_dir / f"kb_{lang}_{current_idx:07d}.wav"
                                        sf.write(str(out_path), audio_array, sr)
                                        extracted += 1
                                        current_idx += 1
                            except:
                                continue
                        
                        del table, audio_data
                        gc.collect()
                        
                    except:
                        continue
                        
            except Exception as e:
                continue
            
            gc.collect()
        
        print(f"    ✅ Extracted {extracted - existing} new files (total: {extracted})")


def download_indicvoices_streaming():
    """Download IndicVoices using true streaming (1 sample at a time)."""
    print("\n" + "═" * 70)
    print("  DOWNLOADING INDICVOICES (STREAMING)")
    print("═" * 70)
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
    except ImportError:
        os.system("pip install datasets soundfile")
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
    
    for lang in ['hindi', 'telugu']:
        print(f"\n  Processing IndicVoices {lang.upper()}...")
        
        output_dir = BASE_DIR / lang / 'indicvoices'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        existing = count_audio_files(output_dir)
        if existing > 50000:
            print(f"    ✅ Already has {existing} files, skipping")
            continue
        
        try:
            # Use streaming mode
            dataset = load_dataset(
                "ai4bharat/IndicVoices",
                lang,  # lowercase: 'hindi' or 'telugu'
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            
            saved = existing
            errors = 0
            
            for i, sample in enumerate(tqdm(dataset, desc=f"IndicVoices {lang}")):
                try:
                    audio = sample['audio']
                    audio_array = np.array(audio['array'], dtype=np.float32)
                    sr = audio['sampling_rate']
                    
                    out_path = output_dir / f"iv_{lang}_{saved:07d}.wav"
                    sf.write(str(out_path), audio_array, sr)
                    saved += 1
                    
                    # Memory cleanup
                    if saved % 1000 == 0:
                        import gc
                        gc.collect()
                    
                    # Limit for safety
                    if saved > 500000:
                        break
                        
                except Exception as e:
                    errors += 1
                    if errors >= 100:
                        print(f"    ⚠️ Too many errors ({errors}), stopping")
                        break
                    continue
            
            print(f"    ✅ Saved {saved - existing} new files (total: {saved})")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:50]}")


def main():
    print("═" * 70)
    print("  MASTER DATA DOWNLOAD SCRIPT")
    print("═" * 70)
    print(f"  Base directory: {BASE_DIR}")
    print("═" * 70)
    
    # Step 1: Check current status
    status = check_data_status()
    
    total_hours = sum(s['hours'] for s in status.values())
    print(f"\n  Current total: ~{total_hours:.0f} hours")
    
    # Step 2: Download wget data if needed
    if status['english']['hours'] < 500 or status['hindi']['hours'] < 300:
        download_wget_data()
    else:
        print("\n  ✅ wget data already sufficient")
    
    # Step 3: Extract HF cached data
    extract_hf_parquet_data()
    
    # Step 4: Try streaming IndicVoices
    if status['hindi']['hours'] < 800 or status['telugu']['hours'] < 400:
        download_indicvoices_streaming()
    
    # Final status
    print("\n" + "═" * 70)
    print("  FINAL DATA STATUS")
    print("═" * 70)
    
    status = check_data_status()
    
    total_hours = sum(s['hours'] for s in status.values())
    total_files = sum(s['files'] for s in status.values())
    
    print(f"\n  📊 GRAND TOTAL: {total_files} files (~{total_hours:.0f} hours)")
    
    # Check if ready for training
    all_complete = all(s['complete'] for s in status.values())
    
    if all_complete:
        print("\n  ✅ ALL LANGUAGES HAVE SUFFICIENT DATA!")
        print("  ✅ Ready for augmentation and training!")
    else:
        print("\n  ⚠️ Some languages need more data")
    
    print("\n" + "═" * 70)
    print("  NEXT STEPS")
    print("═" * 70)
    print("  1. Augment Telugu: python augment_telugu_data.py")
    print("     OR Augment ALL: python augment_all_data.py")
    print("  2. Start training: python train_codec_production.py \\")
    print("       --data_dirs /workspace/data/english /workspace/data/hindi /workspace/data/telugu")
    print("═" * 70)


if __name__ == "__main__":
    main()
