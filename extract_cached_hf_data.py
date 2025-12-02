#!/usr/bin/env python3
"""
Extract Audio from Cached HuggingFace Parquet Files

You have 33GB of Kathbath data cached but the audio extraction failed.
This script extracts audio directly from the cached parquet files.

Run on RunPod: python extract_cached_hf_data.py
"""

import os
import gc
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")
HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"
BATCH_SIZE = 50  # Small batch to avoid OOM


def extract_kathbath_from_cache():
    """Extract Kathbath audio from cached parquet files."""
    import pyarrow.parquet as pq
    import soundfile as sf
    
    print("\n" + "=" * 70)
    print("  EXTRACTING KATHBATH FROM CACHE")
    print("=" * 70)
    
    # Find cached Kathbath data
    kathbath_cache = None
    for subdir in HF_CACHE.iterdir():
        if 'kathbath' in subdir.name.lower():
            kathbath_cache = subdir
            break
    
    if not kathbath_cache:
        print("❌ Kathbath cache not found!")
        return
    
    print(f"📁 Found cache: {kathbath_cache}")
    
    # Find parquet files for each language
    for lang in ['hindi', 'telugu']:
        print(f"\n{'─' * 50}")
        print(f"  Processing {lang.upper()}")
        print(f"{'─' * 50}")
        
        output_dir = BASE_DIR / lang / "kathbath"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing files
        existing = len(list(output_dir.glob("*.wav")))
        if existing > 50000:
            print(f"  ✅ Already has {existing} files, skipping...")
            continue
        
        # Find parquet files for this language
        parquet_files = list(kathbath_cache.rglob(f"*{lang}*.parquet"))
        
        if not parquet_files:
            # Try alternative paths
            parquet_files = list(kathbath_cache.rglob(f"*train*.parquet"))
            parquet_files = [f for f in parquet_files if lang in str(f)]
        
        if not parquet_files:
            print(f"  ⚠️ No parquet files found for {lang}")
            # List what's in the cache
            print(f"  Available files: {list(kathbath_cache.rglob('*.parquet'))[:5]}")
            continue
        
        print(f"  Found {len(parquet_files)} parquet files")
        
        saved_count = existing
        errors = 0
        
        for pq_file in tqdm(parquet_files, desc=f"Kathbath {lang}"):
            try:
                # Read parquet file
                table = pq.read_table(pq_file)
                df = table.to_pandas()
                
                # Process each row
                for idx, row in df.iterrows():
                    try:
                        audio_data = row.get('audio', {})
                        
                        if isinstance(audio_data, dict):
                            audio_array = audio_data.get('array', audio_data.get('bytes'))
                            sample_rate = audio_data.get('sampling_rate', 16000)
                        else:
                            continue
                        
                        if audio_array is None:
                            continue
                        
                        # Convert to numpy if needed
                        if isinstance(audio_array, bytes):
                            # Decode bytes to audio
                            import io
                            audio_array, sample_rate = sf.read(io.BytesIO(audio_array))
                        elif isinstance(audio_array, list):
                            audio_array = np.array(audio_array, dtype=np.float32)
                        
                        # Save audio
                        output_path = output_dir / f"kathbath_{lang}_{saved_count:06d}.wav"
                        sf.write(output_path, audio_array, sample_rate)
                        saved_count += 1
                        
                    except Exception as e:
                        errors += 1
                        if errors < 5:
                            print(f"  ⚠️ Row error: {e}")
                        continue
                
                # Cleanup
                del table, df
                gc.collect()
                
            except Exception as e:
                print(f"  ⚠️ File error {pq_file.name}: {e}")
                continue
        
        print(f"  ✅ Extracted {saved_count - existing} files ({errors} errors)")
        print(f"  Total: {saved_count} files")


def extract_using_datasets_lib():
    """Alternative: Use datasets library with batch processing."""
    from datasets import load_dataset
    import soundfile as sf
    
    print("\n" + "=" * 70)
    print("  EXTRACTING USING DATASETS LIBRARY (BATCH MODE)")
    print("=" * 70)
    
    for lang in ['hindi', 'telugu']:
        print(f"\n{'─' * 50}")
        print(f"  Processing Kathbath {lang.upper()}")
        print(f"{'─' * 50}")
        
        output_dir = BASE_DIR / lang / "kathbath"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        existing = len(list(output_dir.glob("*.wav")))
        if existing > 50000:
            print(f"  ✅ Already has {existing} files, skipping...")
            continue
        
        try:
            # Load dataset - it should use cached data
            print(f"  Loading dataset (using cache)...")
            dataset = load_dataset(
                "ai4bharat/Kathbath",
                lang,  # 'hindi' or 'telugu' - LOWERCASE!
                split="train",
                trust_remote_code=False
            )
            
            total = len(dataset)
            print(f"  Total samples: {total}")
            
            saved_count = existing
            
            # Process in small batches
            for start_idx in tqdm(range(0, total, BATCH_SIZE), desc=f"Kathbath {lang}"):
                end_idx = min(start_idx + BATCH_SIZE, total)
                
                try:
                    # Select batch
                    batch_indices = list(range(start_idx, end_idx))
                    batch = dataset.select(batch_indices)
                    
                    for sample in batch:
                        try:
                            audio = sample['audio']
                            audio_array = np.array(audio['array'], dtype=np.float32)
                            sample_rate = audio['sampling_rate']
                            
                            output_path = output_dir / f"kathbath_{lang}_{saved_count:06d}.wav"
                            sf.write(output_path, audio_array, sample_rate)
                            saved_count += 1
                            
                        except Exception as e:
                            continue
                    
                    # Force cleanup after each batch
                    del batch
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ⚠️ Batch error at {start_idx}: {e}")
                    gc.collect()
                    continue
            
            print(f"  ✅ Extracted {saved_count - existing} files")
            
            # Cleanup
            del dataset
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def main():
    print("=" * 70)
    print("  CACHED HUGGINGFACE DATA EXTRACTOR")
    print("=" * 70)
    
    # Check cache
    if HF_CACHE.exists():
        print(f"\n📁 HuggingFace cache: {HF_CACHE}")
        
        # List cached datasets
        for subdir in HF_CACHE.iterdir():
            if subdir.is_dir():
                try:
                    import subprocess
                    result = subprocess.run(['du', '-sh', str(subdir)], 
                                          capture_output=True, text=True, timeout=10)
                    size = result.stdout.split()[0] if result.stdout else "?"
                    print(f"  {subdir.name}: {size}")
                except:
                    print(f"  {subdir.name}")
    else:
        print("❌ No HuggingFace cache found!")
        return
    
    # Install dependencies
    print("\n📦 Checking dependencies...")
    try:
        import soundfile
        print("  ✅ soundfile")
    except:
        os.system("pip install soundfile")
    
    try:
        import pyarrow
        print("  ✅ pyarrow")
    except:
        os.system("pip install pyarrow")
    
    # Try datasets library method first (more reliable)
    print("\n🔄 Method 1: Using datasets library with batch processing...")
    try:
        extract_using_datasets_lib()
    except Exception as e:
        print(f"  ❌ Method 1 failed: {e}")
        print("\n🔄 Method 2: Direct parquet extraction...")
        try:
            extract_kathbath_from_cache()
        except Exception as e:
            print(f"  ❌ Method 2 failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  EXTRACTION SUMMARY")
    print("=" * 70)
    
    for lang in ['hindi', 'telugu']:
        output_dir = BASE_DIR / lang / "kathbath"
        count = len(list(output_dir.glob("*.wav"))) if output_dir.exists() else 0
        status = "✅" if count > 10000 else "⚠️" if count > 0 else "❌"
        print(f"  {status} {lang}/kathbath: {count} files")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
