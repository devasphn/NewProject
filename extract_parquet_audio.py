#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  LOW-MEMORY PARQUET AUDIO EXTRACTOR
  
  Extracts audio from cached HuggingFace parquet files WITHOUT using the
  datasets library's audio decoding (which causes std::bad_alloc).
  
  Uses pyarrow to read row-groups one at a time and decodes audio bytes
  directly with soundfile.
  
  Run on RunPod: python extract_parquet_audio.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import io
import gc
import sys
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")
HF_CACHE = Path("/root/.cache/huggingface/datasets")

# Target datasets and their output locations
DATASETS = {
    'kathbath': {
        'hindi': BASE_DIR / 'hindi' / 'kathbath',
        'telugu': BASE_DIR / 'telugu' / 'kathbath',
    },
    'indicvoices': {
        'hindi': BASE_DIR / 'hindi' / 'indicvoices',
        'telugu': BASE_DIR / 'telugu' / 'indicvoices',
    }
}


def find_parquet_files(cache_dir: Path, dataset_name: str) -> dict:
    """Find all parquet files for a dataset in the cache."""
    results = {}
    
    # Search for dataset directory
    for subdir in cache_dir.iterdir():
        if dataset_name.lower() in subdir.name.lower():
            print(f"  Found cache: {subdir.name}")
            
            # Find all parquet files
            parquet_files = list(subdir.rglob("*.parquet"))
            print(f"  Total parquet files: {len(parquet_files)}")
            
            # Group by language
            for pf in parquet_files:
                path_str = str(pf).lower()
                
                for lang in ['hindi', 'telugu', 'hi', 'te']:
                    if lang in path_str or f"/{lang}/" in path_str:
                        # Normalize language name
                        lang_key = 'hindi' if lang in ['hindi', 'hi'] else 'telugu'
                        
                        if lang_key not in results:
                            results[lang_key] = []
                        results[lang_key].append(pf)
                        break
            
            break
    
    return results


def extract_audio_from_parquet(parquet_path: Path, output_dir: Path, 
                               start_idx: int = 0) -> int:
    """
    Extract audio from a single parquet file using pyarrow.
    Returns number of files extracted.
    """
    import pyarrow.parquet as pq
    import soundfile as sf
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open parquet file
        pf = pq.ParquetFile(parquet_path)
        num_row_groups = pf.num_row_groups
        
        extracted = 0
        errors = 0
        current_idx = start_idx
        
        # Process each row group separately (low memory)
        for rg_idx in range(num_row_groups):
            try:
                # Read just this row group
                table = pf.read_row_group(rg_idx)
                
                # Get audio column
                audio_col = None
                for col_name in ['audio', 'Audio', 'speech', 'Speech']:
                    if col_name in table.column_names:
                        audio_col = col_name
                        break
                
                if audio_col is None:
                    print(f"    ⚠️ No audio column found in {parquet_path.name}")
                    continue
                
                # Convert to Python objects
                audio_data = table.column(audio_col).to_pylist()
                
                for item in audio_data:
                    try:
                        audio_bytes = None
                        sample_rate = 16000
                        
                        # Handle different data formats
                        if isinstance(item, dict):
                            # Format: {'bytes': b'...', 'sampling_rate': 16000}
                            audio_bytes = item.get('bytes') or item.get('array')
                            sample_rate = item.get('sampling_rate', 16000)
                        elif isinstance(item, bytes):
                            audio_bytes = item
                        elif isinstance(item, (list, np.ndarray)):
                            # Already decoded array
                            audio_array = np.array(item, dtype=np.float32)
                            output_path = output_dir / f"audio_{current_idx:07d}.wav"
                            sf.write(str(output_path), audio_array, sample_rate)
                            extracted += 1
                            current_idx += 1
                            continue
                        
                        if audio_bytes is None:
                            continue
                        
                        # Decode audio bytes
                        if isinstance(audio_bytes, bytes):
                            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                        else:
                            audio_array = np.array(audio_bytes, dtype=np.float32)
                            sr = sample_rate
                        
                        # Save to WAV
                        output_path = output_dir / f"audio_{current_idx:07d}.wav"
                        sf.write(str(output_path), audio_array, sr)
                        extracted += 1
                        current_idx += 1
                        
                    except Exception as e:
                        errors += 1
                        if errors <= 3:
                            print(f"    ⚠️ Item error: {str(e)[:50]}")
                        continue
                
                # Free memory
                del table, audio_data
                gc.collect()
                
            except Exception as e:
                print(f"    ⚠️ Row group {rg_idx} error: {str(e)[:50]}")
                continue
        
        return extracted, current_idx
        
    except Exception as e:
        print(f"  ❌ File error: {e}")
        return 0, start_idx


def extract_dataset(dataset_name: str, language: str, output_dir: Path):
    """Extract all audio for a dataset/language combination."""
    print(f"\n{'─' * 60}")
    print(f"  Extracting {dataset_name} - {language.upper()}")
    print(f"{'─' * 60}")
    
    # Check existing files
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.wav")))
    
    if existing > 50000:
        print(f"  ✅ Already has {existing} files, skipping...")
        return existing
    
    # Find parquet files
    parquet_files = find_parquet_files(HF_CACHE, dataset_name)
    
    if language not in parquet_files:
        print(f"  ⚠️ No parquet files found for {language}")
        return 0
    
    files = parquet_files[language]
    print(f"  Found {len(files)} parquet files for {language}")
    
    total_extracted = existing
    current_idx = existing
    
    for pf in tqdm(files, desc=f"{dataset_name}/{language}"):
        extracted, current_idx = extract_audio_from_parquet(pf, output_dir, current_idx)
        total_extracted += extracted
        
        # Progress update every 10 files
        if total_extracted % 10000 == 0:
            print(f"    Progress: {total_extracted} files extracted")
        
        gc.collect()
    
    print(f"  ✅ Total extracted: {total_extracted} files")
    return total_extracted


def main():
    print("═" * 70)
    print("  LOW-MEMORY PARQUET AUDIO EXTRACTOR")
    print("═" * 70)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import pyarrow.parquet as pq
        print("  ✅ pyarrow")
    except ImportError:
        print("  ❌ pyarrow not found, installing...")
        os.system("pip install pyarrow")
        import pyarrow.parquet as pq
    
    try:
        import soundfile as sf
        print("  ✅ soundfile")
    except ImportError:
        print("  ❌ soundfile not found, installing...")
        os.system("pip install soundfile")
        import soundfile as sf
    
    # Check cache
    print(f"\n📁 HuggingFace cache: {HF_CACHE}")
    if not HF_CACHE.exists():
        print("  ❌ Cache directory not found!")
        return
    
    # List cached datasets
    print("\n📂 Cached datasets:")
    for subdir in HF_CACHE.iterdir():
        if subdir.is_dir():
            try:
                import subprocess
                result = subprocess.run(['du', '-sh', str(subdir)], 
                                      capture_output=True, text=True, timeout=10)
                size = result.stdout.split()[0] if result.stdout else "?"
                print(f"    {subdir.name}: {size}")
            except:
                print(f"    {subdir.name}")
    
    # Extract each dataset
    results = {}
    
    for dataset_name, languages in DATASETS.items():
        results[dataset_name] = {}
        
        for lang, output_dir in languages.items():
            count = extract_dataset(dataset_name, lang, output_dir)
            results[dataset_name][lang] = count
    
    # Summary
    print("\n" + "═" * 70)
    print("  EXTRACTION SUMMARY")
    print("═" * 70)
    
    total = 0
    for dataset_name, languages in results.items():
        print(f"\n  {dataset_name}:")
        for lang, count in languages.items():
            status = "✅" if count > 0 else "❌"
            print(f"    {status} {lang}: {count} files")
            total += count
    
    print(f"\n  TOTAL: {total} files")
    
    # Estimate hours (average ~5 seconds per file)
    hours = total * 5 / 3600
    print(f"  Estimated: ~{hours:.0f} hours")
    
    print("\n" + "═" * 70)
    print("  NEXT STEPS")
    print("═" * 70)
    print("  1. Check data: python check_data_status.py")
    print("  2. Augment: python augment_telugu_data.py (for Telugu)")
    print("           OR python augment_all_data.py (for ALL languages)")
    print("  3. Train: python train_codec_production.py")
    print("═" * 70)


if __name__ == "__main__":
    main()
