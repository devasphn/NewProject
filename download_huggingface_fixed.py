#!/usr/bin/env python3
"""
Fixed HuggingFace Data Downloader - Batch Processing to Avoid OOM

The original script failed with std::bad_alloc because it tried to load
entire datasets into memory. This version:
1. Processes data in small batches
2. Uses streaming mode where possible
3. Saves audio files incrementally
4. Has better error handling

Run on RunPod: python download_huggingface_fixed.py
"""

import os
import sys
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")
BATCH_SIZE = 100  # Process this many samples at a time
MAX_WORKERS = 4

# Languages to download - USE FULL NAMES (not language codes!)
# Kathbath configs: 'hindi', 'telugu' (lowercase)
# IndicVoices configs: 'hindi', 'telugu' (lowercase)
LANGUAGES = {
    'hindi': 'hindi',    # NOT 'hi'
    'telugu': 'telugu'   # NOT 'te'
}


def check_hf_login():
    """Check HuggingFace login status."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✅ HuggingFace: Logged in")
            return True
        else:
            print("❌ HuggingFace: Not logged in")
            print("   Run: huggingface-cli login")
            return False
    except Exception as e:
        print(f"❌ HuggingFace check failed: {e}")
        return False


def download_kathbath_streaming(lang_name: str, lang_code: str):
    """
    Download Kathbath using streaming mode to avoid loading entire dataset.
    """
    from datasets import load_dataset
    
    output_dir = BASE_DIR / lang_name / "kathbath"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_files = list(output_dir.glob("*.wav"))
    if len(existing_files) > 10000:
        print(f"  ✅ Kathbath {lang_name} already has {len(existing_files)} files, skipping...")
        return len(existing_files)
    
    print(f"\n📥 Downloading Kathbath {lang_name} (streaming mode)...")
    
    try:
        # Use streaming to avoid loading entire dataset
        dataset = load_dataset(
            "ai4bharat/Kathbath",
            lang_code,
            split="train",
            streaming=True,
            trust_remote_code=False
        )
        
        saved_count = len(existing_files)
        errors = 0
        
        # Process in batches
        batch = []
        for i, sample in enumerate(tqdm(dataset, desc=f"Kathbath {lang_name}")):
            try:
                # Get audio data
                audio = sample['audio']
                audio_array = np.array(audio['array'], dtype=np.float32)
                sample_rate = audio['sampling_rate']
                
                # Save to file
                output_path = output_dir / f"kathbath_{lang_code}_{saved_count:06d}.wav"
                sf.write(output_path, audio_array, sample_rate)
                saved_count += 1
                
                # Memory cleanup every 1000 samples
                if saved_count % 1000 == 0:
                    gc.collect()
                    
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  ⚠️ Error on sample {i}: {e}")
                continue
        
        print(f"  ✅ Saved {saved_count} files ({errors} errors)")
        return saved_count
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 0


def download_kathbath_parquet(lang_name: str, lang_code: str):
    """
    Alternative: Download Kathbath by reading parquet files directly.
    This avoids the audio decoding issue.
    """
    from datasets import load_dataset
    import pyarrow.parquet as pq
    
    output_dir = BASE_DIR / lang_name / "kathbath"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_files = list(output_dir.glob("*.wav"))
    if len(existing_files) > 10000:
        print(f"  ✅ Kathbath {lang_name} already has {len(existing_files)} files, skipping...")
        return len(existing_files)
    
    print(f"\n📥 Downloading Kathbath {lang_name} (parquet mode)...")
    
    try:
        # First, just get the dataset info without loading audio
        dataset = load_dataset(
            "ai4bharat/Kathbath",
            lang_code,
            split="train",
            trust_remote_code=False
        )
        
        saved_count = len(existing_files)
        
        # Process in batches to avoid OOM
        total = len(dataset)
        print(f"  Total samples: {total}")
        
        for start_idx in tqdm(range(0, total, BATCH_SIZE), desc=f"Kathbath {lang_name}"):
            end_idx = min(start_idx + BATCH_SIZE, total)
            
            try:
                # Get batch
                batch = dataset.select(range(start_idx, end_idx))
                
                for i, sample in enumerate(batch):
                    try:
                        audio = sample['audio']
                        audio_array = np.array(audio['array'], dtype=np.float32)
                        sample_rate = audio['sampling_rate']
                        
                        output_path = output_dir / f"kathbath_{lang_code}_{saved_count:06d}.wav"
                        sf.write(output_path, audio_array, sample_rate)
                        saved_count += 1
                        
                    except Exception as e:
                        continue
                
                # Force garbage collection after each batch
                del batch
                gc.collect()
                
            except Exception as e:
                print(f"  ⚠️ Batch error at {start_idx}: {e}")
                continue
        
        print(f"  ✅ Saved {saved_count} files")
        return saved_count
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        
        # Try streaming as fallback
        print("  🔄 Trying streaming mode as fallback...")
        return download_kathbath_streaming(lang_name, lang_code)


def download_indicvoices_streaming(lang_name: str, lang_code: str):
    """
    Download IndicVoices using streaming mode.
    """
    from datasets import load_dataset
    
    output_dir = BASE_DIR / lang_name / "indicvoices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_files = list(output_dir.glob("*.wav"))
    if len(existing_files) > 10000:
        print(f"  ✅ IndicVoices {lang_name} already has {len(existing_files)} files, skipping...")
        return len(existing_files)
    
    print(f"\n📥 Downloading IndicVoices {lang_name} (streaming mode)...")
    
    try:
        # IndicVoices uses LOWERCASE language names: 'hindi', 'telugu'
        # NOT title case like 'Hindi' or 'Telugu'
        
        dataset = load_dataset(
            "ai4bharat/IndicVoices",
            lang_code,  # Already lowercase: 'hindi' or 'telugu'
            split="train",
            streaming=True,
            trust_remote_code=False
        )
        
        saved_count = len(existing_files)
        errors = 0
        
        for i, sample in enumerate(tqdm(dataset, desc=f"IndicVoices {lang_name}")):
            try:
                audio = sample['audio']
                audio_array = np.array(audio['array'], dtype=np.float32)
                sample_rate = audio['sampling_rate']
                
                output_path = output_dir / f"indicvoices_{lang_code}_{saved_count:06d}.wav"
                sf.write(output_path, audio_array, sample_rate)
                saved_count += 1
                
                if saved_count % 1000 == 0:
                    gc.collect()
                    
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  ⚠️ Error: {e}")
                continue
        
        print(f"  ✅ Saved {saved_count} files ({errors} errors)")
        return saved_count
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 0


def download_commonvoice(lang_name: str, lang_code: str):
    """
    Download Common Voice. Note: Needs to accept license on HF first.
    Common Voice uses ISO language codes: 'hi' for Hindi, 'te' for Telugu
    """
    from datasets import load_dataset
    
    output_dir = BASE_DIR / lang_name / "commonvoice"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_files = list(output_dir.glob("*.wav"))
    if len(existing_files) > 1000:
        print(f"  ✅ Common Voice {lang_name} already has {len(existing_files)} files, skipping...")
        return len(existing_files)
    
    print(f"\n📥 Downloading Common Voice {lang_name}...")
    
    # Common Voice uses ISO language codes
    cv_lang_map = {
        'hindi': 'hi',
        'telugu': 'te'
    }
    cv_code = cv_lang_map.get(lang_code, lang_code)
    
    # Try multiple dataset versions
    dataset_names = [
        "mozilla-foundation/common_voice_17_0",
        "mozilla-foundation/common_voice_16_1",
        "mozilla-foundation/common_voice_16_0",
        "mozilla-foundation/common_voice_13_0",
    ]
    
    for ds_name in dataset_names:
        try:
            print(f"  Trying {ds_name}...")
            dataset = load_dataset(
                ds_name,
                cv_code,
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            
            saved_count = len(existing_files)
            
            for i, sample in enumerate(tqdm(dataset, desc=f"CommonVoice {lang_name}")):
                try:
                    audio = sample['audio']
                    audio_array = np.array(audio['array'], dtype=np.float32)
                    sample_rate = audio['sampling_rate']
                    
                    output_path = output_dir / f"cv_{cv_code}_{saved_count:06d}.wav"
                    sf.write(output_path, audio_array, sample_rate)
                    saved_count += 1
                    
                    if saved_count % 1000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    continue
            
            print(f"  ✅ Saved {saved_count} files from {ds_name}")
            return saved_count
            
        except Exception as e:
            print(f"  ⚠️ {ds_name} failed: {str(e)[:50]}...")
            continue
    
    print(f"  ❌ All Common Voice versions failed")
    print("     Accept license at: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
    return 0


def count_audio_files(directory: Path) -> int:
    """Count audio files in directory."""
    if not directory.exists():
        return 0
    count = 0
    for ext in ['.wav', '.flac', '.mp3', '.ogg']:
        count += len(list(directory.rglob(f'*{ext}')))
    return count


def main():
    print("=" * 70)
    print("  HUGGINGFACE DATA DOWNLOADER (FIXED - BATCH PROCESSING)")
    print("=" * 70)
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 70)
    
    # Check login
    if not check_hf_login():
        print("\n⚠️ Please login first:")
        print("   huggingface-cli login")
        print("\nContinuing anyway (some datasets may fail)...")
    
    # Install required packages
    print("\n📦 Checking dependencies...")
    try:
        import soundfile
        print("  ✅ soundfile")
    except:
        os.system("pip install soundfile")
    
    try:
        import datasets
        print("  ✅ datasets")
    except:
        os.system("pip install datasets")
    
    results = {}
    
    # Download for each language
    for lang_name, lang_code in LANGUAGES.items():
        print(f"\n{'=' * 70}")
        print(f"  {lang_name.upper()}")
        print(f"{'=' * 70}")
        
        results[lang_name] = {}
        
        # Kathbath
        try:
            count = download_kathbath_streaming(lang_name, lang_code)
            results[lang_name]['kathbath'] = count
        except Exception as e:
            print(f"  ❌ Kathbath failed: {e}")
            results[lang_name]['kathbath'] = 0
        
        gc.collect()
        
        # IndicVoices
        try:
            count = download_indicvoices_streaming(lang_name, lang_code)
            results[lang_name]['indicvoices'] = count
        except Exception as e:
            print(f"  ❌ IndicVoices failed: {e}")
            results[lang_name]['indicvoices'] = 0
        
        gc.collect()
        
        # Common Voice (often needs auth)
        try:
            count = download_commonvoice(lang_name, lang_code)
            results[lang_name]['commonvoice'] = count
        except Exception as e:
            print(f"  ❌ Common Voice failed: {e}")
            results[lang_name]['commonvoice'] = 0
        
        gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    
    total_files = 0
    for lang_name, datasets in results.items():
        lang_total = sum(datasets.values())
        total_files += lang_total
        print(f"\n  {lang_name.upper()}:")
        for ds_name, count in datasets.items():
            status = "✅" if count > 0 else "❌"
            print(f"    {status} {ds_name}: {count} files")
        print(f"    Total: {lang_total} files")
    
    print(f"\n  GRAND TOTAL: {total_files} files")
    
    # Estimate hours (rough: 5-10 seconds per file average)
    estimated_hours = total_files * 7.5 / 3600  # 7.5 sec avg per file
    print(f"  Estimated: ~{estimated_hours:.0f} hours")
    
    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("  1. Check data status: python check_data_status.py")
    print("  2. Start training: python train_codec_production.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
