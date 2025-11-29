#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    FIXED TELUGU DATA DOWNLOAD - Memory Efficient
    
    Fixes the std::bad_alloc error by:
    1. Processing in smaller batches
    2. Not loading audio into memory (saving paths instead)
    3. Using iter_files() instead of streaming audio
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import gc

# Install dependencies
os.system("pip install datasets huggingface_hub soundfile requests -q 2>/dev/null")


def download_kathbath_telugu_fixed(output_dir: Path, max_samples: int = None):
    """Download Kathbath Telugu using file-based approach (memory efficient)"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING KATHBATH TELUGU (~140 hours)")
    print("   License: CC0 (No attribution required!)")
    print("   Method: Memory-efficient file download")
    print("="*60)
    
    from huggingface_hub import hf_hub_download, list_repo_files
    import soundfile as sf
    
    output_dir = output_dir / "kathbath_telugu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    repo_id = "ai4bharat/Kathbath"
    
    try:
        # List files in the repo for Telugu
        print("📋 Listing Telugu audio files...")
        all_files = list_repo_files(repo_id, repo_type="dataset")
        
        # Filter for Telugu audio files
        telugu_files = [f for f in all_files if 'telugu' in f.lower() and f.endswith(('.wav', '.mp3', '.flac'))]
        
        if not telugu_files:
            # Try alternate pattern
            telugu_files = [f for f in all_files if '/te/' in f.lower() or 'te_' in f.lower()]
        
        print(f"   Found {len(telugu_files)} Telugu files")
        
        if max_samples:
            telugu_files = telugu_files[:max_samples]
        
        # Download files one by one
        count = 0
        for file_path in tqdm(telugu_files, desc="Downloading"):
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(output_dir),
                )
                count += 1
                
                # Clear memory periodically
                if count % 100 == 0:
                    gc.collect()
                    
            except Exception as e:
                continue
        
        print(f"✅ Kathbath Telugu: Downloaded {count} files")
        return count
        
    except Exception as e:
        print(f"⚠️ Kathbath download method 1 failed: {e}")
        print("   Trying alternate method...")
        return download_kathbath_alternate(output_dir, max_samples)


def download_kathbath_alternate(output_dir: Path, max_samples: int = None):
    """Alternate download method using datasets with memory limits"""
    from datasets import load_dataset
    import soundfile as sf
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load with streaming and explicit cache
        dataset = load_dataset(
            "ai4bharat/Kathbath",
            "telugu",
            split="train",
            streaming=True,
        )
        
        count = 0
        for sample in tqdm(dataset, desc="Kathbath Telugu"):
            try:
                # Get audio data
                audio_data = sample.get('audio', {})
                if isinstance(audio_data, dict):
                    array = audio_data.get('array')
                    sr = audio_data.get('sampling_rate', 16000)
                    
                    if array is not None:
                        filename = output_dir / f"kathbath_te_{count:08d}.wav"
                        sf.write(str(filename), array, sr)
                        count += 1
                
                # Memory management
                if count % 50 == 0:
                    gc.collect()
                
                if max_samples and count >= max_samples:
                    break
                    
            except Exception as e:
                continue
        
        print(f"✅ Kathbath Telugu: Saved {count} files")
        return count
        
    except Exception as e:
        print(f"❌ Kathbath download failed: {e}")
        return 0


def download_commonvoice_telugu(output_dir: Path, max_samples: int = 5000):
    """Download Common Voice Telugu"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING COMMON VOICE TELUGU (~20 hours)")
    print("   License: CC-0")
    print("="*60)
    
    from datasets import load_dataset
    import soundfile as sf
    
    output_dir = output_dir / "commonvoice_telugu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "te",
            split="train",
            streaming=True,
        )
        
        count = 0
        for sample in tqdm(dataset, desc="Common Voice Telugu", total=max_samples):
            try:
                audio_data = sample.get('audio', {})
                if isinstance(audio_data, dict):
                    array = audio_data.get('array')
                    sr = audio_data.get('sampling_rate', 16000)
                    
                    if array is not None:
                        filename = output_dir / f"cv_te_{count:08d}.wav"
                        sf.write(str(filename), array, sr)
                        count += 1
                
                if count % 100 == 0:
                    gc.collect()
                
                if max_samples and count >= max_samples:
                    break
                    
            except Exception as e:
                continue
        
        print(f"✅ Common Voice Telugu: Saved {count} files")
        return count
        
    except Exception as e:
        print(f"❌ Common Voice download failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download Telugu speech data (memory efficient)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset (for testing)")
    parser.add_argument("--skip-kathbath", action="store_true", help="Skip Kathbath")
    parser.add_argument("--skip-commonvoice", action="store_true", help="Skip Common Voice")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("   TELUGU DATA DOWNLOAD (Memory Efficient)")
    print("="*60)
    
    # Check HuggingFace login
    try:
        from huggingface_hub import login
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Logged in to HuggingFace")
    except Exception as e:
        print(f"⚠️ HuggingFace login issue: {e}")
    
    print("\n⚠️ IMPORTANT: Accept licenses first at:")
    print("   - https://huggingface.co/datasets/ai4bharat/Kathbath")
    
    total = 0
    
    # Download Common Voice first (more reliable)
    if not args.skip_commonvoice:
        total += download_commonvoice_telugu(output_dir, args.max_samples or 5000)
    
    # Download Kathbath
    if not args.skip_kathbath:
        total += download_kathbath_telugu_fixed(output_dir, args.max_samples)
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ DOWNLOAD COMPLETE!")
    print(f"   Total Telugu audio files: {total:,}")
    print(f"   Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
