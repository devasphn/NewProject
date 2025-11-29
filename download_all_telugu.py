#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    DOWNLOAD ALL TELUGU DATA - 470+ HOURS
    
    Sources:
    1. Kathbath Telugu: ~140h (CC0 - No attribution!)
    2. IndicVoices Telugu: ~300h+ (CC-BY-4.0)
    3. Common Voice Telugu: ~20h (CC-0)
    
    PREREQUISITES:
    1. pip install datasets huggingface_hub soundfile tqdm
    2. huggingface-cli login
    3. Accept licenses at:
       - https://huggingface.co/datasets/ai4bharat/Kathbath
       - https://huggingface.co/datasets/ai4bharat/IndicVoices
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Install dependencies if needed
os.system("pip install datasets huggingface_hub soundfile tqdm -q")

import soundfile as sf
from datasets import load_dataset
from huggingface_hub import login


def download_kathbath_telugu(output_dir: Path, max_samples: int = None):
    """Download Kathbath Telugu (~140 hours)"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING KATHBATH TELUGU (~140 hours)")
    print("   License: CC0 (No attribution required!)")
    print("="*60)
    
    output_dir = output_dir / "kathbath_telugu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset(
            "ai4bharat/Kathbath",
            "telugu",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for sample in tqdm(dataset, desc="Kathbath Telugu"):
            try:
                audio = sample['audio']
                filename = output_dir / f"kathbath_te_{count:08d}.wav"
                sf.write(str(filename), audio['array'], audio['sampling_rate'])
                count += 1
                
                if max_samples and count >= max_samples:
                    break
            except Exception as e:
                continue
        
        print(f"✅ Kathbath Telugu: Saved {count} files")
        return count
        
    except Exception as e:
        print(f"❌ Error downloading Kathbath: {e}")
        print("   Make sure you accepted the license at:")
        print("   https://huggingface.co/datasets/ai4bharat/Kathbath")
        return 0


def download_indicvoices_telugu(output_dir: Path, max_samples: int = None):
    """Download IndicVoices Telugu (~300+ hours)"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING INDICVOICES TELUGU (~300+ hours)")
    print("   License: CC-BY-4.0")
    print("="*60)
    
    output_dir = output_dir / "indicvoices_telugu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset(
            "ai4bharat/IndicVoices",
            "telugu",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for sample in tqdm(dataset, desc="IndicVoices Telugu"):
            try:
                audio = sample['audio']
                filename = output_dir / f"indicvoices_te_{count:08d}.wav"
                sf.write(str(filename), audio['array'], audio['sampling_rate'])
                count += 1
                
                if max_samples and count >= max_samples:
                    break
            except Exception as e:
                continue
        
        print(f"✅ IndicVoices Telugu: Saved {count} files")
        return count
        
    except Exception as e:
        print(f"❌ Error downloading IndicVoices: {e}")
        print("   Make sure you accepted the license at:")
        print("   https://huggingface.co/datasets/ai4bharat/IndicVoices")
        return 0


def download_commonvoice_telugu(output_dir: Path, max_samples: int = None):
    """Download Common Voice Telugu (~20 hours)"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING COMMON VOICE TELUGU (~20 hours)")
    print("   License: CC-0")
    print("="*60)
    
    output_dir = output_dir / "commonvoice_telugu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "te",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for sample in tqdm(dataset, desc="Common Voice Telugu"):
            try:
                audio = sample['audio']
                filename = output_dir / f"cv_te_{count:08d}.wav"
                sf.write(str(filename), audio['array'], audio['sampling_rate'])
                count += 1
                
                if max_samples and count >= max_samples:
                    break
            except Exception as e:
                continue
        
        print(f"✅ Common Voice Telugu: Saved {count} files")
        return count
        
    except Exception as e:
        print(f"❌ Error downloading Common Voice: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download all Telugu speech data")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--skip-kathbath", action="store_true", help="Skip Kathbath")
    parser.add_argument("--skip-indicvoices", action="store_true", help="Skip IndicVoices")
    parser.add_argument("--skip-commonvoice", action="store_true", help="Skip Common Voice")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("   TELUGU DATA DOWNLOAD - 470+ HOURS")
    print("="*60)
    
    # Check HuggingFace login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        print("⚠️ No HF_TOKEN found. Run: huggingface-cli login")
        print("⚠️ Accept licenses first:")
        print("   - https://huggingface.co/datasets/ai4bharat/Kathbath")
        print("   - https://huggingface.co/datasets/ai4bharat/IndicVoices")
    
    total = 0
    
    # Download each dataset
    if not args.skip_kathbath:
        total += download_kathbath_telugu(output_dir, args.max_samples)
    
    if not args.skip_indicvoices:
        total += download_indicvoices_telugu(output_dir, args.max_samples)
    
    if not args.skip_commonvoice:
        total += download_commonvoice_telugu(output_dir, args.max_samples)
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ DOWNLOAD COMPLETE!")
    print(f"   Total Telugu audio files: {total:,}")
    print(f"   Output directory: {output_dir}")
    print("="*60)
    print("\nNext step: Train codec with:")
    print(f"  python train_codec_production.py --data_dirs {output_dir}/*")


if __name__ == "__main__":
    main()
