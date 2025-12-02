#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  DOWNLOAD HUGGINGFACE DATASETS
  IndicVoices, Kathbath, Common Voice for Hindi and Telugu
  
  PREREQUISITES:
  1. pip install datasets soundfile tqdm huggingface_hub
  2. huggingface-cli login
  3. Accept licenses at:
     - https://huggingface.co/datasets/ai4bharat/IndicVoices
     - https://huggingface.co/datasets/ai4bharat/Kathbath
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import gc
from pathlib import Path
from tqdm import tqdm
import argparse

# Check dependencies
try:
    from datasets import load_dataset
    import soundfile as sf
except ImportError:
    print("Installing required packages...")
    os.system("pip install datasets soundfile tqdm huggingface_hub")
    from datasets import load_dataset
    import soundfile as sf


def save_audio_streaming(dataset, output_dir, prefix, max_samples=None):
    """Save audio from streaming dataset to disk efficiently"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    errors = 0
    
    print(f"\n📥 Downloading to {output_dir}...")
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Downloading {prefix}")):
        if max_samples and i >= max_samples:
            break
        
        try:
            audio = sample.get('audio')
            if audio is None:
                continue
                
            array = audio['array']
            sr = audio['sampling_rate']
            
            # Save as WAV
            output_file = output_path / f"{prefix}_{count:08d}.wav"
            sf.write(str(output_file), array, sr)
            count += 1
            
            # Memory cleanup every 1000 samples
            if count % 1000 == 0:
                gc.collect()
                
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"\n⚠️ Error on sample {i}: {e}")
            continue
    
    print(f"✅ Saved {count} files to {output_dir}")
    if errors > 0:
        print(f"⚠️ {errors} samples failed")
    
    return count


def save_audio_batch(dataset, output_dir, prefix):
    """Save audio from non-streaming dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    print(f"\n📥 Downloading to {output_dir}...")
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Downloading {prefix}")):
        try:
            audio = sample.get('audio')
            if audio is None:
                continue
                
            array = audio['array']
            sr = audio['sampling_rate']
            
            output_file = output_path / f"{prefix}_{count:08d}.wav"
            sf.write(str(output_file), array, sr)
            count += 1
            
        except Exception as e:
            continue
    
    print(f"✅ Saved {count} files to {output_dir}")
    return count


def download_indicvoices(languages=['hindi', 'telugu'], base_dir='/workspace/data'):
    """Download IndicVoices dataset"""
    print("\n" + "="*70)
    print("  INDICVOICES (19,550 hours across 22 languages)")
    print("  License: CC BY 4.0")
    print("="*70)
    
    for lang in languages:
        output_dir = f"{base_dir}/{lang}/indicvoices"
        
        try:
            print(f"\n🔄 Loading IndicVoices {lang}...")
            
            # Use streaming to avoid memory issues
            ds = load_dataset(
                "ai4bharat/IndicVoices",
                lang,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            # Download samples
            save_audio_streaming(ds, output_dir, f"indicvoices_{lang[:2]}")
            
        except Exception as e:
            print(f"\n❌ IndicVoices {lang} failed: {e}")
            print("   Make sure you:")
            print("   1. Run: huggingface-cli login")
            print("   2. Accept license: https://huggingface.co/datasets/ai4bharat/IndicVoices")


def download_kathbath(languages=['hindi', 'telugu'], base_dir='/workspace/data'):
    """Download Kathbath dataset"""
    print("\n" + "="*70)
    print("  KATHBATH (1,684 hours across 12 languages)")
    print("  License: CC0 (No attribution required!)")
    print("="*70)
    
    for lang in languages:
        output_dir = f"{base_dir}/{lang}/kathbath"
        
        try:
            print(f"\n🔄 Loading Kathbath {lang}...")
            
            # Kathbath is smaller, can load directly
            ds = load_dataset(
                "ai4bharat/Kathbath",
                lang,
                split="train",
                trust_remote_code=True
            )
            
            save_audio_batch(ds, output_dir, f"kathbath_{lang[:2]}")
            
        except Exception as e:
            print(f"\n❌ Kathbath {lang} failed: {e}")
            print("   Accept license: https://huggingface.co/datasets/ai4bharat/Kathbath")


def download_commonvoice(languages=['te', 'hi'], base_dir='/workspace/data'):
    """Download Common Voice dataset"""
    print("\n" + "="*70)
    print("  COMMON VOICE (Mozilla)")
    print("  License: CC0 (Completely free)")
    print("="*70)
    
    lang_map = {'te': 'telugu', 'hi': 'hindi'}
    
    for lang in languages:
        output_lang = lang_map.get(lang, lang)
        output_dir = f"{base_dir}/{output_lang}/commonvoice"
        
        try:
            print(f"\n🔄 Loading Common Voice {lang}...")
            
            ds = load_dataset(
                "mozilla-foundation/common_voice_16_1",
                lang,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            save_audio_streaming(ds, output_dir, f"cv_{lang}")
            
        except Exception as e:
            print(f"\n❌ Common Voice {lang} failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace speech datasets")
    parser.add_argument("--base_dir", type=str, default="/workspace/data",
                       help="Base directory for data")
    parser.add_argument("--indicvoices", action="store_true",
                       help="Download IndicVoices")
    parser.add_argument("--kathbath", action="store_true",
                       help="Download Kathbath")
    parser.add_argument("--commonvoice", action="store_true",
                       help="Download Common Voice")
    parser.add_argument("--all", action="store_true",
                       help="Download all datasets")
    parser.add_argument("--languages", nargs="+", default=['hindi', 'telugu'],
                       help="Languages to download")
    
    args = parser.parse_args()
    
    # If no specific dataset selected, download all
    if not (args.indicvoices or args.kathbath or args.commonvoice):
        args.all = True
    
    print("\n" + "═"*70)
    print("  HUGGINGFACE DATA DOWNLOAD SCRIPT")
    print("═"*70)
    print(f"  Base directory: {args.base_dir}")
    print(f"  Languages: {args.languages}")
    print("═"*70)
    
    if args.all or args.kathbath:
        download_kathbath(args.languages, args.base_dir)
    
    if args.all or args.commonvoice:
        cv_langs = ['te' if l == 'telugu' else 'hi' for l in args.languages]
        download_commonvoice(cv_langs, args.base_dir)
    
    if args.all or args.indicvoices:
        download_indicvoices(args.languages, args.base_dir)
    
    # Final summary
    print("\n" + "═"*70)
    print("  DOWNLOAD COMPLETE!")
    print("═"*70)
    
    for lang in args.languages:
        lang_dir = Path(args.base_dir) / lang
        if lang_dir.exists():
            wav_count = len(list(lang_dir.rglob("*.wav")))
            print(f"  {lang.upper()}: {wav_count} audio files")
    
    print("═"*70)


if __name__ == "__main__":
    main()
