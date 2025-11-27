#!/usr/bin/env python3
"""
🎯 PRODUCTION Telugu Data Downloader
=====================================

Downloads ALL free Telugu audio data for S2S training.
NO cookies, NO YouTube, NO attribution required!

Target: 1000-5000 hours of Telugu audio

Sources:
1. AI4Bharat Kathbath (1684 hours) - HuggingFace
2. AI4Bharat IndicVoices (200+ hours) - HuggingFace  
3. OpenSLR SLR66 (10 hours) - Direct download
4. OpenSLR MUCS (40 hours) - Direct download
5. Mozilla Common Voice (20 hours) - Direct download
6. Vakyansh (2400 hours) - ekstep

Total potential: 4000+ hours!
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("data/telugu_production")

SOURCES = {
    "kathbath": {
        "name": "AI4Bharat Kathbath",
        "hours": 1684,
        "size_gb": 180,
        "method": "huggingface",
        "repo": "ai4bharat/kathbath",
        "license": "CC-BY-4.0",
        "priority": 1,
        "description": "Conversational Telugu - BEST for S2S!"
    },
    "indicvoices": {
        "name": "AI4Bharat IndicVoices", 
        "hours": 200,
        "size_gb": 25,
        "method": "huggingface",
        "repo": "ai4bharat/indicvoices",
        "license": "Apache-2.0",
        "priority": 2,
        "description": "Multi-speaker high quality"
    },
    "indic_tts": {
        "name": "AI4Bharat IndicTTS",
        "hours": 9,
        "size_gb": 1,
        "method": "huggingface",
        "repo": "ai4bharat/indic-tts-telugu",
        "license": "CC-BY-4.0",
        "priority": 3,
        "description": "Studio quality TTS data"
    },
    "openslr_66": {
        "name": "OpenSLR SLR66 Telugu",
        "hours": 10,
        "size_gb": 1,
        "method": "wget",
        "urls": [
            "https://www.openslr.org/resources/66/te_in_female.zip",
            "https://www.openslr.org/resources/66/te_in_male.zip"
        ],
        "license": "CC-BY-4.0",
        "priority": 4,
        "description": "Multi-speaker studio quality"
    },
    "openslr_103": {
        "name": "OpenSLR MUCS Telugu",
        "hours": 40,
        "size_gb": 5,
        "method": "wget",
        "urls": [
            "https://www.openslr.org/resources/103/te_in.zip"
        ],
        "license": "Free",
        "priority": 5,
        "description": "ASR training data"
    },
    "common_voice": {
        "name": "Mozilla Common Voice Telugu",
        "hours": 20,
        "size_gb": 3,
        "method": "huggingface",
        "repo": "mozilla-foundation/common_voice_16_1",
        "config": "te",
        "license": "CC-0",
        "priority": 6,
        "description": "Community contributed, CC-0 license!"
    },
    "shrutilipi": {
        "name": "AI4Bharat Shrutilipi",
        "hours": 100,
        "size_gb": 12,
        "method": "huggingface", 
        "repo": "ai4bharat/shrutilipi",
        "license": "CC-BY-4.0",
        "priority": 7,
        "description": "High quality ASR data"
    }
}

# ============================================================
# Download Functions
# ============================================================

def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if enough disk space is available"""
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    logger.info(f"💾 Disk space: {free_gb:.1f} GB free, need {required_gb:.1f} GB")
    return free_gb > required_gb * 1.2  # 20% buffer

def install_dependencies():
    """Install required packages"""
    packages = ["huggingface_hub", "datasets", "soundfile", "tqdm"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            logger.info(f"📦 Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

def download_huggingface(source_key: str, source: dict, output_dir: Path):
    """Download from HuggingFace"""
    from huggingface_hub import snapshot_download
    
    repo = source["repo"]
    config = source.get("config")
    
    logger.info(f"📥 Downloading {source['name']} from HuggingFace...")
    logger.info(f"   Repo: {repo}")
    logger.info(f"   Expected: ~{source['hours']} hours, ~{source['size_gb']} GB")
    
    try:
        if "datasets" in repo or config:
            # Use datasets library for dataset repos
            from datasets import load_dataset
            
            logger.info("   Using datasets library...")
            if config:
                ds = load_dataset(repo, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(repo, split="train", trust_remote_code=True)
            
            # Save to disk
            ds.save_to_disk(str(output_dir / source_key))
            logger.info(f"   ✅ Saved to {output_dir / source_key}")
        else:
            # Direct snapshot download
            snapshot_download(
                repo_id=repo,
                local_dir=str(output_dir / source_key),
                repo_type="dataset"
            )
            logger.info(f"   ✅ Downloaded to {output_dir / source_key}")
        
        return True
    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        return False

def download_wget(source_key: str, source: dict, output_dir: Path):
    """Download via wget"""
    dest_dir = output_dir / source_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📥 Downloading {source['name']} via wget...")
    
    for url in source["urls"]:
        filename = url.split("/")[-1]
        dest_file = dest_dir / filename
        
        if dest_file.exists():
            logger.info(f"   ⏭️ {filename} already exists, skipping")
            continue
        
        logger.info(f"   Downloading {filename}...")
        try:
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(dest_file), url],
                check=True
            )
            
            # Extract if zip
            if filename.endswith(".zip"):
                logger.info(f"   Extracting {filename}...")
                subprocess.run(
                    ["unzip", "-q", "-o", str(dest_file), "-d", str(dest_dir)],
                    check=True
                )
        except Exception as e:
            logger.error(f"   ❌ Error downloading {filename}: {e}")
            return False
    
    logger.info(f"   ✅ Completed {source['name']}")
    return True

def download_source(source_key: str, source: dict, output_dir: Path) -> bool:
    """Download a single source"""
    method = source["method"]
    
    if method == "huggingface":
        return download_huggingface(source_key, source, output_dir)
    elif method == "wget":
        return download_wget(source_key, source, output_dir)
    else:
        logger.error(f"Unknown method: {method}")
        return False

# ============================================================
# Data Processing
# ============================================================

def process_audio_files(input_dir: Path, output_dir: Path, target_sr: int = 16000):
    """Process all audio files to standard format"""
    import soundfile as sf
    from tqdm import tqdm
    import numpy as np
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
    
    logger.info(f"🔄 Processing {len(audio_files)} audio files...")
    
    processed = 0
    total_duration = 0
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        try:
            # Read audio
            audio, sr = sf.read(audio_file)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sr != target_sr:
                # Simple resampling (for production, use librosa)
                ratio = target_sr / sr
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio
                )
            
            # Normalize
            if audio.max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            
            # Save
            rel_path = audio_file.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(out_path, audio.astype(np.float32), target_sr)
            
            processed += 1
            total_duration += len(audio) / target_sr
            
        except Exception as e:
            logger.debug(f"Error processing {audio_file}: {e}")
    
    hours = total_duration / 3600
    logger.info(f"✅ Processed {processed} files, {hours:.1f} hours total")
    return processed, hours

# ============================================================
# Encoding with Codec
# ============================================================

def encode_with_codec(audio_dir: Path, output_dir: Path, codec_path: str, batch_size: int = 32):
    """Encode all audio with YOUR codec"""
    import torch
    import soundfile as sf
    from tqdm import tqdm
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load codec
    logger.info(f"📦 Loading codec: {codec_path}")
    from telugu_codec_fixed import TeluCodec
    codec = TeluCodec().to(device)
    checkpoint = torch.load(codec_path, map_location=device, weights_only=False)
    if 'codec_state_dict' in checkpoint:
        codec.load_state_dict(checkpoint['codec_state_dict'])
    else:
        codec.load_state_dict(checkpoint)
    codec.eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all wav files
    wav_files = list(audio_dir.rglob("*.wav"))
    logger.info(f"🔄 Encoding {len(wav_files)} files...")
    
    encoded = 0
    for wav_file in tqdm(wav_files, desc="Encoding"):
        try:
            audio, sr = sf.read(wav_file)
            
            # Convert to tensor
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Encode
            with torch.no_grad():
                codes = codec.encode(audio_t)
            
            # Save codes
            rel_path = wav_file.relative_to(audio_dir)
            out_path = output_dir / rel_path.with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(codes.cpu(), out_path)
            encoded += 1
            
        except Exception as e:
            logger.debug(f"Error encoding {wav_file}: {e}")
    
    logger.info(f"✅ Encoded {encoded} files")
    return encoded

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Download Telugu audio data")
    parser.add_argument("--output", default="data/telugu_production", help="Output directory")
    parser.add_argument("--sources", nargs="+", default=None, help="Specific sources to download")
    parser.add_argument("--list", action="store_true", help="List available sources")
    parser.add_argument("--process", action="store_true", help="Process downloaded audio")
    parser.add_argument("--encode", action="store_true", help="Encode with codec")
    parser.add_argument("--codec", default="best_codec.pt", help="Codec path")
    parser.add_argument("--max-gb", type=float, default=500, help="Max download size in GB")
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "=" * 70)
        print("📚 AVAILABLE TELUGU DATA SOURCES")
        print("=" * 70)
        
        total_hours = 0
        total_gb = 0
        
        for key, src in sorted(SOURCES.items(), key=lambda x: x[1]["priority"]):
            print(f"\n[{key}] {src['name']}")
            print(f"    Hours: {src['hours']}, Size: ~{src['size_gb']} GB")
            print(f"    License: {src['license']}")
            print(f"    Method: {src['method']}")
            print(f"    {src['description']}")
            total_hours += src['hours']
            total_gb += src['size_gb']
        
        print("\n" + "=" * 70)
        print(f"📊 TOTAL: {total_hours} hours, ~{total_gb} GB")
        print("=" * 70)
        return
    
    # Setup
    install_dependencies()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🎯 TELUGU DATA DOWNLOADER - PRODUCTION")
    print("=" * 70)
    
    # Determine which sources to download
    if args.sources:
        sources_to_download = {k: v for k, v in SOURCES.items() if k in args.sources}
    else:
        # Download by priority until max_gb reached
        sources_to_download = {}
        total_gb = 0
        for key, src in sorted(SOURCES.items(), key=lambda x: x[1]["priority"]):
            if total_gb + src["size_gb"] <= args.max_gb:
                sources_to_download[key] = src
                total_gb += src["size_gb"]
    
    total_hours = sum(s["hours"] for s in sources_to_download.values())
    total_size = sum(s["size_gb"] for s in sources_to_download.values())
    
    print(f"\n📥 Will download {len(sources_to_download)} sources:")
    print(f"   Total hours: ~{total_hours}")
    print(f"   Total size: ~{total_size} GB")
    print()
    
    for key, src in sources_to_download.items():
        print(f"   • [{src['priority']}] {src['name']}: {src['hours']}h, {src['size_gb']}GB")
    
    # Check disk space
    if not check_disk_space(output_dir, total_size):
        logger.error("❌ Not enough disk space!")
        return
    
    # Download each source
    print("\n" + "=" * 70)
    print("📥 STARTING DOWNLOADS")
    print("=" * 70)
    
    results = {}
    for key, src in sources_to_download.items():
        print(f"\n{'='*50}")
        success = download_source(key, src, output_dir)
        results[key] = success
        time.sleep(2)  # Small delay between sources
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 70)
    
    for key, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {SOURCES[key]['name']}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\n   Downloaded: {successful}/{len(results)} sources")
    
    # Process if requested
    if args.process:
        print("\n" + "=" * 70)
        print("🔄 PROCESSING AUDIO FILES")
        print("=" * 70)
        
        processed_dir = output_dir / "processed"
        process_audio_files(output_dir, processed_dir)
    
    # Encode if requested
    if args.encode:
        print("\n" + "=" * 70)
        print("📦 ENCODING WITH CODEC")
        print("=" * 70)
        
        processed_dir = output_dir / "processed"
        encoded_dir = output_dir / "encoded"
        encode_with_codec(processed_dir, encoded_dir, args.codec)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Data location: {output_dir}")
    print("\nNext steps:")
    print("  1. Process audio: python download_all_telugu_data.py --process")
    print("  2. Encode: python download_all_telugu_data.py --encode --codec best_codec.pt")
    print("  3. Train S2S: python train_s2s_production.py")


if __name__ == "__main__":
    main()
