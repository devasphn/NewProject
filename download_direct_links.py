#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  DIRECT DOWNLOAD SCRIPT - NO HUGGINGFACE LIBRARY
  
  Downloads IndicVoices-R directly via wget (bypasses HuggingFace entirely)
  
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║  VERIFIED WORKING LINKS (from official GitHub repo)                       ║
  ║  Source: https://github.com/AI4Bharat/IndicVoices-R/blob/master/data_links.txt ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
  
  Run on RunPod: python download_direct_links.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import subprocess
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFIED DIRECT DOWNLOAD LINKS (FROM OFFICIAL GITHUB REPO - NO HUGGINGFACE!)
# Source: https://github.com/AI4Bharat/IndicVoices-R/blob/master/data_links.txt
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/workspace/data")

# IndicVoices-R - VERIFIED Direct tar.gz downloads (1700+ hours total, 22 languages)
# These URLs are from the official data_links.txt in the GitHub repo
INDICVOICES_R_LINKS = {
    'hindi': "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Hindi.tar.gz",
    'telugu': "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Telugu.tar.gz",
}

# Full list of all IndicVoices-R languages (for reference)
ALL_INDICVOICES_R = [
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Assamese.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Bengali.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Bodo.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Dogri.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Gujarati.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Hindi.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Kannada.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Kashmiri.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Konkani.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Maithili.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Malayalam.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Marathi.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Manipuri.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Nepali.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Odia.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Punjabi.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Sanskrit.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Santali.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Sindhi.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Tamil.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Telugu.tar.gz",
    "https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Urdu.tar.gz",
]


def run_command(cmd: str, cwd: str = None, timeout: int = 7200) -> tuple:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def download_with_wget(url: str, output_dir: Path, extract: bool = True) -> bool:
    """Download a file using wget and optionally extract it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split('/')[-1]
    filepath = output_dir / filename
    
    print(f"  📥 Downloading: {filename}")
    print(f"     URL: {url}")
    print(f"     To: {output_dir}")
    
    # Download with wget
    cmd = f'wget -c -q --show-progress "{url}" -O "{filepath}"'
    success, output = run_command(cmd, cwd=str(output_dir))
    
    if not success:
        # Try curl as fallback
        print(f"  ⚠️ wget failed, trying curl...")
        cmd = f'curl -L -C - -o "{filepath}" "{url}"'
        success, output = run_command(cmd, cwd=str(output_dir))
    
    if not success:
        print(f"  ❌ Download failed: {output[:100]}")
        return False
    
    print(f"  ✅ Downloaded: {filename}")
    
    # Check file size
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"     Size: {size_mb:.1f} MB")
        
        if size_mb < 1:
            print(f"  ⚠️ File too small, may be an error page")
            # Read first 500 bytes to check
            with open(filepath, 'rb') as f:
                header = f.read(500)
                if b'<!DOCTYPE' in header or b'<html' in header:
                    print(f"  ❌ Downloaded HTML error page, not data")
                    filepath.unlink()
                    return False
    
    # Extract if requested
    if extract and filepath.exists():
        print(f"  📦 Extracting: {filename}")
        
        if filename.endswith('.tar.gz') or filename.endswith('.tgz'):
            cmd = f'tar -xzf "{filepath}" -C "{output_dir}"'
        elif filename.endswith('.tar'):
            cmd = f'tar -xf "{filepath}" -C "{output_dir}"'
        elif filename.endswith('.zip'):
            cmd = f'unzip -o "{filepath}" -d "{output_dir}"'
        else:
            print(f"  ⚠️ Unknown archive format, skipping extraction")
            return True
        
        success, output = run_command(cmd, cwd=str(output_dir), timeout=1800)
        
        if success:
            print(f"  ✅ Extracted successfully")
            # Remove archive to save space
            filepath.unlink()
        else:
            print(f"  ⚠️ Extraction failed: {output[:100]}")
            return False
    
    return True


def count_audio_files(directory: Path) -> int:
    """Count audio files in directory."""
    if not directory.exists():
        return 0
    
    count = 0
    for ext in ['.wav', '.flac', '.mp3', '.ogg']:
        result = subprocess.run(
            f'find "{directory}" -name "*{ext}" 2>/dev/null | wc -l',
            shell=True, capture_output=True, text=True
        )
        try:
            count += int(result.stdout.strip())
        except:
            pass
    return count


def check_url_exists(url: str) -> bool:
    """Check if a URL exists without downloading."""
    cmd = f'curl -sI "{url}" | head -1'
    success, output = run_command(cmd, timeout=30)
    return success and ('200' in output or '302' in output)


def download_indicvoices_r():
    """Download IndicVoices-R dataset (direct links)."""
    print("\n" + "═" * 70)
    print("  DOWNLOADING INDICVOICES-R (DIRECT LINKS)")
    print("═" * 70)
    print("  Source: https://github.com/AI4Bharat/IndicVoices-R")
    print("  Total: ~1700 hours across 22 languages")
    print("═" * 70)
    
    for lang, url in INDICVOICES_R_LINKS.items():
        print(f"\n{'─' * 50}")
        print(f"  {lang.upper()}")
        print(f"{'─' * 50}")
        
        output_dir = BASE_DIR / lang / "indicvoices_r"
        
        # Check if already downloaded
        existing = count_audio_files(output_dir)
        if existing > 10000:
            print(f"  ✅ Already has {existing} files, skipping")
            continue
        
        # Check if URL is valid
        print(f"  Checking URL...")
        if not check_url_exists(url):
            print(f"  ⚠️ URL not accessible, trying alternative...")
            # Try original IndicVoices
            url = INDICVOICES_ORIGINAL.get(lang)
            if url and check_url_exists(url):
                output_dir = BASE_DIR / lang / "indicvoices"
                print(f"  Found alternative: IndicVoices original")
            else:
                print(f"  ❌ No valid URL found for {lang}")
                continue
        
        # Download
        success = download_with_wget(url, output_dir)
        
        if success:
            files = count_audio_files(output_dir)
            print(f"  📊 Total files: {files}")


def download_kathbath_direct():
    """Try to download Kathbath via direct links."""
    print("\n" + "═" * 70)
    print("  DOWNLOADING KATHBATH (DIRECT LINKS)")
    print("═" * 70)
    
    for lang, url in KATHBATH_LINKS.items():
        print(f"\n{'─' * 50}")
        print(f"  {lang.upper()}")
        print(f"{'─' * 50}")
        
        output_dir = BASE_DIR / lang / "kathbath"
        
        # Check if already downloaded
        existing = count_audio_files(output_dir)
        if existing > 10000:
            print(f"  ✅ Already has {existing} files, skipping")
            continue
        
        # Check if URL is valid
        print(f"  Checking URL...")
        if check_url_exists(url):
            success = download_with_wget(url, output_dir)
            if success:
                files = count_audio_files(output_dir)
                print(f"  📊 Total files: {files}")
        else:
            print(f"  ⚠️ Direct link not available")
            print(f"     Kathbath requires HuggingFace download")
            print(f"     See: https://huggingface.co/datasets/ai4bharat/Kathbath")


def main():
    print("═" * 70)
    print("  DIRECT DOWNLOAD SCRIPT (NO HUGGINGFACE LIBRARY)")
    print("═" * 70)
    print(f"  Base directory: {BASE_DIR}")
    print("═" * 70)
    
    # Check tools
    print("\n📦 Checking tools...")
    for tool in ['wget', 'curl', 'tar']:
        result = subprocess.run(f'which {tool}', shell=True, capture_output=True)
        status = "✅" if result.returncode == 0 else "❌"
        print(f"  {status} {tool}")
    
    # Download IndicVoices-R
    download_indicvoices_r()
    
    # Try Kathbath direct links
    download_kathbath_direct()
    
    # Final summary
    print("\n" + "═" * 70)
    print("  DOWNLOAD SUMMARY")
    print("═" * 70)
    
    total_files = 0
    for lang in ['english', 'hindi', 'telugu']:
        lang_dir = BASE_DIR / lang
        files = count_audio_files(lang_dir)
        hours = files * 5 / 3600  # Estimate 5 sec per file
        total_files += files
        print(f"  {lang}: {files} files (~{hours:.0f}h)")
    
    total_hours = total_files * 5 / 3600
    print(f"\n  TOTAL: {total_files} files (~{total_hours:.0f}h)")
    
    print("\n" + "═" * 70)
    print("  NEXT STEPS")
    print("═" * 70)
    print("  1. Check status: python check_data_status.py")
    print("  2. Augment data: python augment_all_data.py")
    print("  3. Train codec: python train_codec_production.py")
    print("═" * 70)


if __name__ == "__main__":
    main()
