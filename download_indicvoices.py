"""
═══════════════════════════════════════════════════════════════════════════════
    INDICVOICES DOWNLOAD SCRIPT
    19,550 HOURS of Indian Language Speech!
    
    Languages: 22 Indian languages including Telugu, Hindi, Tamil, etc.
    License: CC-BY-4.0 (Commercial use allowed with attribution)
    
    NOTE: You need to accept the license on HuggingFace first!
    Visit: https://huggingface.co/datasets/ai4bharat/IndicVoices
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# Install required packages
os.system("pip install datasets huggingface_hub soundfile tqdm -q")

from datasets import load_dataset
from huggingface_hub import login

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Languages to download (all 22 available)
LANGUAGES = [
    "telugu",
    "hindi", 
    "tamil",
    "kannada",
    "malayalam",
    "marathi",
    "gujarati",
    "bengali",
    "odia",
    "punjabi",
    "assamese",
    "bodo",
    "dogri",
    "kashmiri",
    "konkani",
    "maithili",
    "manipuri",
    "nepali",
    "sanskrit",
    "santali",
    "sindhi",
    "urdu"
]

# Priority languages (download first)
PRIORITY_LANGUAGES = ["telugu", "hindi", "tamil", "kannada", "malayalam", "bengali", "marathi"]

# Output directory
OUTPUT_DIR = Path("data/indicvoices")

# Maximum samples per language (set to None for all)
MAX_SAMPLES_PER_LANGUAGE = None  # Set to 50000 to limit

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def download_language(language: str, max_samples: int = None):
    """Download a single language from IndicVoices"""
    
    output_path = OUTPUT_DIR / language
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {language}...")
    
    try:
        # Load dataset (streaming to handle large size)
        dataset = load_dataset(
            "ai4bharat/IndicVoices",
            language,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for sample in tqdm(dataset, desc=f"Saving {language}"):
            try:
                audio = sample['audio']
                audio_array = audio['array']
                sample_rate = audio['sampling_rate']
                
                # Save audio file
                filename = output_path / f"{language}_{count:08d}.wav"
                sf.write(str(filename), audio_array, sample_rate)
                
                count += 1
                
                if max_samples and count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error saving sample {count}: {e}")
                continue
        
        print(f"✅ {language}: Saved {count} samples")
        return count
        
    except Exception as e:
        print(f"❌ Failed to download {language}: {e}")
        return 0


def main():
    print("═" * 70)
    print("   INDICVOICES DOWNLOAD - 19,550 HOURS")
    print("═" * 70)
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        print("⚠️ No HF_TOKEN found. You may need to login:")
        print("   huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        print("")
        print("⚠️ IMPORTANT: Accept the license first at:")
        print("   https://huggingface.co/datasets/ai4bharat/IndicVoices")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download priority languages first
    print("\n" + "─" * 70)
    print("📥 Downloading PRIORITY languages...")
    print("─" * 70)
    
    total_samples = 0
    for lang in PRIORITY_LANGUAGES:
        count = download_language(lang, MAX_SAMPLES_PER_LANGUAGE)
        total_samples += count
    
    # Download remaining languages
    print("\n" + "─" * 70)
    print("📥 Downloading remaining languages...")
    print("─" * 70)
    
    remaining = [l for l in LANGUAGES if l not in PRIORITY_LANGUAGES]
    for lang in remaining:
        count = download_language(lang, MAX_SAMPLES_PER_LANGUAGE)
        total_samples += count
    
    # Summary
    print("\n" + "═" * 70)
    print(f"✅ DOWNLOAD COMPLETE!")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Output: {OUTPUT_DIR}")
    print("═" * 70)


if __name__ == "__main__":
    main()
