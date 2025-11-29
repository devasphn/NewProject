"""
═══════════════════════════════════════════════════════════════════════════════
    KATHBATH DOWNLOAD SCRIPT
    1,684 HOURS of Indian Language Speech (140+ hours Telugu!)
    
    Languages: 12 Indian languages
    License: CC0 (No attribution required!)
    
    NOTE: You need to accept the agreement on HuggingFace first!
    Visit: https://huggingface.co/datasets/ai4bharat/Kathbath
═══════════════════════════════════════════════════════════════════════════════
"""

import os
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

# All available languages in Kathbath
LANGUAGES = [
    "telugu",      # ~140 hours
    "hindi",       # ~140 hours
    "tamil",       # ~140 hours
    "bengali",     # ~140 hours
    "gujarati",    # ~140 hours
    "kannada",     # ~140 hours
    "malayalam",   # ~140 hours
    "marathi",     # ~140 hours
    "odia",        # ~140 hours
    "punjabi",     # ~140 hours
    "sanskrit",    # ~140 hours
    "urdu",        # ~140 hours
]

# Priority (download first)
PRIORITY = ["telugu", "hindi", "tamil", "bengali"]

OUTPUT_DIR = Path("data/kathbath")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def download_language(language: str, max_samples: int = None):
    """Download a single language from Kathbath"""
    
    output_path = OUTPUT_DIR / language
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading Kathbath {language}...")
    
    try:
        # Load dataset
        dataset = load_dataset(
            "ai4bharat/Kathbath",
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
                
                filename = output_path / f"kathbath_{language}_{count:08d}.wav"
                sf.write(str(filename), audio_array, sample_rate)
                
                count += 1
                
                if max_samples and count >= max_samples:
                    break
                    
            except Exception as e:
                continue
        
        print(f"✅ Kathbath {language}: Saved {count} samples")
        return count
        
    except Exception as e:
        print(f"❌ Failed to download {language}: {e}")
        print(f"   Make sure you accepted the agreement at:")
        print(f"   https://huggingface.co/datasets/ai4bharat/Kathbath")
        return 0


def main():
    print("═" * 70)
    print("   KATHBATH DOWNLOAD - 1,684 HOURS (12 Indian Languages)")
    print("═" * 70)
    print("   License: CC0 (No attribution required!)")
    print("═" * 70)
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        print("⚠️ No HF_TOKEN found.")
        print("⚠️ IMPORTANT: Accept the agreement first at:")
        print("   https://huggingface.co/datasets/ai4bharat/Kathbath")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download priority languages first
    print("\n📥 Downloading PRIORITY languages...")
    
    total = 0
    for lang in PRIORITY:
        total += download_language(lang)
    
    # Download rest
    print("\n📥 Downloading remaining languages...")
    
    for lang in LANGUAGES:
        if lang not in PRIORITY:
            total += download_language(lang)
    
    print("\n" + "═" * 70)
    print(f"✅ KATHBATH DOWNLOAD COMPLETE!")
    print(f"   Total samples: {total:,}")
    print("═" * 70)


if __name__ == "__main__":
    main()
