#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD FREE SPEECH DATASETS - NO YOUTUBE SCRAPING NEEDED!
# This script downloads all required datasets for codec and S2S training
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   DOWNLOADING FREE SPEECH DATASETS FOR TELUGU S2S PROJECT${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Base directories
DATA_DIR="/workspace/speech_data"
TELUGU_DIR="/workspace/telugu_data"
CODEC_DATA_DIR="/workspace/codec_training_data"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TELUGU_DIR"
mkdir -p "$CODEC_DATA_DIR"
mkdir -p "$DATA_DIR/librispeech"
mkdir -p "$DATA_DIR/vctk"
mkdir -p "$TELUGU_DIR/openslr"
mkdir -p "$TELUGU_DIR/indictts"

echo -e "${YELLOW}[1/5] Downloading LibriSpeech train-clean-100 (100 hours, ~6GB)${NC}"
echo "This is for CODEC training - language doesn't matter!"
cd "$DATA_DIR/librispeech"

if [ ! -f "train-clean-100.tar.gz" ]; then
    wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz
    echo "Extracting LibriSpeech..."
    tar -xzf train-clean-100.tar.gz
    echo -e "${GREEN}✓ LibriSpeech train-clean-100 downloaded${NC}"
else
    echo -e "${GREEN}✓ LibriSpeech already exists, skipping${NC}"
fi

echo ""
echo -e "${YELLOW}[2/5] Downloading OpenSLR Telugu SLR66 (~1GB, ~10 hours)${NC}"
echo "Multi-speaker Telugu recordings"
cd "$TELUGU_DIR/openslr"

if [ ! -f "te_in_male.zip" ]; then
    wget -c https://www.openslr.org/resources/66/te_in_male.zip
    wget -c https://www.openslr.org/resources/66/te_in_female.zip
    wget -c https://www.openslr.org/resources/66/line_index_male.tsv
    wget -c https://www.openslr.org/resources/66/line_index_female.tsv
    
    echo "Extracting Telugu data..."
    unzip -o te_in_male.zip
    unzip -o te_in_female.zip
    echo -e "${GREEN}✓ OpenSLR Telugu downloaded${NC}"
else
    echo -e "${GREEN}✓ OpenSLR Telugu already exists, skipping${NC}"
fi

echo ""
echo -e "${YELLOW}[3/5] Downloading IndicTTS Telugu from HuggingFace (~8.7 hours)${NC}"
echo "Studio-quality Telugu recordings"
cd "$TELUGU_DIR/indictts"

python3 << 'EOF'
import os
try:
    from datasets import load_dataset
    print("Downloading IndicTTS Telugu from HuggingFace...")
    ds = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train")
    
    # Save to disk
    output_dir = "/workspace/telugu_data/indictts/audio"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sample in enumerate(ds):
        audio = sample["audio"]
        text = sample.get("text", sample.get("sentence", ""))
        
        # Save audio
        import soundfile as sf
        audio_path = f"{output_dir}/sample_{i:05d}.wav"
        sf.write(audio_path, audio["array"], audio["sampling_rate"])
        
        # Save transcription
        with open(f"{output_dir}/sample_{i:05d}.txt", "w") as f:
            f.write(text)
        
        if i % 100 == 0:
            print(f"Processed {i} samples...")
    
    print(f"✓ Saved {len(ds)} IndicTTS samples")
except Exception as e:
    print(f"Warning: Could not download IndicTTS: {e}")
    print("You can manually download from: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu")
EOF

echo -e "${GREEN}✓ IndicTTS Telugu processed${NC}"

echo ""
echo -e "${YELLOW}[4/5] Downloading LJSpeech (24 hours, ~2.6GB)${NC}"
echo "High-quality single-speaker English for codec training"
cd "$DATA_DIR"

if [ ! -d "LJSpeech-1.1" ]; then
    wget -c https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    echo "Extracting LJSpeech..."
    tar -xjf LJSpeech-1.1.tar.bz2
    echo -e "${GREEN}✓ LJSpeech downloaded${NC}"
else
    echo -e "${GREEN}✓ LJSpeech already exists, skipping${NC}"
fi

echo ""
echo -e "${YELLOW}[5/5] Downloading Mozilla Common Voice Telugu (Optional)${NC}"
cd "$TELUGU_DIR"

python3 << 'EOF'
try:
    from datasets import load_dataset
    print("Downloading Common Voice Telugu (this may take a while)...")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "te", split="train", trust_remote_code=True)
    
    output_dir = "/workspace/telugu_data/common_voice/audio"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for i, sample in enumerate(ds):
        if i >= 5000:  # Limit to 5000 samples to save time
            break
        try:
            audio = sample["audio"]
            text = sample.get("sentence", "")
            
            import soundfile as sf
            audio_path = f"{output_dir}/cv_{i:05d}.wav"
            sf.write(audio_path, audio["array"], audio["sampling_rate"])
            
            with open(f"{output_dir}/cv_{i:05d}.txt", "w") as f:
                f.write(text)
            count += 1
        except:
            continue
        
        if i % 500 == 0:
            print(f"Processed {i} Common Voice samples...")
    
    print(f"✓ Saved {count} Common Voice samples")
except Exception as e:
    print(f"Warning: Could not download Common Voice: {e}")
    print("This is optional - you can proceed without it")
EOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   DOWNLOAD COMPLETE! SUMMARY:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Calculate sizes
echo "Dataset sizes:"
du -sh "$DATA_DIR/librispeech" 2>/dev/null || echo "LibriSpeech: Not found"
du -sh "$DATA_DIR/LJSpeech-1.1" 2>/dev/null || echo "LJSpeech: Not found"
du -sh "$TELUGU_DIR/openslr" 2>/dev/null || echo "OpenSLR Telugu: Not found"
du -sh "$TELUGU_DIR/indictts" 2>/dev/null || echo "IndicTTS Telugu: Not found"
du -sh "$TELUGU_DIR/common_voice" 2>/dev/null || echo "Common Voice Telugu: Not found"

echo ""
echo "Total speech data:"
du -sh "$DATA_DIR" 2>/dev/null
du -sh "$TELUGU_DIR" 2>/dev/null

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   NEXT STEPS:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "1. Verify downloads:"
echo "   ls -la $DATA_DIR"
echo "   ls -la $TELUGU_DIR"
echo ""
echo "2. Start codec training:"
echo "   python train_codec_dac.py --data_dir $DATA_DIR"
echo ""
echo "3. Train speaker embeddings:"
echo "   python train_speakers.py --data_dir $TELUGU_DIR"
echo ""
echo -e "${GREEN}No YouTube scraping needed! All data is FREE and LEGAL.${NC}"
