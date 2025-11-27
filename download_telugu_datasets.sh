#!/bin/bash
# Download Telugu Audio Datasets for S2S Training
# ================================================

echo "========================================"
echo "📥 Downloading Telugu Audio Datasets"
echo "========================================"

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    echo "📦 Installing unzip..."
    apt-get update -qq && apt-get install -y -qq unzip
fi

mkdir -p data/telugu_raw

# 1. OpenSLR SLR66 - Telugu Multi-speaker (10 hours)
echo ""
echo "📥 [1/5] Downloading OpenSLR SLR66 (Telugu Multi-speaker)..."
cd data/telugu_raw
wget -q --show-progress https://www.openslr.org/resources/66/te_in_female.zip
wget -q --show-progress https://www.openslr.org/resources/66/te_in_male.zip
unzip -q te_in_female.zip -d slr66_female
unzip -q te_in_male.zip -d slr66_male
rm -f te_in_female.zip te_in_male.zip
cd ../..
echo "✅ SLR66 downloaded!"

# 2. MUCS Telugu - OpenSLR 103 (40 hours ASR data)
echo ""
echo "📥 [2/5] Downloading MUCS Telugu (OpenSLR 103)..."
cd data/telugu_raw
wget -q --show-progress https://www.openslr.org/resources/103/te_in.zip
unzip -q te_in.zip -d mucs_telugu
rm -f te_in.zip
cd ../..
echo "✅ MUCS downloaded!"

# 3. Common Voice Telugu
echo ""
echo "📥 [3/5] Common Voice Telugu..."
echo "⚠️  Please download manually from: https://commonvoice.mozilla.org/en/datasets"
echo "    Select 'Telugu' and download the latest version"
mkdir -p data/telugu_raw/common_voice

# 4. IndicTTS Telugu (if available)
echo ""
echo "📥 [4/5] IndicTTS Telugu..."
echo "⚠️  Please download from AI4Bharat: https://github.com/AI4Bharat/Indic-TTS"
mkdir -p data/telugu_raw/indic_tts

# 5. Kathbath Conversational (1684 hours - BEST for conversation!)
echo ""
echo "📥 [5/5] Kathbath Conversational Dataset..."
echo "⚠️  This is the BEST dataset for conversation training!"
echo "    Download from: https://github.com/AI4Bharat/vistaar"
echo "    Or use HuggingFace: ai4bharat/kathbath"
mkdir -p data/telugu_raw/kathbath

echo ""
echo "========================================"
echo "📊 Dataset Summary"
echo "========================================"
echo ""
echo "Downloaded automatically:"
echo "  - OpenSLR SLR66: ~10 hours (multi-speaker)"
echo "  - MUCS Telugu: ~40 hours (ASR data)"
echo ""
echo "Please download manually:"
echo "  - Common Voice Telugu: ~5-10 hours"
echo "  - IndicTTS Telugu: ~8.7 hours"  
echo "  - Kathbath: ~1684 hours (conversational!)"
echo ""
echo "========================================"
echo ""
echo "📁 Data location: data/telugu_raw/"
echo ""
echo "Next step: Process the data for S2S training"
echo "  python process_telugu_datasets.py"
echo ""
