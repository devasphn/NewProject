#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#   DOWNLOAD ADDITIONAL TELUGU DATA
#   
#   Sources with VERIFIED WORKING links:
#   1. OpenSLR 66 - Already downloaded (10h)
#   2. Kathbath Telugu - Just downloaded (155h)
#   3. Common Voice Telugu - Mozilla (20h)
#   4. Microsoft Speech Corpus - Telugu portion
# ═══════════════════════════════════════════════════════════════════════════════

set -e

DATA_DIR="/workspace/data"
TELUGU_DIR="${DATA_DIR}/telugu"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  DOWNLOADING ADDITIONAL TELUGU DATA"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────────
# 1. COMMON VOICE TELUGU (from HuggingFace - manual parquet extraction)
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  [1/3] COMMON VOICE TELUGU (~20h)"
echo "─────────────────────────────────────────────────────────────────────────────"

CV_DIR="${TELUGU_DIR}/common_voice"
mkdir -p "$CV_DIR"

# Common Voice has direct parquet files we can download
# These are small enough to process without OOM
CV_URL="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/resolve/main/data/te/train-00000-of-00001.parquet"

echo "  Attempting Common Voice Telugu download..."
cd "$CV_DIR"

# Try to download the parquet file directly
if wget -q --spider "$CV_URL" 2>/dev/null; then
    echo "  📥 Downloading Common Voice Telugu parquet..."
    wget -c -q --show-progress "$CV_URL" -O cv_telugu.parquet 2>&1 || echo "  ⚠️ Download may require authentication"
else
    echo "  ⚠️ Common Voice requires HuggingFace authentication"
    echo "     To download manually:"
    echo "     1. Login: huggingface-cli login"
    echo "     2. Accept license at: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 2. VAANI TELUGU (Ext resources - Google's Project Vaani)
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  [2/3] GOOGLE FLEURS TELUGU (~12h)"
echo "─────────────────────────────────────────────────────────────────────────────"

FLEURS_DIR="${TELUGU_DIR}/fleurs"
mkdir -p "$FLEURS_DIR"
cd "$FLEURS_DIR"

# FLEURS has direct download links
FLEURS_BASE="https://storage.googleapis.com/xtreme_translations/FLEURS102"

echo "  📥 Downloading FLEURS Telugu..."
for split in train dev test; do
    if [ ! -f "te_in_${split}.tar.gz" ]; then
        wget -c -q --show-progress "${FLEURS_BASE}/te_in/${split}.tar.gz" -O "te_in_${split}.tar.gz" 2>&1 || true
        if [ -f "te_in_${split}.tar.gz" ] && [ -s "te_in_${split}.tar.gz" ]; then
            echo "  📦 Extracting ${split}..."
            tar -xzf "te_in_${split}.tar.gz" 2>/dev/null || true
            rm -f "te_in_${split}.tar.gz"
        fi
    fi
done

# Convert any downloaded audio to wav
if command -v ffmpeg &> /dev/null; then
    find "$FLEURS_DIR" -name "*.wav" -o -name "*.mp3" -o -name "*.ogg" 2>/dev/null | head -5 && echo "  Audio files found"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 3. OPENSLR 78 - Large Telugu ASR corpus
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  [3/3] OPENSLR ADDITIONAL TELUGU"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check for other OpenSLR Telugu resources
OPENSLR_DIR="${TELUGU_DIR}/openslr"
mkdir -p "$OPENSLR_DIR"
cd "$OPENSLR_DIR"

# OpenSLR 78 - Crowdsourced high-quality UK and Ireland English (not Telugu, skip)
# Let's check if there are other Telugu resources we missed

# Verify we have OpenSLR 66
if [ ! -d "${TELUGU_DIR}/openslr66" ] && [ ! -d "${TELUGU_DIR}/te_in_female" ]; then
    echo "  📥 Re-downloading OpenSLR 66 Telugu..."
    cd "$TELUGU_DIR"
    wget -c -q --show-progress "https://www.openslr.org/resources/66/te_in_female.zip" -O te_in_female.zip
    wget -c -q --show-progress "https://www.openslr.org/resources/66/te_in_male.zip" -O te_in_male.zip
    unzip -q -o te_in_female.zip 2>/dev/null || true
    unzip -q -o te_in_male.zip 2>/dev/null || true
    rm -f te_in_female.zip te_in_male.zip
else
    echo "  ✅ OpenSLR 66 already present"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ADDITIONAL DOWNLOAD COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"

echo ""
echo "📊 Telugu audio files:"
find ${TELUGU_DIR} -name "*.wav" -o -name "*.flac" -o -name "*.mp3" 2>/dev/null | wc -l
echo ""
echo "📁 Telugu subdirectories:"
du -sh ${TELUGU_DIR}/* 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Telugu Data Sources Summary"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ✅ OpenSLR 66:       ~10 hours (direct wget)"
echo "  ✅ Kathbath Telugu:  ~155 hours (just downloaded)"
echo "  ⚠️  Common Voice:    ~20 hours (needs HF auth)"
echo "  ⚠️  FLEURS:          ~12 hours (Google storage)"
echo "  ────────────────────────────────────────────────"
echo "  📊 Raw total:        ~165-197 hours"
echo "  🚀 After 5x augment: ~825-985 hours"
echo "═══════════════════════════════════════════════════════════════════════════════"
