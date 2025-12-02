#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#   INDICVOICES-R DIRECT DOWNLOAD SCRIPT
#   
#   ╔═══════════════════════════════════════════════════════════════════════════╗
#   ║  VERIFIED WORKING LINKS (from official AI4Bharat GitHub repo)             ║
#   ║  Source: https://github.com/AI4Bharat/IndicVoices-R/blob/master/data_links.txt
#   ╚═══════════════════════════════════════════════════════════════════════════╝
#   
#   This script downloads Hindi and Telugu data directly via wget.
#   NO HUGGINGFACE LIBRARY REQUIRED - bypasses all OOM issues!
#   
#   Run on RunPod: bash download_indicvoices_direct.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Configuration
DATA_DIR="/workspace/data"
HINDI_DIR="${DATA_DIR}/hindi/indicvoices_r"
TELUGU_DIR="${DATA_DIR}/telugu/indicvoices_r"

# VERIFIED URLs from official GitHub repo
HINDI_URL="https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Hindi.tar.gz"
TELUGU_URL="https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Telugu.tar.gz"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  INDICVOICES-R DIRECT DOWNLOAD"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Source: Official AI4Bharat GitHub repo"
echo "  Data: ~1700 hours across 22 languages (we download Hindi + Telugu)"
echo "═══════════════════════════════════════════════════════════════════════════════"

# Create directories
mkdir -p "$HINDI_DIR"
mkdir -p "$TELUGU_DIR"

# ─────────────────────────────────────────────────────────────────────────────────
# HINDI
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  DOWNLOADING HINDI"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check if already downloaded
HINDI_COUNT=$(find "$HINDI_DIR" -name "*.wav" 2>/dev/null | wc -l)
if [ "$HINDI_COUNT" -gt 10000 ]; then
    echo "  ✅ Hindi already has $HINDI_COUNT files, skipping..."
else
    echo "  📥 URL: $HINDI_URL"
    echo "  📁 Destination: $HINDI_DIR"
    echo ""
    
    cd "$HINDI_DIR"
    
    # Download with progress
    echo "  Downloading Hindi.tar.gz..."
    wget -c --progress=bar:force "$HINDI_URL" -O Hindi.tar.gz 2>&1
    
    # Check download
    if [ -f "Hindi.tar.gz" ]; then
        FILE_SIZE=$(stat -c%s "Hindi.tar.gz" 2>/dev/null || stat -f%z "Hindi.tar.gz" 2>/dev/null)
        echo "  Downloaded: $(numfmt --to=iec $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes")"
        
        # Extract
        echo "  📦 Extracting..."
        tar -xzf Hindi.tar.gz
        
        # Cleanup
        rm -f Hindi.tar.gz
        
        # Count files
        HINDI_COUNT=$(find . -name "*.wav" | wc -l)
        echo "  ✅ Hindi: $HINDI_COUNT audio files"
    else
        echo "  ❌ Download failed!"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────────
# TELUGU
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  DOWNLOADING TELUGU"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check if already downloaded
TELUGU_COUNT=$(find "$TELUGU_DIR" -name "*.wav" 2>/dev/null | wc -l)
if [ "$TELUGU_COUNT" -gt 5000 ]; then
    echo "  ✅ Telugu already has $TELUGU_COUNT files, skipping..."
else
    echo "  📥 URL: $TELUGU_URL"
    echo "  📁 Destination: $TELUGU_DIR"
    echo ""
    
    cd "$TELUGU_DIR"
    
    # Download with progress
    echo "  Downloading Telugu.tar.gz..."
    wget -c --progress=bar:force "$TELUGU_URL" -O Telugu.tar.gz 2>&1
    
    # Check download
    if [ -f "Telugu.tar.gz" ]; then
        FILE_SIZE=$(stat -c%s "Telugu.tar.gz" 2>/dev/null || stat -f%z "Telugu.tar.gz" 2>/dev/null)
        echo "  Downloaded: $(numfmt --to=iec $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes")"
        
        # Extract
        echo "  📦 Extracting..."
        tar -xzf Telugu.tar.gz
        
        # Cleanup
        rm -f Telugu.tar.gz
        
        # Count files
        TELUGU_COUNT=$(find . -name "*.wav" | wc -l)
        echo "  ✅ Telugu: $TELUGU_COUNT audio files"
    else
        echo "  ❌ Download failed!"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  DOWNLOAD COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"

# Final counts
echo ""
echo "📊 Final audio file counts:"
echo "  English: $(find ${DATA_DIR}/english -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l) files"
echo "  Hindi:   $(find ${DATA_DIR}/hindi -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l) files"
echo "  Telugu:  $(find ${DATA_DIR}/telugu -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l) files"

TOTAL=$(find ${DATA_DIR} -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l)
echo ""
echo "  TOTAL: $TOTAL files"

# Disk usage
echo ""
echo "💾 Disk usage:"
du -sh ${DATA_DIR}/* 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  1. Check status:  python check_data_status.py"
echo "  2. Augment data:  python augment_all_data.py"
echo "  3. Train codec:   python train_codec_production.py \\"
echo "       --data_dirs /workspace/data/english /workspace/data/hindi /workspace/data/telugu"
echo "═══════════════════════════════════════════════════════════════════════════════"
