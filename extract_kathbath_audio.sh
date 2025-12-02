#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#   EXTRACT AND CONVERT KATHBATH AUDIO
#   
#   Extracts Hindi and Telugu audio from Kathbath raw data
#   Converts M4A to WAV (16kHz mono)
# ═══════════════════════════════════════════════════════════════════════════════

set -e

KATHBATH_RAW="/workspace/data/kathbath_raw/kb_data_clean_m4a"
DATA_DIR="/workspace/data"
HINDI_OUT="${DATA_DIR}/hindi/kathbath"
TELUGU_OUT="${DATA_DIR}/telugu/kathbath"

# Number of parallel processes for conversion
NPROC=8

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  EXTRACTING KATHBATH HINDI & TELUGU AUDIO"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Source: ${KATHBATH_RAW}"
echo "  Hindi output: ${HINDI_OUT}"
echo "  Telugu output: ${TELUGU_OUT}"
echo "═══════════════════════════════════════════════════════════════════════════════"

# Check source exists
if [ ! -d "$KATHBATH_RAW" ]; then
    echo "❌ Kathbath raw data not found at: $KATHBATH_RAW"
    echo "   Looking for data..."
    find /workspace/data -name "kb_data*" -type d 2>/dev/null || true
    exit 1
fi

# Show structure
echo ""
echo "📁 Kathbath directory structure:"
ls -la "$KATHBATH_RAW" 2>/dev/null | head -20

# Create output directories
mkdir -p "$HINDI_OUT"
mkdir -p "$TELUGU_OUT"

# ─────────────────────────────────────────────────────────────────────────────────
# EXTRACT HINDI
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  PROCESSING HINDI"
echo "─────────────────────────────────────────────────────────────────────────────"

HINDI_SRC="${KATHBATH_RAW}/hindi"
if [ -d "$HINDI_SRC" ]; then
    M4A_COUNT=$(find "$HINDI_SRC" -name "*.m4a" 2>/dev/null | wc -l)
    echo "  📁 Found Hindi directory with $M4A_COUNT m4a files"
    
    if [ "$M4A_COUNT" -gt 0 ]; then
        echo "  🔄 Converting m4a to wav (16kHz mono) using $NPROC processes..."
        echo "     This may take 30-60 minutes..."
        
        # Convert using parallel processing
        find "$HINDI_SRC" -name "*.m4a" -print0 | \
            xargs -0 -P $NPROC -I {} bash -c '
                input="{}"
                filename=$(basename "$input" .m4a)
                output="'"$HINDI_OUT"'/${filename}.wav"
                if [ ! -f "$output" ]; then
                    ffmpeg -y -loglevel error -i "$input" -ar 16000 -ac 1 "$output" 2>/dev/null
                fi
            '
        
        CONVERTED=$(find "$HINDI_OUT" -name "*.wav" | wc -l)
        echo "  ✅ Hindi: Converted $CONVERTED files to ${HINDI_OUT}"
    fi
else
    echo "  ⚠️ Hindi directory not found at: $HINDI_SRC"
    echo "  Looking for hindi data..."
    find "$KATHBATH_RAW" -type d -iname "*hindi*" 2>/dev/null || true
fi

# ─────────────────────────────────────────────────────────────────────────────────
# EXTRACT TELUGU
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  PROCESSING TELUGU"
echo "─────────────────────────────────────────────────────────────────────────────"

TELUGU_SRC="${KATHBATH_RAW}/telugu"
if [ -d "$TELUGU_SRC" ]; then
    M4A_COUNT=$(find "$TELUGU_SRC" -name "*.m4a" 2>/dev/null | wc -l)
    echo "  📁 Found Telugu directory with $M4A_COUNT m4a files"
    
    if [ "$M4A_COUNT" -gt 0 ]; then
        echo "  🔄 Converting m4a to wav (16kHz mono) using $NPROC processes..."
        echo "     This may take 30-60 minutes..."
        
        # Convert using parallel processing
        find "$TELUGU_SRC" -name "*.m4a" -print0 | \
            xargs -0 -P $NPROC -I {} bash -c '
                input="{}"
                filename=$(basename "$input" .m4a)
                output="'"$TELUGU_OUT"'/${filename}.wav"
                if [ ! -f "$output" ]; then
                    ffmpeg -y -loglevel error -i "$input" -ar 16000 -ac 1 "$output" 2>/dev/null
                fi
            '
        
        CONVERTED=$(find "$TELUGU_OUT" -name "*.wav" | wc -l)
        echo "  ✅ Telugu: Converted $CONVERTED files to ${TELUGU_OUT}"
    fi
else
    echo "  ⚠️ Telugu directory not found at: $TELUGU_SRC"
    echo "  Looking for telugu data..."
    find "$KATHBATH_RAW" -type d -iname "*telugu*" 2>/dev/null || true
fi

# ─────────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  EXTRACTION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"

echo ""
echo "📊 Final audio file counts:"
echo "  English: $(find ${DATA_DIR}/english -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l) files"
echo "  Hindi:   $(find ${DATA_DIR}/hindi -name "*.wav" 2>/dev/null | wc -l) files"
echo "  Telugu:  $(find ${DATA_DIR}/telugu -name "*.wav" 2>/dev/null | wc -l) files"

echo ""
echo "💾 Disk usage:"
du -sh ${DATA_DIR}/english 2>/dev/null || true
du -sh ${DATA_DIR}/hindi 2>/dev/null || true
du -sh ${DATA_DIR}/telugu 2>/dev/null || true

# Calculate hours (rough estimate: 10 files ≈ 1 minute for short utterances)
TELUGU_FILES=$(find ${DATA_DIR}/telugu -name "*.wav" 2>/dev/null | wc -l)
TELUGU_HOURS_EST=$(echo "scale=1; $TELUGU_FILES / 600" | bc 2>/dev/null || echo "N/A")
echo ""
echo "📈 Estimated Telugu hours: ~${TELUGU_HOURS_EST}h (will verify with check_data_status.py)"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  1. Download more Telugu: bash download_more_telugu.sh"
echo "  2. Check exact hours: python check_data_status.py"
echo "  3. Augment Telugu 5x: python augment_telugu_data.py --expansion 5"
echo "  4. Train codec: python train_codec_production.py"
echo "═══════════════════════════════════════════════════════════════════════════════"
