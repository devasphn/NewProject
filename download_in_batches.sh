#!/bin/bash
# Batch download script to avoid YouTube rate limiting
# Downloads in small batches with 2-hour breaks

set -e

echo "========================================="
echo "BATCH DOWNLOAD - ANTI RATE-LIMIT MODE"
echo "========================================="
echo ""
echo "This script downloads in small batches:"
echo "  - Download 50 videos"
echo "  - Wait 2 hours"
echo "  - Repeat"
echo ""
echo "Timeline: ~10-14 days for complete collection"
echo "Advantage: ZERO rate limiting"
echo ""
echo "Press Ctrl+C to stop anytime (progress saved)"
echo ""

# Configuration
BATCH_SIZE=50
WAIT_HOURS=2
MAX_BATCHES=40  # ~2000 videos total

OUTPUT_DIR="/workspace/telugu_data_production"
CONFIG_FILE="data_sources_PRODUCTION.yaml"

# Counter
batch_num=1

echo "Starting batch downloads..."
echo ""

while [ $batch_num -le $MAX_BATCHES ]; do
    echo "========================================"
    echo "BATCH $batch_num of $MAX_BATCHES"
    echo "========================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Run download with max-downloads limit
    python download_telugu_data_PRODUCTION.py \
        --config $CONFIG_FILE \
        --output $OUTPUT_DIR \
        --max-downloads-per-channel $BATCH_SIZE
    
    # Check if we should continue
    if [ $batch_num -lt $MAX_BATCHES ]; then
        echo ""
        echo "========================================"
        echo "BATCH $batch_num COMPLETE"
        echo "========================================"
        echo ""
        echo "Waiting $WAIT_HOURS hours before next batch..."
        echo "This avoids YouTube rate limiting"
        echo ""
        echo "Next batch starts at: $(date -d "+$WAIT_HOURS hours" '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "You can press Ctrl+C to stop"
        echo "Progress is saved - you can resume later"
        echo ""
        
        # Wait
        sleep $((WAIT_HOURS * 3600))
        
        batch_num=$((batch_num + 1))
    else
        echo ""
        echo "========================================"
        echo "ALL BATCHES COMPLETE!"
        echo "========================================"
        echo ""
        break
    fi
done

echo ""
echo "Collection complete!"
echo ""
echo "Next steps:"
echo "  1. Extract audio: python download_telugu_data_PRODUCTION.py --skip-download"
echo "  2. Process audio: python process_audio.py"
echo "  3. Train codec: python train_codec_dac.py"
echo ""
