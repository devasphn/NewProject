#!/bin/bash
# Quick audio extraction from downloaded videos

echo "========================================================================"
echo "EXTRACTING AUDIO FROM 232 VIDEOS"
echo "========================================================================"
echo ""

VIDEO_DIR="/workspace/telugu_data_production/raw_videos"
AUDIO_DIR="/workspace/telugu_data_production/raw_audio"

# Create audio directory
mkdir -p "$AUDIO_DIR"

# Count videos
video_count=$(find "$VIDEO_DIR" -type f \( -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" \) | wc -l)

echo "Found $video_count videos"
echo "Extracting to: $AUDIO_DIR"
echo ""
echo "This will take ~2-3 hours"
echo "Press Ctrl+C to cancel..."
echo ""

sleep 3

# Extract audio
count=0
find "$VIDEO_DIR" -type f \( -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" \) | while read video; do
    count=$((count + 1))
    
    # Get relative path
    rel_path="${video#$VIDEO_DIR/}"
    
    # Create output path
    audio_file="$AUDIO_DIR/${rel_path%.*}.wav"
    
    # Create directory
    mkdir -p "$(dirname "$audio_file")"
    
    # Skip if already exists
    if [ -f "$audio_file" ]; then
        echo "[$count/$video_count] SKIP: ${rel_path%.*}.wav (already exists)"
        continue
    fi
    
    # Extract audio
    ffmpeg -i "$video" \
        -ar 16000 \
        -ac 1 \
        -c:a pcm_s16le \
        -y \
        "$audio_file" \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$count/$video_count] OK: ${rel_path%.*}.wav"
    else
        echo "[$count/$video_count] ERROR: ${rel_path%.*}.wav"
    fi
done

echo ""
echo "========================================================================"
echo "AUDIO EXTRACTION COMPLETE!"
echo "========================================================================"
echo ""

# Count extracted files
audio_count=$(find "$AUDIO_DIR" -type f -name "*.wav" | wc -l)
audio_size=$(du -sh "$AUDIO_DIR" | cut -f1)

echo "Extracted: $audio_count audio files"
echo "Size: $audio_size"
echo ""

# Estimate hours (30MB per hour at 16kHz mono)
audio_size_mb=$(du -sm "$AUDIO_DIR" | cut -f1)
hours=$((audio_size_mb / 30))

echo "Estimated hours: ~$hours hours"
echo ""
echo "Next step:"
echo "  python prepare_speaker_data.py \\"
echo "      --data_dir $AUDIO_DIR \\"
echo "      --output_dir /workspace/telugu_poc_data \\"
echo "      --no_balance"
echo ""
