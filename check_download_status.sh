#!/bin/bash
# Quick status checker for Telugu data collection

echo "========================================================================"
echo "TELUGU DATA COLLECTION - STATUS CHECK"
echo "========================================================================"
echo ""

DATA_DIR="/workspace/telugu_data_production"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Collection not started yet"
    echo ""
    echo "To start:"
    echo "  python download_telugu_data_PRODUCTION.py \\"
    echo "      --config data_sources_PRODUCTION.yaml \\"
    echo "      --output /workspace/telugu_data_production"
    echo ""
    exit 1
fi

# Count videos
echo "📊 DOWNLOAD STATISTICS"
echo "------------------------------------------------------------------------"

if [ -d "$DATA_DIR/raw_videos" ]; then
    video_count=$(find "$DATA_DIR/raw_videos" -type f \( -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    echo "Videos downloaded: $video_count"
    
    if [ $video_count -gt 0 ]; then
        video_size=$(du -sh "$DATA_DIR/raw_videos" 2>/dev/null | cut -f1)
        echo "Video storage used: $video_size"
        
        # Estimate hours (rough: 500MB per hour of 720p video)
        video_size_gb=$(du -sb "$DATA_DIR/raw_videos" 2>/dev/null | cut -f1)
        if [ ! -z "$video_size_gb" ]; then
            video_hours=$((video_size_gb / 1024 / 1024 / 1024 / 1 * 2))
            echo "Estimated video hours: ~$video_hours hours"
        fi
    fi
else
    echo "Videos downloaded: 0"
fi

echo ""

# Count audio files
if [ -d "$DATA_DIR/raw_audio" ]; then
    audio_count=$(find "$DATA_DIR/raw_audio" -type f -name "*.wav" 2>/dev/null | wc -l)
    echo "Audio files extracted: $audio_count"
    
    if [ $audio_count -gt 0 ]; then
        audio_size=$(du -sh "$DATA_DIR/raw_audio" 2>/dev/null | cut -f1)
        echo "Audio storage used: $audio_size"
        
        # Estimate hours (30MB per hour of 16kHz mono)
        audio_size_mb=$(du -sm "$DATA_DIR/raw_audio" 2>/dev/null | cut -f1)
        if [ ! -z "$audio_size_mb" ]; then
            audio_hours=$((audio_size_mb / 30))
            echo "Estimated audio hours: ~$audio_hours hours"
        fi
    fi
else
    echo "Audio files extracted: 0"
fi

echo ""
echo "------------------------------------------------------------------------"

# Show collection stats if exists
if [ -f "$DATA_DIR/collection_stats.json" ]; then
    echo ""
    echo "📈 COLLECTION PROGRESS"
    echo "------------------------------------------------------------------------"
    cat "$DATA_DIR/collection_stats.json"
    echo "------------------------------------------------------------------------"
fi

echo ""

# Disk space
echo "💾 DISK SPACE"
echo "------------------------------------------------------------------------"
df -h /workspace | grep -E "Filesystem|/workspace"
echo "------------------------------------------------------------------------"

echo ""

# Check if download is running
echo "🔄 DOWNLOAD STATUS"
echo "------------------------------------------------------------------------"
if pgrep -f "download_telugu_data_PRODUCTION.py" > /dev/null; then
    echo "✅ Download is RUNNING"
    echo ""
    echo "Monitor progress with:"
    echo "  tail -f $DATA_DIR/collection_log_*.txt"
else
    echo "⏸️  Download is STOPPED"
    echo ""
    echo "To resume:"
    echo "  python download_telugu_data_PRODUCTION.py \\"
    echo "      --config data_sources_PRODUCTION.yaml \\"
    echo "      --output /workspace/telugu_data_production"
fi
echo "------------------------------------------------------------------------"

echo ""

# Progress estimate
if [ ! -z "$video_count" ] && [ $video_count -gt 0 ]; then
    echo "🎯 PROGRESS ESTIMATE"
    echo "------------------------------------------------------------------------"
    target_videos=1500
    progress=$((video_count * 100 / target_videos))
    remaining=$((target_videos - video_count))
    
    echo "Current: $video_count videos"
    echo "Target: $target_videos videos"
    echo "Progress: $progress%"
    echo "Remaining: $remaining videos"
    echo ""
    
    if [ $video_count -lt 200 ]; then
        quality="Too small - keep collecting!"
    elif [ $video_count -lt 500 ]; then
        quality="Minimal viable - acceptable quality"
    elif [ $video_count -lt 1000 ]; then
        quality="Good - production quality achievable"
    else
        quality="Excellent - high production quality!"
    fi
    
    echo "Quality prediction: $quality"
    echo "------------------------------------------------------------------------"
fi

echo ""
echo "========================================================================"
echo ""
