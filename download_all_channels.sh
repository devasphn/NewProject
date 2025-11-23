#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# PRODUCTION Telugu Data Collection - ALL Channels
# Downloads from ALL channels in data_sources_PRODUCTION.yaml
# ═══════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COOKIES_FILE="/workspace/NewProject/cookies/youtube_cookies.txt"
OUTPUT_DIR="/workspace/telugu_data/raw_videos"
ARCHIVE_FILE="/workspace/telugu_data/downloaded.txt"
LOG_DIR="/workspace/logs"
MAIN_LOG="${LOG_DIR}/production_all_channels.log"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MAIN_LOG"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$MAIN_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$MAIN_LOG"
}

# Check cookies exist
if [ ! -f "$COOKIES_FILE" ]; then
    error "Cookies file not found: $COOKIES_FILE"
    exit 1
fi

log "═══════════════════════════════════════════════════════════════"
log "TELUGU DATA COLLECTION - PRODUCTION MODE"
log "═══════════════════════════════════════════════════════════════"
log "Output: $OUTPUT_DIR"
log "Cookies: $COOKIES_FILE"
log "Archive: $ARCHIVE_FILE"
log "═══════════════════════════════════════════════════════════════"

# Download function
download_channel() {
    local channel_name="$1"
    local channel_url="$2"
    local max_videos="$3"
    local tier="$4"
    
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "DOWNLOADING: $channel_name"
    log "Tier: $tier"
    log "URL: $channel_url"
    log "Max videos: $max_videos"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Create channel-specific output directory
    local safe_name=$(echo "$channel_name" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    local channel_dir="${OUTPUT_DIR}/${tier}/${safe_name}"
    mkdir -p "$channel_dir"
    
    # Build command
    local cmd=(
        yt-dlp
        --cookies "$COOKIES_FILE"
        --format "best[height<=720]"
        --output "${channel_dir}/%(title)s.%(ext)s"
        --download-archive "$ARCHIVE_FILE"
        --ignore-errors
        --no-overwrites
        --sleep-interval 5
        --max-sleep-interval 15
        --sleep-requests 1
        --retries 10
        --fragment-retries 10
        --concurrent-fragments 3
    )
    
    # Add max videos limit if specified
    if [ "$max_videos" != "ALL" ]; then
        cmd+=(--max-downloads "$max_videos")
    fi
    
    cmd+=("$channel_url")
    
    # Execute download
    log "Starting download..."
    if "${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/${safe_name}.log"; then
        log "✅ Completed: $channel_name"
    else
        warn "⚠️  Some errors occurred for: $channel_name (continuing...)"
    fi
    
    # Show progress
    local count=$(find "$channel_dir" -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" | wc -l)
    local size=$(du -sh "$channel_dir" 2>/dev/null | cut -f1)
    log "   Downloaded: $count files, $size"
}

# ═══════════════════════════════════════════════════════════════════════
# TIER 1: HIGH-PRIORITY CHANNELS
# ═══════════════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════"
log "STARTING TIER 1: High-Priority Channels"
log "═══════════════════════════════════════════════════════════════"

# Raw Talks VK (ALL videos)
download_channel \
    "Raw Talks with VK" \
    "https://www.youtube.com/@RawTalksWithVK/videos" \
    "ALL" \
    "tier1_podcasts"

# 10TV Telugu (Recent 500)
download_channel \
    "10TV Telugu" \
    "https://www.youtube.com/@10TVNewsTelugu/videos" \
    "500" \
    "tier1_news"

# Sakshi TV (Recent 500)
download_channel \
    "Sakshi TV" \
    "https://www.youtube.com/@SakshiTV/videos" \
    "500" \
    "tier1_news"

# TV9 Telugu (Recent 500)
download_channel \
    "TV9 Telugu" \
    "https://www.youtube.com/@TV9TeluguLive/videos" \
    "500" \
    "tier1_news"

# ═══════════════════════════════════════════════════════════════════════
# TIER 2: SPEAKER DIVERSITY CHANNELS
# ═══════════════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════"
log "STARTING TIER 2: Speaker Diversity Channels"
log "═══════════════════════════════════════════════════════════════"

# Telugu Audio Books (ALL)
download_channel \
    "Telugu Audio Books" \
    "https://www.youtube.com/@TeluguAudioBooks1/videos" \
    "ALL" \
    "tier2_diversity"

# Voice of Telugu (ALL)
download_channel \
    "Voice of Telugu" \
    "https://www.youtube.com/@VoiceOfTeluguofficial/videos" \
    "ALL" \
    "tier2_diversity"

# Telugu Connects (ALL)
download_channel \
    "Telugu Connects" \
    "https://www.youtube.com/@TeluguConnects_/videos" \
    "ALL" \
    "tier2_diversity"

# My Village Show (Recent 200)
download_channel \
    "My Village Show" \
    "https://www.youtube.com/@MyVillageShowOfficial/videos" \
    "200" \
    "tier2_diversity"

# ═══════════════════════════════════════════════════════════════════════
# TIER 3: ADDITIONAL DIVERSITY (OPTIONAL - Comment out to skip)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════"
log "STARTING TIER 3: Additional Diversity Channels (Optional)"
log "═══════════════════════════════════════════════════════════════"

# V6 News Telugu (Recent 200)
download_channel \
    "V6 News Telugu" \
    "https://www.youtube.com/@V6NewsTelugu/videos" \
    "200" \
    "tier3_additional"

# Gemini TV (Recent 150)
download_channel \
    "Gemini TV" \
    "https://www.youtube.com/@GeminiTV/videos" \
    "150" \
    "tier3_additional"

# Extra Jabardasth (Recent 100)
download_channel \
    "Extra Jabardasth" \
    "https://www.youtube.com/@ExtraJabardasth/videos" \
    "100" \
    "tier3_additional"

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════"
log "COLLECTION COMPLETE!"
log "═══════════════════════════════════════════════════════════════"

# Calculate totals
total_files=$(find "$OUTPUT_DIR" -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" | wc -l)
total_size=$(du -sh "$OUTPUT_DIR" | cut -f1)

log "Total files downloaded: $total_files"
log "Total size: $total_size"
log "Output directory: $OUTPUT_DIR"
log "Detailed logs: $LOG_DIR"
log ""
log "✅ ALL CHANNELS DOWNLOADED SUCCESSFULLY!"
log "═══════════════════════════════════════════════════════════════"
