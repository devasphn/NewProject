#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SINGLE CHANNEL DOWNLOAD - Test one channel at a time
# Usage: bash download_single_channel.sh <channel_name>
# Examples:
#   bash download_single_channel.sh rawtalksvk
#   bash download_single_channel.sh 10tv
#   bash download_single_channel.sh sakshitv
#   bash download_single_channel.sh tv9
# ═══════════════════════════════════════════════════════════════════════

COOKIES="/workspace/NewProject/cookies/youtube_cookies.txt"
OUTPUT="/workspace/telugu_data/raw_videos"
ARCHIVE="/workspace/telugu_data/downloaded.txt"

# Common options
YT_OPTS=(
    --cookies "$COOKIES"
    --format "best[height<=720]"
    --download-archive "$ARCHIVE"
    --ignore-errors
    --no-overwrites
    --sleep-interval 5
    --max-sleep-interval 15
    --sleep-requests 1
    --retries 10
    --fragment-retries 10
    --concurrent-fragments 3
)

# Channel selection
case "$1" in
    rawtalksvk|raw|vk)
        echo "━━━ Raw Talks with VK ━━━"
        mkdir -p "$OUTPUT/tier1_podcasts/RawTalksVK"
        yt-dlp "${YT_OPTS[@]}" \
            --output "$OUTPUT/tier1_podcasts/RawTalksVK/%(title)s.%(ext)s" \
            "https://www.youtube.com/@RawTalksWithVK/videos"
        ;;
        
    10tv|10TV)
        echo "━━━ 10TV Telugu (First 500) ━━━"
        mkdir -p "$OUTPUT/tier1_news/10TV"
        yt-dlp "${YT_OPTS[@]}" \
            --playlist-items 1-500 \
            --output "$OUTPUT/tier1_news/10TV/%(title)s.%(ext)s" \
            "https://www.youtube.com/@10TVNewsTelugu/videos"
        ;;
        
    sakshi|sakshitv|SakshiTV)
        echo "━━━ Sakshi TV (First 500) ━━━"
        mkdir -p "$OUTPUT/tier1_news/SakshiTV"
        yt-dlp "${YT_OPTS[@]}" \
            --playlist-items 1-500 \
            --output "$OUTPUT/tier1_news/SakshiTV/%(title)s.%(ext)s" \
            "https://www.youtube.com/@SakshiTV/videos"
        ;;
        
    tv9|TV9)
        echo "━━━ TV9 Telugu (First 500) ━━━"
        mkdir -p "$OUTPUT/tier1_news/TV9"
        yt-dlp "${YT_OPTS[@]}" \
            --playlist-items 1-500 \
            --output "$OUTPUT/tier1_news/TV9/%(title)s.%(ext)s" \
            "https://www.youtube.com/@TV9TeluguLive/videos"
        ;;
        
    *)
        echo "Usage: $0 <channel>"
        echo ""
        echo "Available channels:"
        echo "  rawtalksvk  - Raw Talks with VK (ALL videos)"
        echo "  10tv        - 10TV Telugu (500 videos)"
        echo "  sakshitv    - Sakshi TV (500 videos)"
        echo "  tv9         - TV9 Telugu (500 videos)"
        exit 1
        ;;
esac

# Show results
echo ""
echo "✅ Download complete!"
echo "Total videos in all channels: $(find $OUTPUT -name '*.mp4' | wc -l)"
echo "Total size: $(du -sh $OUTPUT | cut -f1)"
