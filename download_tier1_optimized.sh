#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# TIER 1 OPTIMIZED - Uses --playlist-items (no slow enumeration)
# Downloads: Raw Talks VK + 3 News Channels
# ═══════════════════════════════════════════════════════════════════════

COOKIES="/workspace/NewProject/cookies/youtube_cookies.txt"
OUTPUT="/workspace/telugu_data/raw_videos"
ARCHIVE="/workspace/telugu_data/downloaded.txt"
LOG_DIR="/workspace/logs"

echo "═══════════════════════════════════════════════════════════════"
echo "TIER 1 OPTIMIZED DOWNLOAD"
echo "═══════════════════════════════════════════════════════════════"
echo "Using --playlist-items (fast, no enumeration)"
echo "Cookies: $COOKIES"
echo "Output: $OUTPUT"
echo "═══════════════════════════════════════════════════════════════"

# Common yt-dlp options
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

# ───────────────────────────────────────────────────────────
# Channel 1: Raw Talks VK (ALL videos - small channel)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 1/4: Raw Talks with VK ━━━"
echo "Videos: ALL (~113 videos)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_podcasts/RawTalksVK"

yt-dlp "${YT_OPTS[@]}" \
    --output "$OUTPUT/tier1_podcasts/RawTalksVK/%(title)s.%(ext)s" \
    "https://www.youtube.com/@RawTalksWithVK/videos" \
    2>&1 | tee "$LOG_DIR/01_RawTalksVK.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_podcasts/RawTalksVK -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# Channel 2: 10TV Telugu (First 500 videos - FAST)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 2/4: 10TV Telugu ━━━"
echo "Videos: First 500 (using playlist-items)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_news/10TV"

yt-dlp "${YT_OPTS[@]}" \
    --playlist-items 1-500 \
    --output "$OUTPUT/tier1_news/10TV/%(title)s.%(ext)s" \
    "https://www.youtube.com/@10TVNewsTelugu/videos" \
    2>&1 | tee "$LOG_DIR/02_10TV.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_news/10TV -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# Channel 3: Sakshi TV (First 500 videos - FAST)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 3/4: Sakshi TV ━━━"
echo "Videos: First 500 (using playlist-items)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_news/SakshiTV"

yt-dlp "${YT_OPTS[@]}" \
    --playlist-items 1-500 \
    --output "$OUTPUT/tier1_news/SakshiTV/%(title)s.%(ext)s" \
    "https://www.youtube.com/@SakshiTV/videos" \
    2>&1 | tee "$LOG_DIR/03_SakshiTV.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_news/SakshiTV -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# Channel 4: TV9 Telugu (First 500 videos - FAST)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 4/4: TV9 Telugu ━━━"
echo "Videos: First 500 (using playlist-items)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_news/TV9"

yt-dlp "${YT_OPTS[@]}" \
    --playlist-items 1-500 \
    --output "$OUTPUT/tier1_news/TV9/%(title)s.%(ext)s" \
    "https://www.youtube.com/@TV9TeluguLive/videos" \
    2>&1 | tee "$LOG_DIR/04_TV9.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_news/TV9 -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# FINAL SUMMARY
# ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ TIER 1 DOWNLOAD COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo "Raw Talks VK: $(find $OUTPUT/tier1_podcasts/RawTalksVK -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "10TV Telugu:  $(find $OUTPUT/tier1_news/10TV -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "Sakshi TV:    $(find $OUTPUT/tier1_news/SakshiTV -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "TV9 Telugu:   $(find $OUTPUT/tier1_news/TV9 -name '*.mp4' 2>/dev/null | wc -l) videos"
echo ""
echo "Total videos: $(find $OUTPUT -name '*.mp4' | wc -l)"
echo "Total size:   $(du -sh $OUTPUT | cut -f1)"
echo "═══════════════════════════════════════════════════════════════"
