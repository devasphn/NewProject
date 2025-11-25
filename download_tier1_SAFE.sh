#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# TIER 1 SAFE MODE - Longer sleep intervals to avoid rate limiting
# Downloads: Raw Talks VK + 3 News Channels
# ═══════════════════════════════════════════════════════════════════════

COOKIES="/workspace/NewProject/cookies/youtube_cookies.txt"
OUTPUT="/workspace/telugu_data/raw_videos"
ARCHIVE="/workspace/telugu_data/downloaded.txt"
LOG_DIR="/workspace/logs"

echo "═══════════════════════════════════════════════════════════════"
echo "TIER 1 SAFE MODE - ANTI-RATE-LIMIT PROTECTION"
echo "═══════════════════════════════════════════════════════════════"
echo "Sleep: 15-45 seconds between videos"
echo "Cookies: $COOKIES"
echo "Output: $OUTPUT"
echo "═══════════════════════════════════════════════════════════════"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not installed!"
    echo "Run: curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
    echo "Then: apt-get install -y nodejs"
    exit 1
fi

echo "✓ Node.js $(node --version) detected"

# Check cookies
if [ ! -f "$COOKIES" ]; then
    echo "ERROR: Cookies file not found at $COOKIES"
    exit 1
fi

echo "✓ Cookies file found"
echo ""

# Common yt-dlp options - SAFE MODE
YT_OPTS=(
    --cookies "$COOKIES"
    --format "best[height<=720]"
    --download-archive "$ARCHIVE"
    --ignore-errors
    --no-overwrites
    --sleep-interval 15              # MIN 15 seconds between videos
    --max-sleep-interval 45          # MAX 45 seconds between videos
    --sleep-requests 2               # 2 seconds between API requests
    --retries 10
    --fragment-retries 10
    --concurrent-fragments 3
    --no-check-certificates
)

# ───────────────────────────────────────────────────────────
# Channel 1: Raw Talks VK (RESUME from where it failed)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 1/4: Raw Talks with VK (RESUME) ━━━"
echo "Videos: Remaining videos (will skip already downloaded)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_podcasts/RawTalksVK"

yt-dlp "${YT_OPTS[@]}" \
    --output "$OUTPUT/tier1_podcasts/RawTalksVK/%(title)s.%(ext)s" \
    "https://www.youtube.com/@RawTalksWithVK/videos" \
    2>&1 | tee -a "$LOG_DIR/01_RawTalksVK_SAFE.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_podcasts/RawTalksVK -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# Channel 2: 10TV Telugu (First 500 videos - SAFE MODE)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 2/4: 10TV Telugu ━━━"
echo "Videos: First 500 (with anti-rate-limit protection)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_news/10TV"

yt-dlp "${YT_OPTS[@]}" \
    --playlist-items 1-500 \
    --output "$OUTPUT/tier1_news/10TV/%(title)s.%(ext)s" \
    "https://www.youtube.com/@10TVNewsTelugu/videos" \
    2>&1 | tee "$LOG_DIR/02_10TV_SAFE.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_news/10TV -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# Channel 3: Sakshi TV (First 500 videos - SAFE MODE)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 3/4: Sakshi TV ━━━"
echo "Videos: First 500 (with anti-rate-limit protection)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_news/SakshiTV"

yt-dlp "${YT_OPTS[@]}" \
    --playlist-items 1-500 \
    --output "$OUTPUT/tier1_news/SakshiTV/%(title)s.%(ext)s" \
    "https://www.youtube.com/@SakshiTV/videos" \
    2>&1 | tee "$LOG_DIR/03_SakshiTV_SAFE.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_news/SakshiTV -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# Channel 4: TV9 Telugu (First 500 videos - SAFE MODE)
# ───────────────────────────────────────────────────────────
echo ""
echo "━━━ 4/4: TV9 Telugu ━━━"
echo "Videos: First 500 (with anti-rate-limit protection)"
echo "Started: $(date)"

mkdir -p "$OUTPUT/tier1_news/TV9"

yt-dlp "${YT_OPTS[@]}" \
    --playlist-items 1-500 \
    --output "$OUTPUT/tier1_news/TV9/%(title)s.%(ext)s" \
    "https://www.youtube.com/@TV9TeluguLive/videos" \
    2>&1 | tee "$LOG_DIR/04_TV9_SAFE.log"

echo "Completed: $(date)"
echo "Downloaded: $(find $OUTPUT/tier1_news/TV9 -name '*.mp4' | wc -l) videos"
echo ""

# ───────────────────────────────────────────────────────────
# FINAL SUMMARY
# ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ TIER 1 SAFE MODE COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo "Raw Talks VK: $(find $OUTPUT/tier1_podcasts/RawTalksVK -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "10TV Telugu:  $(find $OUTPUT/tier1_news/10TV -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "Sakshi TV:    $(find $OUTPUT/tier1_news/SakshiTV -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "TV9 Telugu:   $(find $OUTPUT/tier1_news/TV9 -name '*.mp4' 2>/dev/null | wc -l) videos"
echo ""
echo "Total videos: $(find $OUTPUT -name '*.mp4' | wc -l)"
echo "Total size:   $(du -sh $OUTPUT | cut -f1)"
echo "═══════════════════════════════════════════════════════════════"
