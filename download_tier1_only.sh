#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# TIER 1 ONLY - High Priority Channels (Fastest, Most Important)
# Downloads: Raw Talks VK + 3 News Channels
# Estimated: 800-1000 hours, ~250-300 GB, ~24-36 hours
# ═══════════════════════════════════════════════════════════════════════

COOKIES="/workspace/NewProject/cookies/youtube_cookies.txt"
OUTPUT="/workspace/telugu_data/raw_videos"
ARCHIVE="/workspace/telugu_data/downloaded.txt"

echo "═══════════════════════════════════════════════════════════════"
echo "TIER 1 DOWNLOAD - High Priority Channels Only"
echo "═══════════════════════════════════════════════════════════════"

# Channel 1: Raw Talks VK (ALL videos - ~113 videos, ~56 GB)
echo ""
echo "━━━ 1/4: Raw Talks with VK (ALL videos) ━━━"
yt-dlp \
    --cookies "$COOKIES" \
    --format "best[height<=720]" \
    --output "$OUTPUT/tier1_podcasts/RawTalksVK/%(title)s.%(ext)s" \
    --download-archive "$ARCHIVE" \
    --ignore-errors --no-overwrites \
    --sleep-interval 5 --max-sleep-interval 15 \
    --retries 10 --fragment-retries 10 \
    --concurrent-fragments 3 \
    "https://www.youtube.com/@RawTalksWithVK/videos"

# Channel 2: 10TV Telugu (500 videos - ~100 GB)
echo ""
echo "━━━ 2/4: 10TV Telugu (Recent 500 videos) ━━━"
yt-dlp \
    --cookies "$COOKIES" \
    --format "best[height<=720]" \
    --output "$OUTPUT/tier1_news/10TV/%(title)s.%(ext)s" \
    --download-archive "$ARCHIVE" \
    --max-downloads 500 \
    --ignore-errors --no-overwrites \
    --sleep-interval 5 --max-sleep-interval 15 \
    --retries 10 --fragment-retries 10 \
    --concurrent-fragments 3 \
    "https://www.youtube.com/@10TVNewsTelugu/videos"

# Channel 3: Sakshi TV (500 videos - ~100 GB)
echo ""
echo "━━━ 3/4: Sakshi TV (Recent 500 videos) ━━━"
yt-dlp \
    --cookies "$COOKIES" \
    --format "best[height<=720]" \
    --output "$OUTPUT/tier1_news/SakshiTV/%(title)s.%(ext)s" \
    --download-archive "$ARCHIVE" \
    --max-downloads 500 \
    --ignore-errors --no-overwrites \
    --sleep-interval 5 --max-sleep-interval 15 \
    --retries 10 --fragment-retries 10 \
    --concurrent-fragments 3 \
    "https://www.youtube.com/@SakshiTV/videos"

# Channel 4: TV9 Telugu (500 videos - ~100 GB)
echo ""
echo "━━━ 4/4: TV9 Telugu (Recent 500 videos) ━━━"
yt-dlp \
    --cookies "$COOKIES" \
    --format "best[height<=720]" \
    --output "$OUTPUT/tier1_news/TV9/%(title)s.%(ext)s" \
    --download-archive "$ARCHIVE" \
    --max-downloads 500 \
    --ignore-errors --no-overwrites \
    --sleep-interval 5 --max-sleep-interval 15 \
    --retries 10 --fragment-retries 10 \
    --concurrent-fragments 3 \
    "https://www.youtube.com/@TV9TeluguLive/videos"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ TIER 1 DOWNLOAD COMPLETE!"
echo "Total downloaded: $(find $OUTPUT -name '*.mp4' | wc -l) videos"
echo "Total size: $(du -sh $OUTPUT | cut -f1)"
echo "═══════════════════════════════════════════════════════════════"
