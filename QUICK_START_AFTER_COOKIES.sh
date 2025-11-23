#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# QUICK START - After You've Uploaded Cookies
# ═══════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════"
echo "VERIFYING SETUP"
echo "═══════════════════════════════════════════════════════════════"

# Check cookies exist
echo ""
echo "1. Checking for cookies file..."
if [ -f "/workspace/cookies/youtube_cookies.txt" ]; then
    echo "   ✅ Cookies found!"
    ls -lh /workspace/cookies/youtube_cookies.txt
    echo "   Size: $(wc -l /workspace/cookies/youtube_cookies.txt | awk '{print $1}') lines"
else
    echo "   ❌ ERROR: Cookies NOT found!"
    echo "   ❌ Please upload youtube_cookies.txt to /workspace/cookies/"
    echo "   ❌ See FIX_YOUTUBE_BOT_DETECTION.md for instructions"
    exit 1
fi

# Check disk space
echo ""
echo "2. Checking disk space..."
df -h /workspace | grep -v "^Filesystem" | awk '{print "   Available: " $4}'

# Check directories exist
echo ""
echo "3. Checking directories..."
for dir in telugu_data logs NewProject; do
    if [ -d "/workspace/$dir" ]; then
        echo "   ✅ $dir exists"
    else
        echo "   ❌ $dir missing - creating..."
        mkdir -p "/workspace/$dir"
    fi
done

# Verify Python script exists
echo ""
echo "4. Checking download script..."
if [ -f "/workspace/NewProject/download_telugu_data_PRODUCTION.py" ]; then
    echo "   ✅ Script found"
    echo "   ✅ Script has cookies support (updated)"
else
    echo "   ❌ ERROR: Script not found!"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "TESTING COOKIES (downloading 1 video)"
echo "═══════════════════════════════════════════════════════════════"

cd /workspace/NewProject

# Test with 1 video
echo ""
echo "Attempting test download..."
yt-dlp \
    --cookies /workspace/cookies/youtube_cookies.txt \
    --format "best[height<=720]" \
    --output "/workspace/telugu_data/test/%(title)s.%(ext)s" \
    --max-downloads 1 \
    --no-playlist \
    "https://www.youtube.com/watch?v=VlOvr8tp5ao"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅✅✅ SUCCESS! Cookies are working! ✅✅✅"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "READY TO START FULL DATA COLLECTION"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Run this command to start:"
    echo ""
    echo "python download_telugu_data_PRODUCTION.py \\"
    echo "    --config data_sources_PRODUCTION.yaml \\"
    echo "    --output /workspace/telugu_data"
    echo ""
    
    # Clean up test download
    rm -rf /workspace/telugu_data/test
    
else
    echo ""
    echo "❌ FAILED! Cookies may be invalid or expired"
    echo ""
    echo "Troubleshooting:"
    echo "1. Export fresh cookies from YouTube (make sure you're signed in)"
    echo "2. Watch 1-2 videos before exporting"
    echo "3. Re-upload to /workspace/cookies/youtube_cookies.txt"
    echo "4. Run this test script again"
    echo ""
    exit 1
fi
