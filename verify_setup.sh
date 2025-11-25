#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# VERIFY SETUP - Check if everything is ready for download
# ═══════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════"
echo "VERIFYING SETUP FOR TIER 1 DOWNLOAD"
echo "═══════════════════════════════════════════════════════════════"
echo ""

READY=true

# Check 1: Node.js
echo "1. Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "   ✅ Node.js installed: $NODE_VERSION"
else
    echo "   ❌ Node.js NOT installed"
    echo "      Fix: curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
    echo "           apt-get install -y nodejs"
    READY=false
fi
echo ""

# Check 2: yt-dlp
echo "2. Checking yt-dlp..."
if command -v yt-dlp &> /dev/null; then
    YTDLP_VERSION=$(yt-dlp --version)
    echo "   ✅ yt-dlp installed: $YTDLP_VERSION"
else
    echo "   ❌ yt-dlp NOT installed"
    echo "      Fix: pip install -U yt-dlp"
    READY=false
fi
echo ""

# Check 3: Cookies file
echo "3. Checking cookies file..."
COOKIES="/workspace/NewProject/cookies/youtube_cookies.txt"
if [ -f "$COOKIES" ]; then
    FILE_SIZE=$(stat -f%z "$COOKIES" 2>/dev/null || stat -c%s "$COOKIES" 2>/dev/null)
    FILE_AGE=$(find "$COOKIES" -mmin +30 2>/dev/null)
    
    if [ "$FILE_SIZE" -lt 500 ]; then
        echo "   ⚠️  Cookies file too small ($FILE_SIZE bytes)"
        echo "      Might be incomplete. Should be 1000-2000 bytes."
        READY=false
    elif [ ! -z "$FILE_AGE" ]; then
        echo "   ⚠️  Cookies file is older than 30 minutes"
        echo "      Recommend exporting fresh cookies"
        echo "      Current age: $(stat -f%Sm -t '%Y-%m-%d %H:%M' "$COOKIES" 2>/dev/null || stat -c%y "$COOKIES" 2>/dev/null | cut -d' ' -f1-2)"
    else
        echo "   ✅ Cookies file found: $FILE_SIZE bytes"
        echo "      Age: $(stat -f%Sm -t '%Y-%m-%d %H:%M' "$COOKIES" 2>/dev/null || stat -c%y "$COOKIES" 2>/dev/null | cut -d' ' -f1-2)"
    fi
else
    echo "   ❌ Cookies file NOT found at $COOKIES"
    echo "      Create directory: mkdir -p /workspace/NewProject/cookies"
    echo "      Upload cookies to this path"
    READY=false
fi
echo ""

# Check 4: Output directory
echo "4. Checking output directory..."
OUTPUT_DIR="/workspace/telugu_data/raw_videos"
if [ -d "$OUTPUT_DIR" ]; then
    EXISTING_FILES=$(find "$OUTPUT_DIR" -name '*.mp4' | wc -l)
    EXISTING_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "   ✅ Output directory exists"
    echo "      Existing videos: $EXISTING_FILES"
    echo "      Current size: $EXISTING_SIZE"
else
    echo "   ⚠️  Output directory doesn't exist (will be created)"
    mkdir -p "$OUTPUT_DIR"
    echo "      Created: $OUTPUT_DIR"
fi
echo ""

# Check 5: Disk space
echo "5. Checking disk space..."
AVAILABLE=$(df -h /workspace | tail -1 | awk '{print $4}')
AVAILABLE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
echo "   Available: $AVAILABLE"
if [ "$AVAILABLE_GB" -lt 400 ]; then
    echo "   ⚠️  Low disk space! Need 400+ GB for Tier 1"
    echo "      Current: ${AVAILABLE_GB}GB"
    READY=false
else
    echo "   ✅ Sufficient disk space"
fi
echo ""

# Check 6: SAFE mode script
echo "6. Checking SAFE mode script..."
SAFE_SCRIPT="/workspace/NewProject/download_tier1_SAFE.sh"
if [ -f "$SAFE_SCRIPT" ]; then
    echo "   ✅ SAFE mode script found"
    if [ -x "$SAFE_SCRIPT" ]; then
        echo "      Executable: Yes"
    else
        echo "      Executable: No (fixing...)"
        chmod +x "$SAFE_SCRIPT"
        echo "      ✅ Made executable"
    fi
else
    echo "   ❌ SAFE mode script NOT found"
    echo "      Run: cd /workspace/NewProject && git pull origin main"
    READY=false
fi
echo ""

# Check 7: Currently running downloads
echo "7. Checking for running downloads..."
if ps aux | grep -q "[y]t-dlp"; then
    echo "   ⚠️  yt-dlp is currently running!"
    echo "      PIDs: $(ps aux | grep "[y]t-dlp" | awk '{print $2}' | tr '\n' ' ')"
    echo "      Consider stopping before starting new download"
else
    echo "   ✅ No active downloads"
fi
echo ""

# Final verdict
echo "═══════════════════════════════════════════════════════════════"
if [ "$READY" = true ]; then
    echo "✅ SYSTEM READY FOR DOWNLOAD!"
    echo ""
    echo "Next steps:"
    echo "  1. Ensure rate limit has expired (wait until 12:30 AM)"
    echo "  2. Export fresh cookies using active session method"
    echo "  3. Run: bash /workspace/NewProject/download_tier1_SAFE.sh"
    echo ""
else
    echo "❌ SYSTEM NOT READY - Fix issues above first"
    echo ""
fi
echo "═══════════════════════════════════════════════════════════════"
