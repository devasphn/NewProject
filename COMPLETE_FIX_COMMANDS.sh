#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# COMPLETE FIX FOR YOUTUBE BOT DETECTION - SAFE COMMAND SEQUENCE
# ═══════════════════════════════════════════════════════════════════════
# Run these commands ONE BY ONE in order
# DO NOT run as a script - copy-paste each section manually
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: STOP RUNNING PROCESSES
# ═══════════════════════════════════════════════════════════════════════

echo "Stopping any running downloads..."
pkill -f yt-dlp
pkill -f download_telugu
pkill -f data_collection
sleep 2
echo "✓ All processes stopped"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: CHECK CURRENT DISK USAGE
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "CHECKING YOUR ACTUAL DISK SPACE (not the 207TB host)"
echo "═══════════════════════════════════════════════════════════════"

# Your container disk (400GB)
echo "Container disk (/workspace):"
df -h /workspace | grep -v "^Filesystem"

# Your volume disk (500GB) if mounted
echo ""
echo "Volume disk (if mounted):"
df -h /runpod-volume 2>/dev/null || echo "  (Volume not mounted or different path)"

echo ""
echo "Current data size:"
du -sh /workspace/telugu_data 2>/dev/null || echo "  No telugu_data directory yet"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: SAFE CLEANUP (CHECK FIRST, THEN DELETE)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "CLEANING UP FAILED DOWNLOADS"
echo "═══════════════════════════════════════════════════════════════"

cd /workspace

# Step 3a: Check what exists
echo "Checking what exists..."
ls -lh telugu_data 2>/dev/null || echo "  No telugu_data directory"

# Step 3b: Show directory structure
echo ""
echo "Directory structure:"
find telugu_data -type d 2>/dev/null | head -20 || echo "  Empty or doesn't exist"

# Step 3c: Show total size
echo ""
echo "Total size before cleanup:"
du -sh telugu_data 2>/dev/null || echo "  0 bytes"

# Step 3d: Delete raw videos (SAFE - can re-download)
echo ""
echo "Deleting raw video files..."
rm -rf telugu_data/raw_videos/* 2>/dev/null
echo "✓ Raw videos deleted"

# Step 3e: Delete extracted audio (SAFE - will be re-extracted)
echo "Deleting extracted audio..."
rm -rf telugu_data/raw_audio/* 2>/dev/null
echo "✓ Extracted audio deleted"

# Step 3f: Verify cleanup
echo ""
echo "Size after cleanup:"
du -sh telugu_data 2>/dev/null || echo "  0 bytes (empty)"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: DELETE LOGS
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "DELETING LOG FILES"
echo "═══════════════════════════════════════════════════════════════"

cd /workspace/NewProject

# List logs before deletion
echo "Existing logs:"
ls -lh *.log nohup.out 2>/dev/null || echo "  No logs found"

# Delete logs
rm -f data_collection.log nohup.out collection_log_*.txt 2>/dev/null
rm -f /workspace/logs/*.log 2>/dev/null

echo "✓ Logs deleted"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: CREATE CLEAN DIRECTORY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "CREATING CLEAN DIRECTORY STRUCTURE"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p /workspace/telugu_data/raw_videos
mkdir -p /workspace/telugu_data/raw_audio
mkdir -p /workspace/telugu_data/processed
mkdir -p /workspace/logs
mkdir -p /workspace/cookies

echo "✓ Directories created:"
ls -la /workspace/ | grep -E "telugu_data|logs|cookies"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: VERIFY EVERYTHING IS CLEAN
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "VERIFICATION"
echo "═══════════════════════════════════════════════════════════════"

echo "Disk space available:"
df -h /workspace | grep -v "^Filesystem" | awk '{print "  Used: " $3 " / " $2 " (" $5 " full)"}'

echo ""
echo "Data directories:"
du -sh /workspace/telugu_data/* 2>/dev/null | awk '{print "  " $0}' || echo "  All empty (good!)"

echo ""
echo "✅ CLEANUP COMPLETE!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "NEXT STEPS:"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Export cookies from YouTube browser session"
echo "2. Save as: /workspace/cookies/youtube_cookies.txt"
echo "3. Run test download to verify cookies work"
echo "4. Start full data collection"
echo ""
echo "See: FIX_YOUTUBE_BOT_DETECTION.md for detailed instructions"
echo ""
