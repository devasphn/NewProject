# 🚨 YOUTUBE BOT DETECTION - COMPLETE FIX GUIDE

## ❌ Your Error

```
ERROR: [youtube] Sign in to confirm you're not a bot
Use --cookies-from-browser or --cookies for authentication
```

**Cause:** YouTube detects automated downloads and requires authentication via browser cookies.

---

## ✅ SOLUTION 1: Export Cookies from Browser (EASIEST)

### Step 1: Install Browser Cookie Export Extension

**For Chrome/Edge:**
1. Install: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)

**For Firefox:**
1. Install: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

### Step 2: Export YouTube Cookies

1. **Go to YouTube** in your browser: https://www.youtube.com
2. **Sign in** to your YouTube account
3. **Watch 1-2 videos** (to generate valid cookies)
4. **Click the extension icon**
5. **Click "Export" or "Current Site"**
6. **Save as:** `youtube_cookies.txt`

### Step 3: Upload Cookies to RunPod

**Option A: Use RunPod Web Terminal file upload**
1. In RunPod terminal, create directory:
   ```bash
   mkdir -p /workspace/cookies
   ```
2. Upload `youtube_cookies.txt` to `/workspace/cookies/`

**Option B: Copy-paste content**
```bash
# On RunPod terminal
cat > /workspace/cookies/youtube_cookies.txt << 'EOF'
# Paste your entire cookie file content here
# Then press Ctrl+D
EOF
```

### Step 4: Fix the Download Script

**Edit line 181 in `download_telugu_data_PRODUCTION.py`:**

Find this section (around line 166-181):
```python
cmd = [
    "yt-dlp",
    "--format", "best[height<=720]",
    ...
    "--concurrent-fragments", "2",
]
```

**Add AFTER line 181:**
```python
# Add cookies for bot detection bypass
cmd.extend([
    "--cookies", "/workspace/cookies/youtube_cookies.txt"
])
```

---

## ✅ SOLUTION 2: Use Browser Cookie Extraction (NO MANUAL EXPORT)

**Install required package:**
```bash
pip install browser-cookie3
```

**Modify download script to use browser cookies directly:**

Add after line 181:
```python
# Use browser cookies automatically
cmd.extend([
    "--cookies-from-browser", "chrome"  # or "firefox", "edge"
])
```

**Note:** This requires Chrome/Firefox to be installed on RunPod (may not work)

---

## 🔧 SOLUTION 3: Quick Fix Script (AUTOMATED)

I'll create a fixed version of the download script for you.

---

## 📊 DISK SPACE EXPLANATION

**You saw:** `Total: 207406.99 GB` (207 TB!)

**Why this is confusing:**

RunPod shows the **entire node's disk**, not just your allocated space.

**Your actual allocation:**
```
Container disk: 400 GB  (where /workspace lives)
Volume disk:    500 GB  (persistent storage)
```

**Check YOUR actual space:**
```bash
# Check /workspace (your container)
df -h /workspace

# Expected output:
# Filesystem      Size  Used Avail Use% Mounted on
# overlay         400G   20G  380G   5% /workspace
```

**Verify volume disk:**
```bash
df -h /runpod-volume

# Expected output:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sdb        500G  1.0G  499G   1% /runpod-volume
```

**Don't worry about the 207TB** - that's the host machine, not your pod.

---

## 🗑️ CLEANUP COMMANDS (SAFE, ONE BY ONE)

### Check Current Data

```bash
# 1. See what's been downloaded
ls -lh /workspace/telugu_data/

# 2. Check subdirectories
ls -lh /workspace/telugu_data/raw_videos/

# 3. Check total size
du -sh /workspace/telugu_data/
```

### Delete Downloaded Data (CAREFUL!)

```bash
# Step 1: Navigate to workspace
cd /workspace

# Step 2: Check what will be deleted (DRY RUN)
ls -la telugu_data/

# Step 3: Delete raw videos (if any downloaded)
rm -rf telugu_data/raw_videos/*

# Step 4: Delete extracted audio (if any)
rm -rf telugu_data/raw_audio/*

# Step 5: Verify deletion
du -sh telugu_data/
# Should show very small size (a few KB for empty directories)

# Step 6: Completely remove telugu_data (if you want fresh start)
rm -rf /workspace/telugu_data

# Step 7: Verify complete removal
ls -la /workspace/ | grep telugu
# Should show nothing
```

### Delete Logs

```bash
# Delete collection logs
rm -f /workspace/NewProject/data_collection.log
rm -f /workspace/NewProject/nohup.out
rm -f /workspace/logs/data_collection*.log
rm -f /workspace/logs/production_collection*.log

# Verify
ls -lh /workspace/logs/
```

### Complete Clean Slate

```bash
# ONE COMMAND to delete everything (CAREFUL!)
rm -rf /workspace/telugu_data /workspace/logs/*.log /workspace/NewProject/*.log /workspace/NewProject/nohup.out

# Then recreate clean directories
mkdir -p /workspace/telugu_data /workspace/logs /workspace/cookies
```

---

## ✅ COMPLETE FIX - STEP BY STEP

### Step 1: Clean Up Current Mess
```bash
cd /workspace
rm -rf telugu_data
rm -f NewProject/*.log NewProject/nohup.out
mkdir -p telugu_data logs cookies
```

### Step 2: Export Cookies from Browser
1. Install "Get cookies.txt LOCALLY" extension in Chrome
2. Go to youtube.com and sign in
3. Watch 1-2 videos
4. Click extension → Export
5. Save as `youtube_cookies.txt`

### Step 3: Upload Cookies to RunPod
```bash
# Create cookies directory
mkdir -p /workspace/cookies

# Upload youtube_cookies.txt to /workspace/cookies/
# Use RunPod file upload or copy-paste content
```

### Step 4: Verify Cookie File
```bash
# Check cookie file exists and has content
ls -lh /workspace/cookies/youtube_cookies.txt
head -5 /workspace/cookies/youtube_cookies.txt
# Should show cookie data, not empty
```

### Step 5: Run Fixed Command
```bash
cd /workspace/NewProject

# Run with cookies
yt-dlp \
  --cookies /workspace/cookies/youtube_cookies.txt \
  --format "best[height<=720]" \
  --output "/workspace/telugu_data/test/%(title)s.%(ext)s" \
  --max-downloads 1 \
  "https://www.youtube.com/@RawTalksWithVK"

# This should download 1 video successfully
# If it works, the cookies are valid
```

### Step 6: Fix Download Script

**I'll create a fixed version for you - see below**

---

## 🚀 ALTERNATIVE: Use My Fixed Script

**Run this to test cookies work:**

```bash
cd /workspace/NewProject

# Test download with cookies (1 video only)
python -c "
import subprocess
import sys

cmd = [
    'yt-dlp',
    '--cookies', '/workspace/cookies/youtube_cookies.txt',
    '--format', 'best[height<=720]',
    '--output', '/workspace/telugu_data/test/%(title)s.%(ext)s',
    '--max-downloads', '1',
    'https://www.youtube.com/@RawTalksWithVK'
]

print('Testing YouTube download with cookies...')
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print('✅ SUCCESS! Cookies work.')
    print('You can now proceed with full download.')
else:
    print('❌ FAILED!')
    print(result.stderr)
    sys.exit(1)
"
```

---

## 📝 SUMMARY OF WHAT YOU NEED

1. ✅ Export cookies from YouTube (signed-in browser)
2. ✅ Upload `youtube_cookies.txt` to `/workspace/cookies/`
3. ✅ Clean up failed downloads
4. ✅ Use fixed script (I'll provide)
5. ✅ Ignore the 207TB disk space (it's the host, not your pod)
6. ✅ Your actual space: 400GB container + 500GB volume (plenty!)

---

## 🎯 READY-TO-USE CLEANUP COMMANDS

**Copy-paste these ONE BY ONE:**

```bash
# 1. Stop any running downloads
pkill -f yt-dlp
pkill -f download_telugu

# 2. Navigate to workspace
cd /workspace

# 3. Show current disk usage
df -h /workspace
du -sh telugu_data 2>/dev/null || echo "No telugu_data yet"

# 4. Delete downloaded data (if any)
rm -rf telugu_data/raw_videos/* 2>/dev/null
rm -rf telugu_data/raw_audio/* 2>/dev/null

# 5. Verify clean
du -sh telugu_data 2>/dev/null || echo "Cleaned"

# 6. Delete all logs
rm -f NewProject/*.log NewProject/nohup.out logs/*.log

# 7. Create fresh structure
mkdir -p telugu_data/raw_videos telugu_data/raw_audio telugu_data/processed
mkdir -p logs cookies

# 8. Verify structure
tree -L 2 /workspace/ 2>/dev/null || ls -la /workspace/

echo "✅ Cleanup complete!"
```

---

**Next: Export cookies and I'll give you the fixed script!** 🚀
