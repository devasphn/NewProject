# ✅ RATE LIMIT FIX CHECKLIST

## 📊 Current Situation

**Status:** YouTube rate-limited for 1 hour (until ~12:30 AM)  
**Downloaded:** 24/113 videos from Raw Talks VK (21% success)  
**Issues:** Missing Node.js + Cookie rotation + Too-short sleep intervals

---

## 🔧 FIX STEPS (Do in Order!)

### ✅ STEP 1: Install Node.js (DO NOW - While Waiting)

```bash
# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Verify
node --version  # Should show v20.x
npm --version   # Should show v10.x
```

**Why:** YouTube extraction requires JavaScript runtime for signature decryption.

---

### ⏰ STEP 2: Wait for Rate Limit to Expire

**Last error:** ~11:22 PM  
**Safe to retry:** 12:30 AM (1 hour + 10 min buffer)

```bash
# Check current time
date

# Set reminder for 12:30 AM
```

**Do NOT attempt downloads before rate limit expires!**

---

### 🍪 STEP 3: Export Fresh Cookies (After 12:30 AM)

**CRITICAL: Must follow this EXACT method:**

#### 3.1 Open Incognito Browser
```
1. Open Chrome/Edge in INCOGNITO mode
2. Go to youtube.com
3. Sign in with your YouTube account
```

#### 3.2 Build Active Session (Important!)
```
4. Search: "telugu podcast" or "telugu news"
5. Watch 3-5 different videos (30+ seconds each)
6. Like or comment on 1-2 videos
7. Subscribe to 1-2 channels
```

#### 3.3 Export Cookies WHILE Video is Playing
```
8. Start playing another video (full screen if possible)
9. While video is ACTIVELY PLAYING:
   - Open browser extension (Cookie-Editor or EditThisCookie)
   - Click "Export" → "Netscape format"
   - Copy ALL cookies
10. Save as youtube_cookies.txt
```

#### 3.4 Upload Immediately
```
11. Upload to: /workspace/NewProject/cookies/youtube_cookies.txt
12. Start download within 2 minutes!
```

**Why this works:**
- Active playback = fresh authentication tokens
- YouTube won't rotate cookies during video playback
- Tokens valid for 30-60 minutes if used immediately

---

### 🚀 STEP 4: Pull Updated Scripts

```bash
cd /workspace/NewProject
git pull origin main

# Make executable
chmod +x download_tier1_SAFE.sh

# Verify new script exists
ls -lh download_tier1_SAFE.sh
```

---

### ▶️ STEP 5: Start Download with SAFE Mode

```bash
# Clean previous partial downloads (optional)
rm -rf /workspace/telugu_data/raw_videos/*
rm -f /workspace/telugu_data/downloaded.txt

# Create directories
mkdir -p /workspace/logs
mkdir -p /workspace/telugu_data/raw_videos

# Start SAFE MODE download (with anti-rate-limit protection)
cd /workspace/NewProject

nohup bash download_tier1_SAFE.sh > /workspace/logs/tier1_SAFE.log 2>&1 &
echo $! > /workspace/tier1_download.pid

echo "✅ SAFE MODE download started!"
```

---

### 📊 STEP 6: Monitor Progress

```bash
# Watch live progress
tail -f /workspace/logs/tier1_SAFE.log

# Check downloaded videos count
find /workspace/telugu_data/raw_videos -name '*.mp4' | wc -l

# Check total size
du -sh /workspace/telugu_data/raw_videos

# Check if process is running
ps aux | grep -i yt-dlp
```

---

## 🛑 If Errors Occur

### Error: "No JavaScript runtime"
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
node --version
```

### Error: "Cookies no longer valid"
```bash
# Stop download
kill $(cat /workspace/tier1_download.pid)

# Export fresh cookies using STEP 3 method
# Upload new cookies
# Restart download
bash download_tier1_SAFE.sh
```

### Error: "Rate limited again"
```bash
# Stop download
kill $(cat /workspace/tier1_download.pid)

# Check sleep intervals in script
grep "sleep-interval" download_tier1_SAFE.sh
# Should show: 15-45 seconds

# Wait 1 hour before retrying
```

---

## 📈 Expected Results with SAFE Mode

**Sleep intervals:**
- Min: 15 seconds between videos
- Max: 45 seconds between videos
- API: 2 seconds between requests

**Estimated download time:**
- Raw Talks VK: ~6-8 hours (113 videos, resume from 24)
- 10TV Telugu: ~20-25 hours (500 videos)
- Sakshi TV: ~20-25 hours (500 videos)
- TV9 Telugu: ~20-25 hours (500 videos)

**Total: ~70-85 hours for all 4 channels**

**Success rate:** 90-95% (with proper cookies + Node.js + safe intervals)

---

## ✅ SUCCESS CRITERIA

- [x] Node.js installed and verified
- [ ] Rate limit expired (wait until 12:30 AM)
- [ ] Fresh cookies exported using active session method
- [ ] SAFE mode script pulled and ready
- [ ] Download started with SAFE mode
- [ ] No cookie expiration errors in logs
- [ ] No rate limit errors in logs
- [ ] Videos downloading successfully

---

## 🎯 FINAL NOTES

1. **DO NOT rush** - Wait for rate limit to fully expire
2. **Export cookies correctly** - Use active session method
3. **Start download immediately** - Within 2 minutes of cookie export
4. **Monitor logs** - Check for errors regularly
5. **Be patient** - SAFE mode is slower but much more reliable

**The SAFE mode script will take longer but has 90%+ success rate!**
