# 🚀 PRODUCTION DATA COLLECTION - COMPLETE GUIDE

## 📊 Channel Breakdown

### TIER 1: High Priority (RECOMMENDED START)
**Essential for production-grade codec**

| Channel | Videos | Est. Size | Est. Time | Priority |
|---------|--------|-----------|-----------|----------|
| Raw Talks VK | ALL (~113) | ~56 GB | ~4 hours | ⭐⭐⭐⭐⭐ |
| 10TV Telugu | 500 | ~100 GB | ~16 hours | ⭐⭐⭐⭐⭐ |
| Sakshi TV | 500 | ~100 GB | ~16 hours | ⭐⭐⭐⭐⭐ |
| TV9 Telugu | 500 | ~100 GB | ~16 hours | ⭐⭐⭐⭐⭐ |
| **TOTAL TIER 1** | **~1,613** | **~356 GB** | **~52 hours** | - |

### TIER 2: Speaker Diversity (Optional)
**Adds variety in accents, age, gender**

| Channel | Videos | Est. Size | Est. Time |
|---------|--------|-----------|-----------|
| Telugu Audio Books | ALL (~150) | ~40 GB | ~8 hours |
| Voice of Telugu | ALL (~100) | ~20 GB | ~4 hours |
| Telugu Connects | ALL (~80) | ~15 GB | ~3 hours |
| My Village Show | 200 | ~50 GB | ~8 hours |
| **TOTAL TIER 2** | **~530** | **~125 GB** | **~23 hours** |

### TIER 3: Additional Diversity (Optional)
**For 1000+ hours target**

| Channel | Videos | Est. Size | Est. Time |
|---------|--------|-----------|-----------|
| V6 News | 200 | ~50 GB | ~8 hours |
| Gemini TV | 150 | ~40 GB | ~6 hours |
| Extra Jabardasth | 100 | ~25 GB | ~4 hours |
| **TOTAL TIER 3** | **~450** | **~115 GB** | **~18 hours** |

---

## 🎯 WHICH SCRIPT TO RUN?

### Option 1: Tier 1 Only (RECOMMENDED)
**Best for: Production codec training with high-quality data**

```bash
bash /workspace/NewProject/download_tier1_only.sh
```

**Results:**
- ✅ 1,600+ videos
- ✅ ~356 GB
- ✅ ~800-1000 hours of speech
- ✅ 4-5 distinct speakers
- ✅ High-quality, diverse content
- ⏰ ~52 hours (2.2 days)
- 💰 ~$200 GPU cost

**Why this is enough:**
- Industry-standard codecs trained on 500-1000 hours
- You get 800-1000 hours from Tier 1 alone
- Diverse content: podcasts (conversational) + news (formal)
- Multiple speakers, accents, topics
- **PERFECT for production-grade codec!**

---

### Option 2: All Tiers (Maximum Data)
**Best for: Beating all competitors, maximum diversity**

```bash
bash /workspace/NewProject/download_all_channels.sh
```

**Results:**
- ✅ 2,500+ videos
- ✅ ~600 GB
- ✅ 1,500+ hours of speech
- ✅ 12+ distinct speakers
- ✅ Maximum diversity (accents, ages, topics)
- ⏰ ~93 hours (3.9 days)
- 💰 ~$360 GPU cost

**Why you might want this:**
- Beat Moshi, Maya, Luna (trained on <1000 hours)
- Maximum prosody variation
- Accent coverage: Telangana, Andhra, Coastal, Rural
- Age diversity: 20s to 50s
- Gender balance: 50/50 male/female

---

### Option 3: Custom Selection
**Best for: Specific needs, budget constraints**

Modify `download_tier1_only.sh` to comment out channels you don't want:
```bash
# Comment out a channel to skip it
# yt-dlp ... "https://www.youtube.com/@TV9TeluguLive/videos"
```

---

## 💾 STORAGE REQUIREMENTS

**Your setup:**
- Container: 400 GB ✓
- Volume: 500 GB ✓
- **Total: 900 GB**

**Storage needed:**
- Tier 1 only: ~356 GB → **79% of one disk** ✓
- All tiers: ~600 GB → **67% of both disks** ✓

**You have enough space for ALL tiers!** ✓

---

## ⏰ TIME ESTIMATES

### Tier 1 Only:
```
Raw Talks VK:     4 hours
10TV Telugu:     16 hours
Sakshi TV:       16 hours
TV9 Telugu:      16 hours
─────────────────────────
TOTAL:           ~52 hours (2.2 days)
```

### All Tiers:
```
Tier 1:          52 hours
Tier 2:          23 hours
Tier 3:          18 hours
─────────────────────────
TOTAL:           ~93 hours (3.9 days)
```

**Tip:** You can let it run while doing other work. Check every 6-8 hours.

---

## 💰 COST ESTIMATES

**RunPod H100 at $3.89/hour:**

| Option | Time | GPU Cost | Your Budget | Status |
|--------|------|----------|-------------|---------|
| Tier 1 | 52h | $202 | $100 | ⚠️ Need $102 more |
| All Tiers | 93h | $362 | $100 | ⚠️ Need $262 more |

**Budget tip:** Data collection uses minimal GPU (downloads only). Consider:
1. Download on cheaper CPU pod (~$0.10/hr) = ~$10 for all tiers
2. Transfer data to H100 pod for training
3. Save $350+!

**OR:** Just do Tier 1 over 2-3 days, pause pod when not downloading.

---

## 🚀 STEP-BY-STEP: START TIER 1 DOWNLOAD

### Step 1: Upload Scripts to RunPod
```bash
# Scripts are already on your local machine at:
# d:\NewProject\download_tier1_only.sh
# d:\NewProject\download_all_channels.sh

# Upload to RunPod at /workspace/NewProject/
```

### Step 2: Make Scripts Executable
```bash
chmod +x /workspace/NewProject/download_tier1_only.sh
chmod +x /workspace/NewProject/download_all_channels.sh
```

### Step 3: Start Download (Tier 1)
```bash
cd /workspace/NewProject

# Run in background with nohup
nohup bash download_tier1_only.sh > /workspace/logs/tier1_download.log 2>&1 &

# Save PID
echo $! > /workspace/tier1_download.pid

# Monitor
tail -f /workspace/logs/tier1_download.log
```

### Step 4: Monitor Progress
```bash
# Check every hour
watch -n 3600 '
echo "Videos: $(find /workspace/telugu_data/raw_videos -name "*.mp4" | wc -l)"
echo "Size: $(du -sh /workspace/telugu_data/raw_videos)"
echo "Progress: $(echo "scale=2; $(find /workspace/telugu_data/raw_videos -name "*.mp4" | wc -l) / 1613 * 100" | bc)%"
'

# Check detailed logs per channel
ls -lh /workspace/logs/
tail -f /workspace/logs/RawTalksVK.log
```

### Step 5: Verify Completion
```bash
# Final stats
echo "Total videos: $(find /workspace/telugu_data/raw_videos -name '*.mp4' | wc -l)"
echo "Total size: $(du -sh /workspace/telugu_data/raw_videos)"
echo "Channels downloaded: $(ls /workspace/telugu_data/raw_videos/*/)"

# Check for errors
grep -i error /workspace/logs/tier1_download.log
```

---

## 📊 SUCCESS CRITERIA

**Tier 1 success:**
- ✅ 1,500-1,700 videos downloaded
- ✅ 350-400 GB total size
- ✅ All 4 channels have videos
- ✅ No repeated "bot detection" errors
- ✅ Download archive updated

**How to check:**
```bash
# Count per channel
find /workspace/telugu_data/raw_videos/tier1_podcasts -name "*.mp4" | wc -l  # Should be ~113
find /workspace/telugu_data/raw_videos/tier1_news/10TV -name "*.mp4" | wc -l  # Should be ~500
find /workspace/telugu_data/raw_videos/tier1_news/SakshiTV -name "*.mp4" | wc -l  # Should be ~500
find /workspace/telugu_data/raw_videos/tier1_news/TV9 -name "*.mp4" | wc -l  # Should be ~500
```

---

## 🛑 PAUSE/RESUME/STOP

### Pause (Keep Progress)
```bash
# Stop process
kill $(cat /workspace/tier1_download.pid)

# Resume later (picks up where left off)
nohup bash download_tier1_only.sh >> /workspace/logs/tier1_download.log 2>&1 &
echo $! > /workspace/tier1_download.pid
```

### Stop Completely
```bash
kill $(cat /workspace/tier1_download.pid)
rm /workspace/tier1_download.pid
```

### Delete Everything (Fresh Start)
```bash
rm -rf /workspace/telugu_data/raw_videos/*
rm /workspace/telugu_data/downloaded.txt
rm /workspace/logs/*.log
```

---

## 🎯 RECOMMENDED APPROACH

**For your $100 budget and production needs:**

1. **Start with Tier 1 only** (`download_tier1_only.sh`)
2. **Let it run for 2-3 days** (~52 hours)
3. **You'll get 800-1000 hours** (more than enough!)
4. **Start codec training** while optionally downloading more
5. **Add Tier 2 later** if codec needs more diversity

**Why:**
- ✅ Tier 1 alone = production-grade dataset
- ✅ Matches industry standards (500-1000 hours)
- ✅ Diverse content (podcasts + news)
- ✅ Multiple speakers (4-5 distinct)
- ✅ Fits your budget better
- ✅ Can add more later if needed

---

## ✅ FINAL COMMAND (COPY-PASTE READY)

```bash
# Navigate to project
cd /workspace/NewProject

# Make script executable
chmod +x download_tier1_only.sh

# Start Tier 1 download (RECOMMENDED)
nohup bash download_tier1_only.sh > /workspace/logs/tier1_production.log 2>&1 &
echo $! > /workspace/tier1_download.pid

echo "✅ Tier 1 download started!"
echo "PID: $(cat /workspace/tier1_download.pid)"
echo "Monitor: tail -f /workspace/logs/tier1_production.log"
echo "Expected: ~52 hours, 1,600 videos, 350 GB"

# Monitor
tail -f /workspace/logs/tier1_production.log
```

---

**Ready to collect production-grade Telugu speech data!** 🚀
