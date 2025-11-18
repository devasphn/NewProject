# Data Collection Improvements - With Node.js

## ✅ Changes Made to data_collection.py

### 1. Node.js Verification
- Added `_verify_nodejs()` method that runs on initialization
- Checks Node.js version and logs it
- Provides installation instructions if not found

### 2. Optimized Download Settings
**Old (Android client workaround):**
```python
--extractor-args youtube:player_client=android
```

**New (Full Node.js capabilities):**
```python
-f bestaudio  # Best audio quality
--audio-quality 0  # Highest quality
--progress  # Show download progress
--ignore-errors  # Continue on individual video errors
```

### 3. Better Filtering
```python
--match-filter "duration < 7200 & duration > 60"
```
- Only downloads videos between 1 minute and 2 hours
- Skips shorts and very long streams

### 4. Enhanced Logging
- Shows download size in GB
- Better error messages with stderr/stdout
- Timeout handling (10 minutes per channel)

### 5. URL Handling
- Automatically appends `/videos` to channel URLs
- Example: `@RawTalksWithVK` → `@RawTalksWithVK/videos`

## 🎯 Expected Behavior

### On Start:
```
2025-11-18 12:00:00,000 - INFO - ✓ Node.js detected: v20.11.0
2025-11-18 12:00:00,001 - INFO - Using device: cuda
2025-11-18 12:00:00,002 - INFO - Starting priority data collection
```

### During Download:
```
2025-11-18 12:00:05,000 - INFO - Processing: raw_talks_vk
2025-11-18 12:00:05,001 - INFO - Downloading from: https://www.youtube.com/@RawTalksWithVK/videos
2025-11-18 12:00:05,002 - INFO - Running: yt-dlp (with Node.js runtime)...
2025-11-18 12:00:05,003 - INFO - Target: https://www.youtube.com/@RawTalksWithVK/videos
```

### On Success:
```
2025-11-18 12:05:30,000 - INFO - ✓ Downloaded 10 files for raw_talks_vk (2.34 GB)
```

### On Failure:
```
2025-11-18 12:05:30,000 - WARNING - No files downloaded for channel_name
2025-11-18 12:05:30,001 - WARNING - stderr: [error details]
```

## 📁 Expected Directory Structure

```
/workspace/telugu_data/
├── raw/
│   ├── raw_talks_vk/
│   │   ├── EP - 110 MAD FUNNNN FT SUMA AKKAAA.wav
│   │   ├── Ep - 109 1st TIME ON RAW TALKS.wav
│   │   └── ... (8 more files)
│   ├── 10TV_Telugu/
│   │   ├── 10TV Telugu News LIVE.wav
│   │   └── ... (9 more files)
│   ├── Sakshi_TV/
│   ├── TV9_Telugu/
│   └── NTV_Telugu/
├── processed/
├── segments/
└── metadata/
```

## 🔧 How to Use

### 1. Verify Node.js
```bash
node --version  # Should show v20.x.x
```

### 2. Run Data Collection
```bash
cd /workspace/NewProject
python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100 \
    --quality "high"
```

### 3. Monitor Progress
```bash
# Watch directory size
watch -n 60 "du -sh /workspace/telugu_data && ls -lh /workspace/telugu_data/raw"

# Check logs
tail -f data_collection.log
```

## ⏱️ Performance Expectations

| Channel | Videos | Avg Duration | Expected Size | Time |
|---------|--------|--------------|---------------|------|
| Raw Talks VK | 10 | 90 min | 2-3 GB | 10-15 min |
| 10TV Telugu | 10 | 30 min | 1-2 GB | 8-12 min |
| Sakshi TV | 10 | 30 min | 1-2 GB | 8-12 min |
| TV9 Telugu | 10 | 30 min | 1-2 GB | 8-12 min |
| NTV Telugu | 10 | 30 min | 1-2 GB | 8-12 min |
| **Total** | **50** | - | **8-12 GB** | **45-60 min** |

## 🐛 Troubleshooting

### If Node.js warning appears:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
node --version
```

### If downloads fail:
1. Check internet connection
2. Verify YouTube URLs are accessible
3. Try manual download:
```bash
yt-dlp -f bestaudio -x --audio-format wav \
    --max-downloads 1 \
    "https://www.youtube.com/@RawTalksWithVK/videos"
```

### If timeout occurs:
- Normal for channels with many long videos
- Script continues to next channel automatically

## 📊 Next Steps After Collection

Once data collection completes (8-12 GB collected):
1. Check `data_collection.log` for summary
2. Verify files: `ls -lh /workspace/telugu_data/raw/*/`
3. Proceed to **Step 4: Prepare Speaker Data** in DEPLOYMENT_MANUAL_V2.md

---

**Status**: Optimized for Node.js runtime with best audio quality and robust error handling!
