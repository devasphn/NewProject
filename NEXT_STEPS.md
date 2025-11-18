# Next Steps - Data Collection

## ✅ What You've Done
1. ✓ Installed all Python packages
2. ✓ Fixed YAML syntax error
3. ✓ Updated YouTube channel URLs
4. ✓ Installed Node.js in RunPod terminal
5. ✓ Fixed data_collection.py to use Android client

## 🚀 What To Do Now

### Step 1: Verify Node.js Installation
```bash
node --version  # Should show v20.x.x
npm --version   # Should show 10.x.x
```

### Step 2: Pull Latest Code
```bash
cd /workspace/NewProject
git pull origin main
```

### Step 3: Start Data Collection
```bash
# Kill any old processes
pkill -f data_collection.py

# Clear old data
rm -rf /workspace/telugu_data/*

# Start fresh in screen
screen -S data_collection

# Inside screen:
python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100 \
    --quality "high"

# Detach: Ctrl+A, then D
```

### Step 4: Monitor Progress (New Terminal)
```bash
# Watch data directory grow
watch -n 60 "du -sh /workspace/telugu_data && ls -lh /workspace/telugu_data/raw"

# Expected:
# - raw_talks_vk/ folder with .wav files
# - 10TV_Telugu/ folder with .wav files
# - Sakshi_TV/ folder with .wav files
# - Size growing to several GB
```

### Step 5: Check Logs
```bash
# Reattach to screen
screen -r data_collection

# You should see:
# - "Downloading from: https://www.youtube.com/@RawTalksWithVK/videos"
# - "Running: yt-dlp with Android client (no JS needed)..."
# - "✓ Downloaded X files for raw_talks_vk"
# - Progress for each channel

# Detach: Ctrl+A, then D
```

## 🔍 Troubleshooting

### If downloads still fail:
```bash
# Test yt-dlp manually
yt-dlp --extractor-args "youtube:player_client=android" \
    --max-downloads 1 \
    -x --audio-format wav \
    "https://www.youtube.com/@RawTalksWithVK/videos"
```

### If "No files downloaded":
- Check internet connection
- Verify YouTube URLs are accessible
- Check /workspace/telugu_data/raw/ for subdirectories

## ⏱️ Expected Timeline
- **10-15 minutes**: First channel (Raw Talks) downloads 10 videos
- **30-60 minutes**: All 5 channels complete (50 videos total)
- **Final size**: 5-10 GB of WAV files

## 📊 After Data Collection Completes
Move to **DEPLOYMENT_MANUAL_V2.md** → **Step 4: Prepare Speaker Data**

---

**Current Status**: Ready to start data collection with Node.js installed and code fixed!
