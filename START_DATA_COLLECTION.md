# 🎯 CORRECT Data Collection Commands - Telugu S2S Production

## ❌ ERROR IN YOUR COMMAND

**What you tried:**
```bash
python data_collection.py \
    --config data_sources_PRODUCTION.yaml \
    --output_dir /workspace/telugu_data/raw \    # ❌ Wrong: script uses --data_dir
    --max_videos 500 \                            # ❌ Wrong: script uses --max_hours
    --download_mode full                          # ❌ Wrong: script uses --quality
```

**Error:** `unrecognized arguments: --output_dir --max_videos --download_mode`

---

## ✅ CORRECT COMMANDS (2 Options)

### Option 1: Using `data_collection.py` (Recommended)

**Correct arguments:**
- `--data_dir` (not --output_dir)
- `--config` 
- `--max_hours` (not --max_videos)
- `--quality` (not --download_mode)

**Command for 1000+ hours:**
```bash
nohup python data_collection.py \
    --config data_sources_PRODUCTION.yaml \
    --data_dir /workspace/telugu_data \
    --max_hours 1000 \
    --quality high \
    > data_collection.log 2>&1 &
```

**Monitor progress:**
```bash
# Watch log in real-time
tail -f data_collection.log

# Check how much data collected so far
du -sh /workspace/telugu_data/

# Check specific subdirectories
du -sh /workspace/telugu_data/*
```

---

### Option 2: Using `download_telugu_data_PRODUCTION.py` (Alternative)

**This script is specifically designed for production data collection.**

**Command:**
```bash
nohup python download_telugu_data_PRODUCTION.py \
    --config data_sources_PRODUCTION.yaml \
    --output /workspace/telugu_data \
    > data_collection.log 2>&1 &
```

**For Tier 1 channels only (highest quality):**
```bash
nohup python download_telugu_data_PRODUCTION.py \
    --config data_sources_PRODUCTION.yaml \
    --output /workspace/telugu_data \
    --tier1-only \
    > data_collection.log 2>&1 &
```

---

## 🎯 RECOMMENDED: Best Command for Production

**For natural, human-like speech with proper prosody:**

```bash
# Use data_collection.py with high quality and 1000 hours target
nohup python data_collection.py \
    --config data_sources_PRODUCTION.yaml \
    --data_dir /workspace/telugu_data \
    --max_hours 1000 \
    --quality high \
    > /workspace/logs/data_collection_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the process ID
echo $! > /workspace/data_collection.pid
```

**Why this is best:**
- ✅ `--quality high` ensures best audio fidelity (crucial for natural prosody)
- ✅ `--max_hours 1000` targets 1000+ hours (enough for production-grade codec)
- ✅ `data_sources_PRODUCTION.yaml` uses curated, high-quality channels
- ✅ Saves PID for easy process management

---

## 📊 Monitoring & Management

### Check Progress
```bash
# Real-time log viewing
tail -f /workspace/logs/data_collection_*.log

# Total data collected
du -sh /workspace/telugu_data/

# Breakdown by subdirectory
du -h --max-depth=1 /workspace/telugu_data/ | sort -rh

# Number of audio files
find /workspace/telugu_data/ -name "*.wav" | wc -l

# Total audio duration (approximate)
find /workspace/telugu_data/ -name "*.wav" -exec soxi -D {} \; 2>/dev/null | awk '{s+=$1} END {print s/3600 " hours"}'
```

### Stop Collection (if needed)
```bash
# Get process ID
cat /workspace/data_collection.pid

# Stop gracefully
kill $(cat /workspace/data_collection.pid)

# Force stop if needed
kill -9 $(cat /workspace/data_collection.pid)
```

### Resume Collection
```bash
# Collection will automatically resume where it left off
nohup python data_collection.py \
    --config data_sources_PRODUCTION.yaml \
    --data_dir /workspace/telugu_data \
    --max_hours 1000 \
    --quality high \
    >> /workspace/logs/data_collection_resumed.log 2>&1 &
```

---

## 🎤 Ensuring Natural, Human-Like Speech

### Quality Settings Explained

**`--quality high`** downloads:
- ✅ Best available audio codec (opus/aac at 128-192kbps)
- ✅ 48kHz sample rate (downsampled to 16kHz in processing)
- ✅ Preserves natural prosody, intonation, rhythm
- ✅ Minimal compression artifacts
- ✅ Better for emotional expression

**vs `--quality medium/low`:**
- ❌ Lower bitrate (64-96kbps)
- ❌ More compression artifacts
- ❌ Loss of subtle prosodic features
- ❌ Unnatural-sounding after codec training

### Data Source Quality (data_sources_PRODUCTION.yaml)

**Ensure your YAML includes:**

1. **Native Telugu speakers** (not dubbing)
2. **Conversational content** (not scripted/robotic)
3. **Emotional variety** (happy, sad, excited, calm)
4. **Multiple speakers** (male/female, young/mature)
5. **Clear audio** (no background music/noise)
6. **Natural pacing** (not too fast/slow)

**Good sources for natural speech:**
- Telugu vlogs (daily life conversations)
- Interviews (spontaneous speech)
- Podcasts (natural dialogue)
- News anchors (clear, expressive speech)
- Storytelling channels (emotional range)

**Avoid:**
- Dubbed content (unnatural prosody)
- Heavily edited audio (removes natural pauses)
- Music with vocals (interferes with speech)
- Low-quality recordings (degrades codec)
- Scripted/read speech (sounds robotic)

---

## 📈 Expected Timeline & Storage

### For 1000 Hours of High-Quality Data

**Download Time:**
```
Speed: ~5-10 hours per 100 hours (depends on network)
Total: ~50-100 hours (2-4 days continuous)
```

**Storage:**
```
Raw videos:           ~300-400 GB (if kept)
Extracted audio:      ~150-200 GB (16kHz WAV)
Processed segments:   ~150-200 GB (after VAD)
Total peak usage:     ~600-800 GB
```

**With 400GB container + 500GB volume:**
```
Container: 400 GB
├── Raw videos: Delete after extraction → saves ~350 GB
├── Extracted audio: ~180 GB (keep)
├── Processing temp: ~50 GB
└── Free: ~170 GB

Volume: 500 GB
├── Backups: ~100 GB
├── Models (future): ~50 GB
└── Free: ~350 GB

✅ You have ENOUGH space with smart management
```

---

## 🔧 Optimization Tips

### Disk Space Management During Collection

**Delete raw videos after extraction:**
```bash
# Add this to a cron job or run periodically
find /workspace/telugu_data/raw_videos -name "*.mp4" -mtime +1 -delete
# Deletes videos older than 1 day (after audio extracted)
```

**Or configure extraction on-the-fly:**
```python
# Both scripts extract audio immediately
# Raw videos can be deleted right after extraction
```

**Monitor disk space:**
```bash
# Create a monitoring script
cat > /workspace/check_space.sh << 'EOF'
#!/bin/bash
while true; do
    USAGE=$(df -h /workspace | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $USAGE -gt 85 ]; then
        echo "⚠️  WARNING: Disk at ${USAGE}%"
        # Auto-cleanup: delete old raw videos
        find /workspace/telugu_data/raw_videos -name "*.mp4" -mtime +1 -delete
    fi
    sleep 300  # Check every 5 minutes
done
EOF
chmod +x /workspace/check_space.sh

# Run in background
nohup /workspace/check_space.sh > /workspace/logs/space_monitor.log 2>&1 &
```

---

## ✅ Pre-Flight Checklist

Before starting collection:

- ✅ 400GB container + 500GB volume (you have this!)
- ✅ ffmpeg installed: `ffmpeg -version`
- ✅ yt-dlp installed: `yt-dlp --version`
- ✅ Node.js installed: `node --version`
- ✅ data_sources_PRODUCTION.yaml exists
- ✅ /workspace/telugu_data/ directory created
- ✅ /workspace/logs/ directory created
- ✅ Enough budget ($100 recharged - good!)

**Check everything:**
```bash
# Verify installations
echo "Checking dependencies..."
ffmpeg -version | head -n1
yt-dlp --version
node --version
python -c "import yaml; print('yaml: ✓')"

# Verify config file
ls -lh data_sources_PRODUCTION.yaml

# Create directories
mkdir -p /workspace/telugu_data /workspace/logs

# Check disk space
df -h /workspace
```

---

## 🚀 START DATA COLLECTION NOW

**Final command (copy-paste ready):**

```bash
# Navigate to project
cd /workspace/NewProject

# Start collection with logging
nohup python data_collection.py \
    --config data_sources_PRODUCTION.yaml \
    --data_dir /workspace/telugu_data \
    --max_hours 1000 \
    --quality high \
    > /workspace/logs/data_collection_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > /workspace/data_collection.pid
echo "Data collection started! PID: $(cat /workspace/data_collection.pid)"

# Watch progress
tail -f /workspace/logs/data_collection_*.log
```

**Alternative (if you prefer the PRODUCTION script):**

```bash
cd /workspace/NewProject

nohup python download_telugu_data_PRODUCTION.py \
    --config data_sources_PRODUCTION.yaml \
    --output /workspace/telugu_data \
    > /workspace/logs/data_collection_production_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > /workspace/data_collection.pid
tail -f /workspace/logs/data_collection_production_*.log
```

---

## 📊 Success Metrics

**You'll know collection is successful when:**

- ✅ Log shows downloads progressing
- ✅ `/workspace/telugu_data/raw/` growing in size
- ✅ WAV files appearing in subdirectories
- ✅ No repeated errors in log
- ✅ Disk space stable (not filling too fast)

**Target:**
- 📈 800-1200 hours of audio (buffer for filtering)
- 🎵 High quality (minimal artifacts)
- 🗣️ Natural, conversational speech
- 👥 Multiple speakers (diversity)
- 🎭 Emotional variety (prosody)

---

## 🎯 After Collection: Next Steps

**When collection completes (2-4 days):**

1. **Validate data quality:**
```bash
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/telugu_data/processed
```

2. **Start codec training:**
```bash
python train_codec_dac.py \
    --data_dir /workspace/telugu_data/processed/train \
    --batch_size 16 \
    --epochs 100 \
    --use_wandb
```

3. **Expected results:**
   - SNR: >25 dB (natural quality)
   - Prosody: Preserved (human-like intonation)
   - Latency: <10ms (real-time capable)

---

## 💡 Pro Tips

1. **Don't wait for 100% completion** - Start training when you have 200-300 hours
2. **Quality > Quantity** - 500 hours of high-quality > 1000 hours of low-quality
3. **Monitor diversity** - Ensure variety in speakers, topics, emotions
4. **Save checkpoints** - Backup data to HuggingFace every 100 hours
5. **Test early** - Train a small model on first 50 hours to verify quality

---

**Ready to collect natural, human-like Telugu speech!** 🚀

**Budget status:** $100 recharged ✓ (enough for 25-50 hours on H100)  
**Storage:** 400GB container + 500GB volume ✓ (sufficient)  
**Target:** 1000+ hours, production-grade, natural prosody ✓

**START COLLECTION NOW!** ⚡
