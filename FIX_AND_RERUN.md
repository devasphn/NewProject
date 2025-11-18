# CRITICAL FIX - Data Collection Path Issue

## 🐛 The Problem

**Root Cause**: The `data_collection.py` script was **IGNORING** your command line arguments!

Line 223 was:
```python
collector = TeluguDataCollector()  # NO ARGUMENTS!
```

This meant `--data_dir /workspace/telugu_data` was completely ignored, and files were saved to the wrong location.

---

## ✅ What I Fixed

### 1. Added argparse Support
- Now properly handles `--data_dir`, `--config`, `--max_hours`, `--quality`

### 2. Fixed Initialization
```python
# OLD (broken):
collector = TeluguDataCollector()

# NEW (fixed):
collector = TeluguDataCollector(config_path=args.config, output_dir=args.data_dir)
```

### 3. Fixed TV9 Telugu URL
- Changed from `@TV9Telugu` (404 error) to channel ID format

---

## 🔍 Where Are The Old Files?

Check if files were saved to wrong location:

```bash
# Check current directory
ls -la /workspace/NewProject/telugu_data 2>/dev/null

# Or check working directory when script ran
find /workspace -name "*.wav" -type f 2>/dev/null | head -10
```

If files exist in wrong location, delete them:
```bash
rm -rf /workspace/NewProject/telugu_data
```

---

## 🚀 CORRECT WAY TO RUN (Fixed Script)

### Step 1: Pull Latest Fixed Code
```bash
cd /workspace/NewProject
git pull origin main
```

### Step 2: Clean Up
```bash
# Remove any old partial data
rm -rf /workspace/telugu_data
rm -rf /workspace/NewProject/telugu_data

# Create fresh directory
mkdir -p /workspace/telugu_data
```

### Step 3: Run With Fixed Script
```bash
cd /workspace/NewProject

python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100 \
    --quality "high"
```

---

## 📊 What You Should See

### On Start:
```
2025-11-18 XX:XX:XX,XXX - INFO - Starting data collection with:
2025-11-18 XX:XX:XX,XXX - INFO -   Data directory: /workspace/telugu_data
2025-11-18 XX:XX:XX,XXX - INFO -   Config file: data_sources.yaml
2025-11-18 XX:XX:XX,XXX - INFO -   Max hours: 100
2025-11-18 XX:XX:XX,XXX - INFO -   Quality: high
2025-11-18 XX:XX:XX,XXX - INFO - ✓ Node.js detected: v12.22.9
2025-11-18 XX:XX:XX,XXX - INFO - Using device: cuda
2025-11-18 XX:XX:XX,XXX - INFO - Starting priority data collection
```

### During Download:
```
2025-11-18 XX:XX:XX,XXX - INFO - Processing: raw_talks_vk
2025-11-18 XX:XX:XX,XXX - INFO - Downloading from: https://www.youtube.com/@RawTalksWithVK/videos
2025-11-18 XX:XX:XX,XXX - INFO - Running: yt-dlp (with Node.js runtime)...
2025-11-18 XX:XX:XX,XXX - INFO - ✓ Downloaded 10 files for raw_talks_vk (10.76 GB)
```

### After Completion:
```bash
# Verify files exist
ls -lh /workspace/telugu_data/raw/

# Should show:
# drwxr-xr-x 2 root root 4.0K Nov 18 XX:XX raw_talks_vk
# drwxr-xr-x 2 root root 4.0K Nov 18 XX:XX 10TV Telugu
# drwxr-xr-x 2 root root 4.0K Nov 18 XX:XX Sakshi TV
```

---

## ✅ Verification After Download

```bash
# 1. Check total size
du -sh /workspace/telugu_data

# 2. Count files
find /workspace/telugu_data/raw -name "*.wav" | wc -l

# 3. List directories
ls -lh /workspace/telugu_data/raw/

# 4. Quick verification
echo "Total: $(du -sh /workspace/telugu_data | awk '{print $1}') | Files: $(find /workspace/telugu_data/raw -name '*.wav' 2>/dev/null | wc -l)"
```

### Expected Output:
```
Total: 12G | Files: 27
```

---

## 🎯 Success Criteria

After running the fixed script, you should have:

- ✓ `/workspace/telugu_data/raw/raw_talks_vk/` - **10 files, ~11 GB**
- ✓ `/workspace/telugu_data/raw/10TV Telugu/` - **9 files, ~430 MB**
- ✓ `/workspace/telugu_data/raw/Sakshi TV/` - **8 files, ~340 MB**
- ✓ Total: **27 WAV files, ~12 GB**

---

## 🚨 If It Still Fails

### Check Arguments Are Parsed:
The log should show:
```
INFO - Starting data collection with:
INFO -   Data directory: /workspace/telugu_data
```

If this line is missing, the fix didn't apply.

### Re-pull Code:
```bash
cd /workspace/NewProject
git stash  # Save any local changes
git pull origin main
git stash pop  # Restore local changes if needed
```

### Manual Check:
```bash
# Verify argparse is in the file
grep "argparse" /workspace/NewProject/data_collection.py

# Should show:
# import argparse
# parser = argparse.ArgumentParser(...)
```

---

## 📝 Summary of Changes

| File | Change | Reason |
|------|--------|---------|
| `data_collection.py` | Added `import argparse` | Enable command line argument parsing |
| `data_collection.py` | Updated `__main__` section | Parse and use --data_dir argument |
| `data_sources.yaml` | Fixed TV9 Telugu URL | Changed from @handle to channel ID |

---

## 🔄 Complete Commands (Copy-Paste)

```bash
# 1. Go to project
cd /workspace/NewProject

# 2. Pull fixed code
git pull origin main

# 3. Clean up
rm -rf /workspace/telugu_data
mkdir -p /workspace/telugu_data

# 4. Run corrected script
python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100 \
    --quality "high"

# 5. Wait for completion (30-60 minutes)

# 6. Verify
echo "Total: $(du -sh /workspace/telugu_data | awk '{print $1}') | Files: $(find /workspace/telugu_data/raw -name '*.wav' 2>/dev/null | wc -l)"
```

---

**This is the PROPER FIX. Run these commands and the data will download to the correct location!**
