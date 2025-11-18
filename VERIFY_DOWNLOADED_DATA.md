# Complete Data Verification Guide

Run these commands in your **RunPod terminal** to verify the downloaded data.

---

## Step 1: Check Total Size

```bash
du -sh /workspace/telugu_data
```

**Expected Output:**
```
12G     /workspace/telugu_data
```
(Approximately 11-12 GB including metadata)

---

## Step 2: Check Each Directory Size

```bash
du -sh /workspace/telugu_data/raw/*
```

**Expected Output:**
```
11G     /workspace/telugu_data/raw/10TV Telugu
430M    /workspace/telugu_data/raw/raw_talks_vk
340M    /workspace/telugu_data/raw/Sakshi TV
```

---

## Step 3: Count Total Files

```bash
find /workspace/telugu_data/raw -name "*.wav" | wc -l
```

**Expected Output:**
```
27
```
(Total number of .wav files downloaded)

---

## Step 4: List Files in Each Directory

### Raw Talks VK:
```bash
ls -lh /workspace/telugu_data/raw/raw_talks_vk/ | head -15
```

**Expected Output:**
```
total 11G
-rw-r--r-- 1 root root 1.2G Nov 18 12:11 EP - 110 MAD FUNNNN FT SUMA AKKAAA.wav
-rw-r--r-- 1 root root 1.1G Nov 18 12:11 Ep - 109 1st TIME ON RAW TALKS.wav
-rw-r--r-- 1 root root 1.0G Nov 18 12:11 EP - 108 FIRST TIME ON RAW TALKS.wav
... (10 files total)
```

### 10TV Telugu:
```bash
ls -lh /workspace/telugu_data/raw/10TV\ Telugu/ | head -15
```

**Expected Output:**
```
total 430M
-rw-r--r-- 1 root root 45M Nov 18 12:12 10TV Telugu News LIVE.wav
-rw-r--r-- 1 root root 48M Nov 18 12:12 Breaking News Live.wav
... (9 files total)
```

### Sakshi TV:
```bash
ls -lh /workspace/telugu_data/raw/Sakshi\ TV/ | head -15
```

**Expected Output:**
```
total 340M
-rw-r--r-- 1 root root 42M Nov 18 12:13 Sakshi TV Live News.wav
-rw-r--r-- 1 root root 38M Nov 18 12:13 Prime Time News.wav
... (8 files total)
```

---

## Step 5: Detailed File Count Per Directory

```bash
echo "=== File Count Per Directory ==="
echo "Raw Talks VK:"
ls -1 /workspace/telugu_data/raw/raw_talks_vk/*.wav 2>/dev/null | wc -l
echo ""
echo "10TV Telugu:"
ls -1 /workspace/telugu_data/raw/10TV\ Telugu/*.wav 2>/dev/null | wc -l
echo ""
echo "Sakshi TV:"
ls -1 /workspace/telugu_data/raw/Sakshi\ TV/*.wav 2>/dev/null | wc -l
```

**Expected Output:**
```
=== File Count Per Directory ===
Raw Talks VK:
10

10TV Telugu:
9

Sakshi TV:
8
```

---

## Step 6: Verify File Integrity (Check if files are valid WAV)

```bash
file /workspace/telugu_data/raw/raw_talks_vk/*.wav | head -3
```

**Expected Output:**
```
/workspace/telugu_data/raw/raw_talks_vk/EP - 110 MAD FUNNNN FT SUMA AKKAAA.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, stereo 44100 Hz
/workspace/telugu_data/raw/raw_talks_vk/Ep - 109 1st TIME ON RAW TALKS.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, stereo 44100 Hz
...
```

All files should show "WAVE audio" - this confirms they are valid audio files.

---

## Step 7: Check Audio Duration (Sample)

```bash
# Install ffprobe if not available
apt-get update && apt-get install -y ffmpeg

# Check duration of first file
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \
    /workspace/telugu_data/raw/raw_talks_vk/*.wav | head -1
```

**Expected Output:**
```
5400.123456
```
(Duration in seconds - Raw Talks videos are typically 60-120 minutes = 3600-7200 seconds)

---

## Step 8: Complete Verification Script

Run this all-in-one verification:

```bash
#!/bin/bash
echo "=================================="
echo "Telugu Data Collection Verification"
echo "=================================="
echo ""

echo "1. Total Size:"
du -sh /workspace/telugu_data
echo ""

echo "2. Directory Sizes:"
du -sh /workspace/telugu_data/raw/*
echo ""

echo "3. Total WAV Files:"
find /workspace/telugu_data/raw -name "*.wav" | wc -l
echo ""

echo "4. Files Per Directory:"
echo "   Raw Talks VK: $(ls -1 /workspace/telugu_data/raw/raw_talks_vk/*.wav 2>/dev/null | wc -l) files"
echo "   10TV Telugu: $(ls -1 /workspace/telugu_data/raw/10TV\ Telugu/*.wav 2>/dev/null | wc -l) files"
echo "   Sakshi TV: $(ls -1 /workspace/telugu_data/raw/Sakshi\ TV/*.wav 2>/dev/null | wc -l) files"
echo ""

echo "5. Sample File Check:"
file /workspace/telugu_data/raw/raw_talks_vk/*.wav | head -1
echo ""

echo "6. Directory Structure:"
tree -L 2 /workspace/telugu_data 2>/dev/null || ls -R /workspace/telugu_data
echo ""

echo "=================================="
echo "Verification Complete!"
echo "=================================="
```

**Save and run:**
```bash
# Copy the script above, then:
bash verify_data.sh
```

---

## Step 9: Check Collection Report

```bash
cat /workspace/NewProject/telugu_data/collection_report.json
```

**Expected Output:**
```json
{
  "total_hours_collected": 0.00,
  "total_segments": 0,
  "sources_processed": [
    {
      "name": "raw_talks_vk",
      "total_hours": 0,
      "total_segments": 0,
      "status": "completed"
    },
    {
      "name": "10TV Telugu",
      "total_hours": 0,
      "total_segments": 0,
      "status": "completed"
    },
    {
      "name": "Sakshi TV",
      "total_hours": 0,
      "total_segments": 0,
      "status": "completed"
    }
  ],
  "errors": []
}
```

---

## Step 10: Verify Audio Quality

```bash
# Check sample rate and bit depth of a few files
for file in /workspace/telugu_data/raw/raw_talks_vk/*.wav; do
    echo "File: $(basename "$file")"
    ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels,bits_per_sample -of default=noprint_wrappers=1 "$file"
    echo "---"
    break  # Just check first file
done
```

**Expected Output:**
```
File: EP - 110 MAD FUNNNN FT SUMA AKKAAA.wav
sample_rate=44100
channels=2
bits_per_sample=16
---
```

---

## ✅ Success Criteria

Your data is **VERIFIED and READY** if:

- ✓ Total size: **11-12 GB**
- ✓ Total files: **27 WAV files**
- ✓ Raw Talks VK: **10 files**
- ✓ 10TV Telugu: **9 files**
- ✓ Sakshi TV: **8 files**
- ✓ All files show "WAVE audio" format
- ✓ No error messages
- ✓ Files are not 0 bytes

---

## 🚨 If Something is Wrong

### If size is less than 10 GB:
```bash
# Check if download is still running
ps aux | grep data_collection

# Check logs
tail -50 /workspace/NewProject/data_collection.log
```

### If files are missing:
```bash
# Re-run data collection
cd /workspace/NewProject
python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100 \
    --quality "high"
```

### If files are corrupted (0 bytes):
```bash
# Find zero-byte files
find /workspace/telugu_data/raw -name "*.wav" -size 0

# Remove them and re-download
rm -f /workspace/telugu_data/raw/*/corrupted_file.wav
```

---

## 📊 Quick One-Liner Verification

```bash
echo "Total: $(du -sh /workspace/telugu_data | awk '{print $1}') | Files: $(find /workspace/telugu_data/raw -name '*.wav' | wc -l) | Raw Talks: $(ls -1 /workspace/telugu_data/raw/raw_talks_vk/*.wav 2>/dev/null | wc -l) | 10TV: $(ls -1 /workspace/telugu_data/raw/10TV\ Telugu/*.wav 2>/dev/null | wc -l) | Sakshi: $(ls -1 /workspace/telugu_data/raw/Sakshi\ TV/*.wav 2>/dev/null | wc -l)"
```

**Expected Output:**
```
Total: 12G | Files: 27 | Raw Talks: 10 | 10TV: 9 | Sakshi: 8
```

---

**Run these commands in your RunPod terminal and share the output to confirm everything is correct!**
