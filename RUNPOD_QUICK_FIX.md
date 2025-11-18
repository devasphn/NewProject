# 🔧 Quick Fix for Speaker Embeddings Error

## Error
```
✗ Error downloading speaker embeddings: Invalid key: 8051 is out of bounds for size 7931
```

## Fix

The speaker IDs in the code were out of range. I've updated them to valid indices.

### Option 1: Pull from GitHub (Easiest)

```bash
cd /workspace/NewProject
git pull origin main
bash startup.sh
```

### Option 2: Quick Manual Fix (If can't pull)

Edit the file directly on RunPod:

```bash
nano /workspace/NewProject/download_models.py
```

Find line with:
```python
speaker_ids = [7306, 8051, 9017, 9166]  # Different speakers
```

Replace with:
```python
speaker_ids = [100, 2000, 4000, 6000]  # Valid indices within dataset
```

Save (Ctrl+O, Enter, Ctrl+X) and run:
```bash
bash startup.sh
```

---

## What Was Wrong

The CMU Arctic dataset has only **7931 samples**, but the code was trying to access indices like 8051 and 9166 which don't exist.

## What Was Fixed

Changed speaker IDs to valid indices:
- ❌ Old: [7306, 8051, 9017, 9166]
- ✅ New: [100, 2000, 4000, 6000]

These new indices are spread across the dataset to get diverse speaker voices.

---

## After Fix Succeeds

You'll see:
```
[5/5] Downloading speaker embeddings...
Dataset size: 7931 samples
✓ Downloaded 4 speaker embeddings

============================================================
All models downloaded successfully!
============================================================
```

Then the script will continue to latency testing!
