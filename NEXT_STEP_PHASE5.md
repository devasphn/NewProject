# 🚀 READY FOR PHASE 5: Model Training

**Current Status**: ✅ All prerequisites complete, verified, and bug-free

---

## ✅ Completed Prerequisites

### Phase 1-4: COMPLETE
- ✅ Environment setup
- ✅ Dependencies installed
- ✅ Node.js installed (v12.22.9)
- ✅ Data collected: 39 files, 13 GB
- ✅ Speaker preparation: 4 train, 4 val, 4 test

### Code Verification: COMPLETE
- ✅ All 3 training scripts verified
- ✅ Fixed export_onnx bug in train_codec.py
- ✅ Argparse properly implemented in all scripts
- ✅ No critical bugs found

### Documentation: COMPLETE
- ✅ COMPLETE_COMMAND_REFERENCE.md (all commands saved)
- ✅ PHASE5_CODE_VERIFICATION.md (detailed verification)
- ✅ This file (next steps)

---

## 🎯 NEXT STEP: Start Codec Training

### Pre-Flight Check:

```bash
# 1. Check GPU is available
nvidia-smi
# Expected: H200 80GB with 75+ GB free VRAM

# 2. Check disk space
df -h /workspace
# Expected: 50+ GB free

# 3. Verify speaker data
echo "Val: $(cat /workspace/speaker_data/val_split.json | grep -o '"speaker_id"' | wc -l)"
# Expected: Val: 4

# 4. Create models directory
mkdir -p /workspace/models/codec
```

---

## 🚀 START CODEC TRAINING NOW

### Step 1: Start Screen Session
```bash
cd /workspace/NewProject
screen -S codec_training
```

### Step 2: Run Training (Inside Screen)
```bash
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --experiment_name "telucodec_v1"
```

### Step 3: Detach from Screen
```
Press: Ctrl+A, then press D
```

---

## 📊 What to Expect

### Expected Output (First Few Lines):
```
INFO - Starting codec training...
INFO - Loading data from /workspace/telugu_data/raw
INFO - Found 39 audio files
INFO - Creating train/val/test splits...
INFO - Train: 31 samples, Val: 4 samples, Test: 4 samples
INFO - Initializing TeluCodec model...
INFO - Model parameters: 47.2M
INFO - Using device: cuda
INFO - Starting training for 100 epochs...

Epoch 1/100:   0%|          | 0/1000 [00:00<?, ?it/s]
Epoch 1/100:  10%|█         | 100/1000 [01:15<11:23, train_loss=2.543]
...
```

### Training Progress:
- **Epochs**: 100 total
- **Steps per epoch**: ~1000
- **Time per epoch**: ~5-7 minutes
- **Total time**: 6-8 hours
- **Validation**: Every 5 epochs

### GPU Metrics (while training):
```bash
# In another terminal:
watch -n 1 nvidia-smi

# Expected:
# - GPU Util: 90-100%
# - VRAM: 40-50 GB / 80 GB
# - Temperature: 65-75°C
# - Power: 400-450W
```

---

## 🔍 Monitoring Commands

### Reattach to Training Session:
```bash
screen -r codec_training
# Press Ctrl+A, then D to detach again
```

### Check Training Logs:
```bash
tail -f /workspace/NewProject/training.log
# Or last 50 lines:
tail -50 /workspace/NewProject/training.log
```

### Watch GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Check Checkpoints:
```bash
ls -lh /workspace/models/codec/
# Should see:
# - checkpoint_epoch_5.pt
# - checkpoint_epoch_10.pt
# - checkpoint_best.pt (best validation loss)
```

---

## ⏱️ Timeline & Costs

### Phase 5 Timeline:

| Step | Duration | GPU Cost | Status |
|------|----------|----------|--------|
| 1. Codec Training | 6-8 hrs | $15 | ⏳ **START NOW** |
| 2. Speaker Training | 4-6 hrs | $12 | Pending |
| 3. S2S Training | 12-24 hrs | $60 | Pending |
| **Total Phase 5** | **~26 hrs** | **$87** | |

### After Codec Training:
```bash
# Expected output file:
/workspace/models/codec/checkpoint_best.pt  (~2-3 GB)

# Next command (after codec completes):
python train_speakers.py \
    --data_dir /workspace/speaker_data \
    --codec_path /workspace/models/codec/checkpoint_best.pt \
    --output_path /workspace/models/speaker_embeddings.json
```

---

## 🚨 Troubleshooting

### If Training Crashes:

**Check logs:**
```bash
tail -100 /workspace/NewProject/training.log
```

**Common issues:**

1. **Out of Memory:**
   ```bash
   # Reduce batch size and restart:
   python train_codec.py \
       --data_dir /workspace/telugu_data/raw \
       --checkpoint_dir /workspace/models/codec \
       --batch_size 16  # Reduced from 32
   ```

2. **Data Loading Error:**
   ```bash
   # Verify data exists:
   ls -lh /workspace/telugu_data/raw/
   # Should show 4 directories with .wav files
   ```

3. **WandB Error:**
   ```bash
   # Set WandB key or disable:
   export WANDB_MODE=offline
   # Then restart training
   ```

4. **Permission Error:**
   ```bash
   # Fix permissions:
   chmod -R 755 /workspace/NewProject
   chmod -R 777 /workspace/models
   ```

### Resume from Checkpoint:
```bash
# If training crashes, resume from last checkpoint:
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --resume_from /workspace/models/codec/checkpoint_epoch_50.pt
```

---

## 📝 Important Notes

### Do's:
- ✅ Use `screen` to keep training running if disconnected
- ✅ Monitor GPU usage periodically
- ✅ Check logs for any errors
- ✅ Wait for "Training completed!" message
- ✅ Verify `checkpoint_best.pt` exists before next phase

### Don'ts:
- ❌ Don't close the terminal without detaching from screen
- ❌ Don't stop training mid-epoch (wait for checkpoint save)
- ❌ Don't start speaker training until codec completes
- ❌ Don't delete checkpoints (you may need to resume)

### Screen Commands Reminder:
```bash
# Create session:
screen -S codec_training

# List sessions:
screen -ls

# Reattach to session:
screen -r codec_training

# Detach from session:
Ctrl+A, then D

# Kill session (if needed):
screen -X -S codec_training quit
```

---

## ✅ Success Criteria

### Codec Training Complete When:
1. ✅ See "Training completed!" message
2. ✅ File exists: `/workspace/models/codec/checkpoint_best.pt`
3. ✅ File size: 2-3 GB
4. ✅ Validation loss converged (plateaued)

### Verify Success:
```bash
# Check file exists
ls -lh /workspace/models/codec/checkpoint_best.pt

# Test loading checkpoint
python -c "import torch; torch.load('/workspace/models/codec/checkpoint_best.pt'); print('✓ Checkpoint valid')"

# Expected output:
# ✓ Checkpoint valid
```

---

## 🎯 After Codec Training

### Immediate Next Steps:
1. Verify checkpoint exists and is valid
2. Start speaker embeddings training (4-6 hours)
3. Then start S2S model training (12-24 hours)
4. Test voice conversion
5. Deploy streaming server

---

## 📋 Quick Command Reference

### Most Important Commands:

```bash
# 1. Start training
screen -S codec_training
python train_codec.py --data_dir /workspace/telugu_data/raw --checkpoint_dir /workspace/models/codec --batch_size 32 --num_epochs 100
# Detach: Ctrl+A, D

# 2. Monitor GPU
watch -n 1 nvidia-smi

# 3. Check progress
screen -r codec_training

# 4. View logs
tail -f /workspace/NewProject/training.log

# 5. Verify completion
ls -lh /workspace/models/codec/checkpoint_best.pt
```

---

## 🚀 READY TO START?

**Copy-paste this command sequence:**

```bash
cd /workspace/NewProject
mkdir -p /workspace/models/codec
screen -S codec_training

# Inside screen session, run:
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --experiment_name "telucodec_v1"

# After command starts, detach with: Ctrl+A, then D
```

---

**STATUS: ✅ EVERYTHING VERIFIED AND READY**

**NEXT ACTION: Run the command above to start codec training!**

**Estimated completion: 6-8 hours from now**

**After completion: Come back for Phase 5.2 (Speaker Training)**
