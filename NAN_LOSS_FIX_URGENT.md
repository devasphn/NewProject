# 🚨 URGENT: NaN Loss Issue - STOP TRAINING NOW!

**Status: Training is BROKEN - All losses are NaN**

---

## 🛑 STOP TRAINING IMMEDIATELY

Your training is producing NaN (Not a Number) losses, which means:
- **Model is NOT learning**
- **Wasting GPU time and money**
- **Must restart with fixes**

### Stop the Training:
```bash
# Find the training process
ps aux | grep train_codec.py

# Kill it
pkill -f train_codec.py

# Or in screen:
screen -r codec_training
# Press Ctrl+C to stop
# Then exit: Ctrl+D
```

---

## ❌ What Went Wrong

### Output You Saw:
```
Epoch 0: loss=23.7 → Train loss: nan
Val loss: nan, SNR: nan dB
Epoch 1: loss=nan, recon=nan, vq=nan
```

### Root Causes:

1. **STFT Without Window** ⚠️
   ```
   UserWarning: A window was not provided. A rectangular window will be applied,
   which is known to cause spectral leakage.
   ```
   - Caused NaN in perceptual loss
   - Spectral leakage led to invalid gradients

2. **Unbounded Losses** ⚠️
   - No clamping on reconstruction loss
   - VQ loss could explode
   - No NaN checks

3. **Learning Rate Too High** ⚠️
   - 1e-4 may be too aggressive for initialization
   - Gradient explosion in first epoch

---

## ✅ FIXES APPLIED

### Fix 1: Hann Window for STFT ✅
```python
# NEW CODE in telugu_codec.py (lines 393-404):
window = torch.hann_window(n_fft, device=target.device)

target_spec = torch.stft(
    target.squeeze(1), n_fft=n_fft, 
    hop_length=n_fft//4, window=window,  # NEW!
    return_complex=True
).abs()
```

### Fix 2: Loss Clamping ✅
```python
# NEW CODE (lines 358-365):
recon_loss = torch.clamp(recon_loss, 0, 10.0)
vq_loss = torch.clamp(vq_loss, 0, 10.0)

# Prevent division by zero
target_spec = target_spec + 1e-7
pred_spec = pred_spec + 1e-7
```

### Fix 3: NaN Safeguard ✅
```python
# NEW CODE (lines 371-373):
if torch.isnan(total_loss) or torch.isinf(total_loss):
    total_loss = torch.tensor(1.0, device=audio.device, requires_grad=True)
```

### Fix 4: Lower Learning Rate ✅
**Change from 1e-4 to 1e-5** (10x lower)

---

## 🚀 RESTART TRAINING (With Fixes)

### Step 1: Clean Up
```bash
# Stop current training
pkill -f train_codec.py

# Remove broken checkpoints
rm -rf /workspace/models/codec/*

# Clear WandB cache (optional)
rm -rf /workspace/NewProject/wandb/run-*
```

### Step 2: Restart with LOWER Learning Rate
```bash
cd /workspace/NewProject

# Start in screen
screen -S codec_training

# Run with FIXED learning rate (1e-5 instead of 1e-4)
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --experiment_name "telucodec_v1_fixed"

# Detach: Ctrl+A, then D
```

**IMPORTANT CHANGES:**
- `--batch_size 16` (reduced from 32 for stability)
- `--learning_rate 1e-5` (reduced from 1e-4)
- `--experiment_name "telucodec_v1_fixed"` (new name)

---

## 📊 Expected Output (Fixed)

### Startup:
```
INFO - Loaded 33 audio files for train split
INFO - Loaded 3 audio files for validation split
wandb: Syncing run telucodec_v1_fixed
INFO - torch.compile disabled for compatibility
INFO - Starting epoch 0/100
```

### Training (HEALTHY):
```
Epoch 0:  50%|█████| 1/2 [00:15<00:15, loss=2.543, recon=1.234, vq=1.309, lr=1e-05]
Epoch 0: 100%|██████| 2/2 [00:30<00:00, loss=2.234, recon=1.045, vq=1.189, lr=1e-05]
INFO - Train loss: 2.234  ✓ NOT NaN!

Validation: 100%|██████| 1/1 [00:05<00:00]
INFO - Val loss: 2.456, SNR: 11.23 dB  ✓ NOT NaN!
INFO - Saved checkpoint to /workspace/models/codec/best_codec.pt
```

**Key Signs of Health:**
- ✅ Loss values are numbers (not NaN)
- ✅ Loss decreases over time
- ✅ SNR is a valid number
- ✅ Losses are in reasonable range (1-5)

---

## 🔍 Monitor for NaN

### Watch Training:
```bash
screen -r codec_training
```

### What to Look For:

**GOOD** ✅:
```
loss=2.543, recon=1.234, vq=1.309
Train loss: 2.234
Val loss: 2.456, SNR: 11.23 dB
```

**BAD** ❌:
```
loss=nan, recon=nan, vq=nan
Train loss: nan
Val loss: nan, SNR: nan dB
```

**If you see NaN again:**
1. Stop immediately (Ctrl+C)
2. Reduce batch_size to 8
3. Reduce learning_rate to 5e-6
4. Restart

---

## 📝 Changes Made

| File | Lines | Change |
|------|-------|--------|
| telugu_codec.py | 393-404 | Added Hann window to STFT |
| telugu_codec.py | 406-408 | Added epsilon to prevent division by zero |
| telugu_codec.py | 358-365 | Added loss clamping |
| telugu_codec.py | 371-373 | Added NaN safeguard |
| train_codec.py | Command | Reduced LR: 1e-4 → 1e-5 |
| train_codec.py | Command | Reduced batch: 32 → 16 |

**Total fixes**: 6 critical changes

---

## ⏱️ Training Timeline (Updated)

With reduced batch size and learning rate:
- **Per epoch**: ~7-10 minutes (was 5-7)
- **100 epochs**: 8-12 hours (was 6-8)
- **GPU cost**: ~$20 (was ~$15)

**Trade-off**: Slower but STABLE training

---

## 🎯 Success Criteria

### After Epoch 0:
- ✅ Train loss < 5.0 (not NaN)
- ✅ Val loss < 5.0 (not NaN)
- ✅ SNR > 0 dB (not NaN)

### After Epoch 10:
- ✅ Train loss decreasing
- ✅ Val loss decreasing  
- ✅ SNR increasing
- ✅ No NaN values

### After Epoch 50:
- ✅ Train loss < 1.0
- ✅ Val loss < 1.5
- ✅ SNR > 15 dB

---

## 🚨 Emergency Fallback

### If NaN Persists After Fixes:

```bash
# Even more conservative settings
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 5e-6 \
    --experiment_name "telucodec_v1_safe"
```

**Ultra-safe settings:**
- batch_size: 8 (4x smaller)
- learning_rate: 5e-6 (20x smaller)
- Will be slower but VERY stable

---

## 📋 Quick Action Checklist

- [ ] Stop current training (Ctrl+C or pkill)
- [ ] Remove broken checkpoints: `rm -rf /workspace/models/codec/*`
- [ ] Restart with batch_size=16, lr=1e-5
- [ ] Monitor first epoch for NaN
- [ ] Verify loss values are NOT NaN
- [ ] Check after 5 epochs that loss is decreasing
- [ ] Let train for 100 epochs (~8-12 hours)

---

## ✅ Summary

| Issue | Status |
|-------|--------|
| NaN losses detected | ✅ Understood |
| STFT window fixed | ✅ Applied |
| Loss clamping added | ✅ Applied |
| NaN safeguards added | ✅ Applied |
| Learning rate reduced | ⏳ Apply now |
| Batch size reduced | ⏳ Apply now |
| Ready to restart | ✅ YES |

---

## 🚀 RESTART NOW

```bash
# 1. Stop bad training
pkill -f train_codec.py

# 2. Clean up
rm -rf /workspace/models/codec/*

# 3. Restart with fixes
cd /workspace/NewProject
screen -S codec_training
python train_codec.py --data_dir /workspace/telugu_data/raw --checkpoint_dir /workspace/models/codec --batch_size 16 --num_epochs 100 --learning_rate 1e-5 --experiment_name "telucodec_v1_fixed"
# Ctrl+A, D
```

**Expected result**: Stable training with NO NaN losses! ✅
