# 🚀 QUICKSTART GUIDE

## Complete Step-by-Step Instructions

Follow these steps **in order** on your RunPod instance.

---

## Step 1: Initial Setup (Run Once)

```bash
# SSH into RunPod or use the terminal
cd /workspace

# Clone the repository
git clone https://github.com/devasphn/NewProject.git
cd NewProject

# Run the setup script (installs everything)
bash setup_runpod.sh
```

**Time: ~10 minutes**

---

## Step 2: Login to HuggingFace

```bash
# Login (required for IndicVoices and Kathbath)
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

Then accept licenses at:
- https://huggingface.co/datasets/ai4bharat/IndicVoices
- https://huggingface.co/datasets/ai4bharat/Kathbath

---

## Step 3: Download Data

```bash
cd /workspace/NewProject

# Download wget-based data (LibriSpeech, Gramvaani, OpenSLR)
# Run in tmux/screen so it continues if disconnected
tmux new -s download

bash download_6000h_data.sh

# This downloads:
# - LibriSpeech: 960h English
# - Gramvaani: 1111h Hindi  
# - OpenSLR 66: 10h Telugu
```

**Time: 2-6 hours depending on internet speed**

```bash
# In another terminal, download HuggingFace data
python download_huggingface_data.py --all

# This downloads:
# - IndicVoices: 800h+ Hindi, 400h+ Telugu
# - Kathbath: 140h Hindi, 155h Telugu
# - Common Voice: 25h Telugu
```

**Time: 4-8 hours**

---

## Step 4: Augment Telugu Data (Fill the Gap)

```bash
# Telugu only has ~600h, need 2000h
# Augmentation expands 5x → ~3000h

python augment_telugu_data.py \
    --input_dir /workspace/data/telugu \
    --output_dir /workspace/data/telugu/augmented \
    --expansion 5x \
    --workers 8
```

**Time: 1-2 hours**

---

## Step 5: Verify Setup

```bash
python verify_setup.py
```

Should show all ✅ checks passing.

---

## Step 6: Start Codec Training

```bash
# Use tmux so training continues if you disconnect
tmux new -s train

python train_codec_production.py \
    --data_dirs /workspace/data/english /workspace/data/hindi /workspace/data/telugu \
    --batch_size 32 \
    --num_epochs 100 \
    --checkpoint_dir /workspace/checkpoints_codec \
    --num_workers 12

# Monitor with TensorBoard
tensorboard --logdir /workspace/checkpoints_codec --port 6006
```

**Time: 40-60 GPU hours**

---

## Step 7: Monitor Training

```bash
# Check GPU usage
nvtop

# Check training logs
tail -f /workspace/checkpoints_codec/training.log

# Attach to tmux session
tmux attach -t train
```

---

## Budget Estimate

| Phase | GPU Hours | Cost (H200 @ $3.39/hr) |
|-------|-----------|------------------------|
| Data Download | 0 | $0 |
| Codec Training | 50h | $170 |
| S2S Training | 80h | $271 |
| Fine-tuning | 30h | $102 |
| **Total** | **160h** | **~$543** |

Storage: ~$50/month for 500GB

**Total Project Cost: ~$600-800** (well under $1,500 budget)

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 16  # or even 8
```

### Slow Data Loading
```bash
# Increase workers (up to CPU count)
--num_workers 16
```

### Training Crashed
```bash
# Resume from checkpoint
python train_codec_production.py \
    --resume /workspace/checkpoints_codec/checkpoint_epoch_50.pt \
    ...
```

### Check Disk Space
```bash
df -h /workspace
# Need ~500GB for all data
```

---

## Files Created

| File | Purpose |
|------|---------|
| `setup_runpod.sh` | One-time system setup |
| `download_6000h_data.sh` | Download wget-based datasets |
| `download_huggingface_data.py` | Download HuggingFace datasets |
| `augment_telugu_data.py` | Expand Telugu data 5x |
| `verify_setup.py` | Pre-training verification |
| `train_codec_production.py` | Main training script |
| `codec_production.py` | Codec architecture |
| `discriminator_dac.py` | GAN discriminator |

---

## Success Criteria

After training, you should have:
- ✅ Codec checkpoint: `best_codec.pt` (~500-800MB)
- ✅ Codec SNR: >18 dB
- ✅ Inference latency: <100ms per batch
- ✅ Works on all 3 languages

---

## Next Steps After Codec

1. **Train S2S Model** - Use codec tokens as input/output
2. **Add Emotion** - Integrate emotion embeddings
3. **Deploy** - RunPod serverless with FastAPI
4. **Optimize** - Quantization, batching, caching

See `MASTER_PLAN_LUNA_EQUIVALENT.md` for full roadmap.
