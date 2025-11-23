# 🚀 RunPod Template Setup - Telugu S2S System

## 📊 Storage Requirements Analysis (80GB Data)

### Current Data Size
- **Raw Data:** 80GB (YouTube videos)
- **Extracted Audio:** ~80GB (WAV files at 16kHz)
- **Total Source Data:** ~160GB

### Training & Model Storage
- **Codec Models:** ~10GB (checkpoints, best model, EMA)
- **S2S Models:** ~20GB (checkpoints, best model, intermediate)
- **Speaker Models:** ~5GB (embeddings, checkpoints)
- **Training Artifacts:** ~30GB (optimizer states, gradients, temp files)
- **Logs & Monitoring:** ~20GB (wandb, tensorboard, JSON logs)

### Total Storage Needed
```
Raw data:          80GB
Extracted audio:   80GB
Models:            35GB
Processing:        30GB
Logs:              20GB
Buffer:            50GB (safety margin)
─────────────────────────
TOTAL:            295GB
```

### ✅ Recommendation
- **Container Disk: 300GB** ✓ SUFFICIENT (but tight - 95% used)
- **Volume Disk: 500GB** ✓ EXCELLENT (plenty of room for scaling)

**ALTERNATIVE (safer):**
- **Container Disk: 400GB** (leaves 25% buffer)
- **Volume Disk: 500GB** (unchanged)

---

## 🔧 RunPod Template Configuration

### Template Details
```
Name: Telugu S2S Production Template
Base Image: runpod/pytorch:2.2.0
GPU: H100 80GB (or A100 80GB)
Container Disk: 400GB (recommended) or 300GB (minimum)
Volume Disk: 500GB
Expose Ports: 8000, 8080, 6006 (tensorboard)
```

### Why `pytorch:2.2.0`?
✅ **Stable** - Well-tested base image  
✅ **CUDA 12.1** - Compatible with H100/A100  
✅ **Python 3.10** - Stable for all dependencies  
✅ **Flash Attention** - Compiles successfully  
✅ **PyTorch 2.2.0** - Has `torch.compile()` support  

**Alternative:** `pytorch:2.4.0` (newer, but 2.2.0 is proven stable)

---

## 🌍 Environment Variables

### Required (Set in RunPod Template)
```bash
# HuggingFace (for downloading WhisperX, models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Weights & Biases (for training monitoring)
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxx

# Project Configuration
PROJECT_NAME=telugu-s2s-production
WANDB_PROJECT=telugu-s2s

# CUDA Optimization
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# Flash Attention
FLASH_ATTENTION_FORCE_BUILD=0

# Data Paths
DATA_DIR=/workspace/telugu_data
MODEL_DIR=/workspace/models
CHECKPOINT_DIR=/workspace/checkpoints
```

### Optional (for optimization)
```bash
# Mixed Precision Training
NVIDIA_TF32_OVERRIDE=1
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Distributed Training (if using multiple GPUs)
MASTER_ADDR=localhost
MASTER_PORT=29500
WORLD_SIZE=1
RANK=0

# Node.js (if needed for web UI development)
NODE_VERSION=20
```

---

## 📜 Start Container Command

**CRITICAL:** This runs ONCE when pod starts, installing everything automatically.

### Option 1: Basic Setup (Fast, ~10 minutes)
```bash
cd /workspace && \
git clone https://github.com/devasphn/NewProject.git && \
cd NewProject && \
pip install --upgrade pip && \
pip install -r requirements_new.txt && \
pip install flash-attn --no-build-isolation && \
apt-get update && apt-get install -y ffmpeg nodejs npm && \
mkdir -p /workspace/telugu_data /workspace/models /workspace/checkpoints && \
echo "✅ Setup complete! Ready to start data collection."
```

### Option 2: Complete Setup (Thorough, ~20 minutes)
```bash
#!/bin/bash
set -e

echo "🚀 Starting Telugu S2S Production Setup..."

# Update system packages
apt-get update
apt-get install -y ffmpeg sox libsox-fmt-all nodejs npm curl wget git-lfs

# Install Node.js 20 (LTS)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Navigate to workspace
cd /workspace

# Clone repository
if [ ! -d "NewProject" ]; then
    git clone https://github.com/devasphn/NewProject.git
fi

cd NewProject

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements_new.txt

# Install Flash Attention (critical for speed)
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install audio processing tools
pip install yt-dlp ffmpeg-python pydub soundfile librosa

# Create directory structure
mkdir -p /workspace/telugu_data/{raw,processed,train,val,test}
mkdir -p /workspace/models/{codec,s2s,speaker}
mkdir -p /workspace/checkpoints/{codec,s2s,speaker}
mkdir -p /workspace/logs

# Verify installations
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Verification:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "from flash_attn import flash_attn_func; print('Flash Attention: ✓')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import whisperx; print('WhisperX: ✓')"
echo "Node.js: $(node --version)"
echo "npm: $(npm --version)"
echo "ffmpeg: $(ffmpeg -version | head -n1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "✅ Setup Complete! Ready for data collection."
echo ""
echo "Next steps:"
echo "1. Login to HuggingFace: huggingface-cli login"
echo "2. Login to WandB: wandb login"
echo "3. Start data collection: python download_telugu_data_PRODUCTION.py"
echo ""
```

**Choose Option 2 for production** - it's more thorough and handles all edge cases.

---

## 🎯 How to Set Up Template in RunPod

### Step 1: Go to RunPod Dashboard
1. Navigate to https://www.runpod.io
2. Click **"Templates"** in sidebar
3. Click **"New Template"**

### Step 2: Configure Template

```yaml
Template Name: Telugu-S2S-Production-v1

Container Image: runpod/pytorch:2.2.0

Docker Command: (leave blank - use start script instead)

Container Disk: 400 GB

Expose HTTP Ports: 8000,8080,6006

Expose TCP Ports: (leave blank)

Environment Variables:
  HF_TOKEN: <your_huggingface_token>
  WANDB_API_KEY: <your_wandb_key>
  PROJECT_NAME: telugu-s2s-production
  WANDB_PROJECT: telugu-s2s
  CUDA_VISIBLE_DEVICES: 0
  PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:512
  TOKENIZERS_PARALLELISM: false
  DATA_DIR: /workspace/telugu_data
  MODEL_DIR: /workspace/models
  CHECKPOINT_DIR: /workspace/checkpoints

Start Script (Option 2 - Complete Setup):
  <paste the complete setup script from Option 2 above>
```

### Step 3: Deploy Pod

1. Click **"Save Template"**
2. Go to **"Pods"** → **"Deploy"**
3. Select your **"Telugu-S2S-Production-v1"** template
4. Choose GPU: **H100 80GB** (or A100 80GB)
5. Volume Disk: **500 GB**
6. Click **"Deploy On-Demand"**

### Step 4: Wait for Setup

- Initial setup: **~20 minutes** (runs automatically)
- Watch progress in **"Logs"** tab
- When you see **"✅ Setup Complete!"**, you're ready!

---

## 🔐 Getting API Keys

### HuggingFace Token
```bash
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "Telugu S2S Production"
4. Type: "Write"
5. Copy token: hf_xxxxxxxxxxxxxxxxxxxx
```

### Weights & Biases Token
```bash
1. Go to: https://wandb.ai/authorize
2. Copy your API key
3. Or run: wandb login
```

---

## ✅ Post-Setup Verification

### After pod starts, run these commands:

```bash
# Connect to pod
# Click "Connect" → "Start Web Terminal"

# Verify directory structure
ls -lh /workspace/
# Should see: NewProject/, telugu_data/, models/, checkpoints/, logs/

# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "from flash_attn import flash_attn_func; print('✓')"
python -c "import whisperx; print('✓')"

# Check GPU
nvidia-smi

# Verify code files
cd /workspace/NewProject
ls -la
# Should see: telugu_codec_fixed.py, train_codec_dac.py, discriminator_dac.py, etc.

# Login to services
huggingface-cli login
wandb login
```

---

## 📋 Complete Workflow After Setup

### Phase 1: Data Collection (Day 1-2)
```bash
cd /workspace/NewProject

# If you already have 80GB data on your PC:
# Upload to RunPod volume or use cloud storage
# Otherwise, download fresh data:

python download_telugu_data_PRODUCTION.py \
  --output_dir /workspace/telugu_data/raw \
  --config data_sources_PRODUCTION.yaml \
  --max_duration 3600 \
  --num_workers 4

# Monitor progress
watch -n 60 du -sh /workspace/telugu_data/raw
```

**Expected:** 80-100GB raw data

### Phase 2: Data Preparation (Day 2)
```bash
# Extract and prepare audio
python prepare_speaker_data.py \
  --data_dir /workspace/telugu_data/raw \
  --output_dir /workspace/telugu_data/processed \
  --sample_rate 16000

# Verify dataset
python -c "
from pathlib import Path
import json
data_dir = Path('/workspace/telugu_data/processed')
for split in ['train', 'val', 'test']:
    files = list(data_dir.glob(f'{split}/**/*.wav'))
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f'{split}: {len(files)} files, {total_gb:.1f}GB')
"
```

**Expected:** ~160GB processed audio (train/val/test splits)

### Phase 3: Codec Training (Day 3-7)
```bash
# Start codec training
python train_codec_dac.py \
  --data_dir /workspace/telugu_data/processed/train \
  --output_dir /workspace/models/codec \
  --batch_size 16 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --discriminator_start_epoch 5 \
  --use_wandb

# Monitor in WandB dashboard
# Or TensorBoard:
tensorboard --logdir /workspace/checkpoints/codec --port 6006
```

**Expected:** >25 dB SNR after 50-70 epochs

### Phase 4: S2S Training (Day 8-14)
```bash
# Train S2S model
python train_s2s.py \
  --codec_path /workspace/models/codec/best_model.pt \
  --data_dir /workspace/telugu_data/processed \
  --output_dir /workspace/models/s2s \
  --batch_size 8 \
  --epochs 50 \
  --use_wandb
```

### Phase 5: Deployment & Testing (Day 15+)
```bash
# Start streaming server
python streaming_server_advanced.py \
  --codec_path /workspace/models/codec/best_model.pt \
  --s2s_path /workspace/models/s2s/best_model.pt \
  --host 0.0.0.0 \
  --port 8000

# Access web UI
# https://your-pod-id-8000.proxy.runpod.net
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Flash Attention Build Fails
```bash
# Solution: Install with fewer parallel jobs
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

### Issue 2: Out of Memory During Training
```bash
# Solution: Reduce batch size
python train_codec_dac.py --batch_size 8  # instead of 16
```

### Issue 3: CUDA Out of Memory
```bash
# Solution: Clear cache between runs
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue 4: Git Clone Fails
```bash
# Solution: Use personal access token
git clone https://YOUR_TOKEN@github.com/devasphn/NewProject.git
```

### Issue 5: 80GB Data Upload to RunPod
```bash
# Option 1: Upload via RunPod network storage (slow)
# Option 2: Use cloud storage
# Upload to Google Drive/S3, then download on pod:

# Google Drive (using gdown)
pip install gdown
gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# AWS S3
aws s3 sync s3://your-bucket/telugu-data /workspace/telugu_data/

# Option 3: Re-download on pod (if you have URLs)
python download_telugu_data_PRODUCTION.py
```

---

## 💰 Cost Estimate

### H100 80GB Pod
- **Rate:** $3.89/hour (on-demand)
- **Setup:** 0.5 hours = $2
- **Data collection:** 4 hours = $16
- **Codec training:** 120 hours = $467
- **S2S training:** 100 hours = $389
- **Testing:** 10 hours = $39
- **TOTAL:** ~$913

### A100 80GB Pod (Alternative)
- **Rate:** $1.89/hour
- **Total time:** ~300 hours (slower)
- **TOTAL:** ~$567

**Recommendation:** Use H100 for faster training, A100 for cost savings

---

## 🎯 Success Criteria

### Codec Training
- ✅ SNR > 25 dB
- ✅ Compression ratio ~80:1 (16kHz → 200Hz)
- ✅ Encoding latency < 10ms
- ✅ Decoding latency < 10ms

### S2S Model
- ✅ First-token latency < 150ms
- ✅ Streaming generation < 50ms per token
- ✅ Natural voice quality
- ✅ Context-aware responses

### Full System
- ✅ End-to-end latency < 400ms
- ✅ Full-duplex streaming works
- ✅ Interruption handling works
- ✅ 4 speakers with distinct voices

---

## 📞 Final Checklist

Before deploying pod, ensure you have:

- ✅ HuggingFace token (write access)
- ✅ WandB API key
- ✅ GitHub repo is clean (only correct files)
- ✅ 80GB data ready (on PC or cloud)
- ✅ Budget: ~$900 for H100 or ~$600 for A100
- ✅ Time: 14-21 days

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Create RunPod template with Option 2 start script
# 2. Deploy H100 80GB pod with 400GB container + 500GB volume
# 3. Wait ~20 mins for setup
# 4. Connect to pod terminal

cd /workspace/NewProject
huggingface-cli login
wandb login

# 5. Upload or download data (80GB)
# 6. Start codec training
python train_codec_dac.py --data_dir /workspace/telugu_data/processed/train --batch_size 16 --epochs 100

# 7. Wait 5-7 days
# 8. Start S2S training
# 9. Deploy server
# 10. Demo to MD! 🎉
```

---

**Ready to start production training!** 🚀
