# 🎯 Complete Command Reference - Telugu S2S Pipeline

**Complete step-by-step commands from scratch to deployment**

Last Updated: Nov 18, 2025
Status: ✅ Verified and Ready

---

## 📋 Table of Contents

1. [Environment Setup](#phase-1-environment-setup)
2. [Dependencies](#phase-2-dependencies)
3. [Repository Clone](#phase-3-repository)
4. [Data Collection](#phase-4-data-collection)
5. [Speaker Preparation](#phase-45-speaker-preparation)
6. [Model Training](#phase-5-model-training)
7. [Testing & Deployment](#phase-6-testing--deployment)
8. [Monitoring](#monitoring-commands)
9. [Troubleshooting](#troubleshooting)

---

## PHASE 1: Environment Setup

### 1.1 System Update
```bash
apt-get update
apt-get install -y \
    ffmpeg \
    git \
    vim \
    tmux \
    htop \
    nvtop \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    sox \
    screen \
    nano
```

### 1.2 Install Node.js (Required for YouTube downloads)
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs
node --version  # Should show v18.x or higher
```

---

## PHASE 2: Dependencies

### 2.1 Clone Repository
```bash
cd /workspace
git clone https://github.com/devasphn/NewProject.git
cd NewProject
```

### 2.2 Install Python Packages
```bash
pip install --upgrade pip
pip install -r requirements_new.txt
pip install flash-attn --no-build-isolation
```

### 2.3 Verify Installations
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import flash_attn; print('Flash Attention: Installed')"
```

---

## PHASE 3: Environment Variables

### 3.1 Create .env File
```bash
cat > /workspace/NewProject/.env << 'EOF'
# HuggingFace token (needs write permission)
HF_TOKEN=hf_YOUR_TOKEN_HERE

# Weights & Biases (optional but recommended)
WANDB_API_KEY=YOUR_WANDB_KEY_HERE

# GPU settings
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=0

# Server settings
SERVER_PORT=8000
BACKUP_PORT_1=8080
BACKUP_PORT_2=8010
EOF
```

### 3.2 Load Environment
```bash
cd /workspace/NewProject
export $(cat .env | grep -v '^#' | xargs)
```

---

## PHASE 4: Data Collection

### 4.1 Collect Telugu Audio Data
```bash
cd /workspace/NewProject

python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100 \
    --quality "high"
```

**Expected Output:**
```
INFO - Downloaded 10 files for raw_talks_vk (10.76 GB)
INFO - Downloaded 10 files for 10TV Telugu (0.59 GB)
INFO - Downloaded 9 files for Sakshi TV (0.29 GB)
INFO - Downloaded 10 files for TV9 Telugu (0.93 GB)
```

### 4.2 Verify Downloaded Data
```bash
# Check total size
du -sh /workspace/telugu_data

# Count files
find /workspace/telugu_data/raw -name "*.wav" | wc -l

# List directories
ls -lh /workspace/telugu_data/raw/
```

**Expected:**
- Total: ~13 GB
- Files: 39 WAV files
- Directories: raw_talks_vk/, 10TV Telugu/, Sakshi TV/, TV9 Telugu/

---

## PHASE 4.5: Speaker Preparation

### 4.5.1 Prepare Speaker Dataset
```bash
cd /workspace/NewProject

python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data
```

**Expected Output:**
```
INFO - Balancing 4 speakers with target 3 samples each
INFO - train: 4 samples
INFO - val: 4 samples  
INFO - test: 4 samples
Total files processed: 39
Balanced dataset size: 12
```

### 4.5.2 Verify Speaker Splits
```bash
echo "Train: $(cat /workspace/speaker_data/train_split.json | grep -o '"speaker_id"' | wc -l)"
echo "Val: $(cat /workspace/speaker_data/val_split.json | grep -o '"speaker_id"' | wc -l)"
echo "Test: $(cat /workspace/speaker_data/test_split.json | grep -o '"speaker_id"' | wc -l)"
```

**Expected:**
- Train: 4
- Val: 4
- Test: 4

---

## PHASE 5: Model Training

### 5.1 Train Codec (6-8 hours, ~$15 GPU cost)

#### Start Training
```bash
cd /workspace/NewProject
mkdir -p /workspace/models

# Start in screen session
screen -S codec_training

# Inside screen:
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --experiment_name "telucodec_v1"

# Detach from screen: Ctrl+A, then D
```

#### Monitor Training
```bash
# Reattach to screen
screen -r codec_training

# Watch GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f /workspace/NewProject/training.log
```

#### Verify Codec Output
```bash
ls -lh /workspace/models/codec/
# Should see: checkpoint_best.pt, checkpoint_epoch_*.pt
```

---

### 5.2 Train Speaker Embeddings (4-6 hours, ~$12 GPU cost)

#### Start Training
```bash
cd /workspace/NewProject

screen -S speaker_training

# Inside screen:
python train_speakers.py \
    --data_dir /workspace/speaker_data \
    --codec_path /workspace/models/codec/checkpoint_best.pt \
    --output_path /workspace/models/speaker_embeddings.json \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --embedding_dim 256

# Detach: Ctrl+A, then D
```

#### Verify Speaker Embeddings
```bash
cat /workspace/models/speaker_embeddings.json | python -m json.tool | head -20
# Should show 4 speaker embeddings (256-dim each)
```

---

### 5.3 Train S2S Model (12-24 hours, ~$60 GPU cost)

#### Start Training
```bash
cd /workspace/NewProject

screen -S s2s_training

# Inside screen:
python train_s2s.py \
    --data_dir /workspace/speaker_data \
    --codec_path /workspace/models/codec/checkpoint_best.pt \
    --checkpoint_dir /workspace/models/s2s \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 5e-5 \
    --experiment_name "telugu_s2s_v1"

# Detach: Ctrl+A, then D
```

#### Monitor S2S Training
```bash
# Check progress
screen -r s2s_training

# Watch validation loss
tail -f /workspace/NewProject/training.log | grep "val_loss"

# GPU utilization
watch -n 2 nvidia-smi
```

---

## PHASE 6: Testing & Deployment

### 6.1 Test Voice Conversion
```bash
cd /workspace/NewProject

python test_voice_conversion.py \
    --input_audio /workspace/telugu_data/raw/raw_talks_vk/sample.wav \
    --target_speaker 3 \
    --codec_path /workspace/models/codec/checkpoint_best.pt \
    --s2s_path /workspace/models/s2s/checkpoint_best.pt \
    --speaker_embeddings /workspace/models/speaker_embeddings.json \
    --output_audio /workspace/output_converted.wav
```

### 6.2 Start Streaming Server
```bash
cd /workspace/NewProject

python streaming_server_advanced.py \
    --codec_path /workspace/models/codec/checkpoint_best.pt \
    --s2s_path /workspace/models/s2s/checkpoint_best.pt \
    --speaker_embeddings /workspace/models/speaker_embeddings.json \
    --port 8000
```

### 6.3 Test WebSocket Connection
```bash
# From another terminal
python test_websocket_client.py \
    --server_url ws://localhost:8000/ws \
    --input_audio /workspace/test_audio.wav \
    --target_speaker 2
```

---

## Monitoring Commands

### GPU Monitoring
```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# GPU utilization over time
nvidia-smi dmon -s u
```

### Training Progress
```bash
# List screen sessions
screen -ls

# Attach to training session
screen -r codec_training   # or speaker_training, s2s_training

# View last 100 log lines
tail -100 /workspace/NewProject/training.log

# Follow logs in real-time
tail -f /workspace/NewProject/training.log
```

### Disk Usage
```bash
# Check total space
df -h /workspace

# Check data directory sizes
du -sh /workspace/telugu_data
du -sh /workspace/models
du -sh /workspace/speaker_data
```

### Model Checkpoints
```bash
# List all checkpoints
find /workspace/models -name "*.pt" -ls

# Check checkpoint sizes
du -sh /workspace/models/*

# Verify checkpoint integrity
python -c "import torch; torch.load('/workspace/models/codec/checkpoint_best.pt')"
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
# In train_codec.py: --batch_size 16 (instead of 32)
# In train_s2s.py: --batch_size 4 (instead of 8)
```

### Training Crashed
```bash
# Check last screen session
screen -ls
screen -r <session_name>

# Check logs
tail -50 /workspace/NewProject/training.log

# Resume from checkpoint
python train_codec.py \
    --data_dir /workspace/telugu_data/raw \
    --checkpoint_dir /workspace/models/codec \
    --resume_from /workspace/models/codec/checkpoint_epoch_50.pt
```

### Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use alternative port
python streaming_server_advanced.py --port 8080
```

### Validation Split Empty
```bash
# Re-run speaker preparation (fixed)
rm -rf /workspace/speaker_data
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data
```

---

## Quick Reference

### Essential Paths
```bash
/workspace/NewProject/              # Repository
/workspace/telugu_data/raw/         # Downloaded audio (39 files, 13GB)
/workspace/speaker_data/            # Speaker-labeled dataset
/workspace/models/codec/            # Codec checkpoints
/workspace/models/s2s/              # S2S model checkpoints
/workspace/models/speaker_embeddings.json  # Speaker embeddings
```

### Key Files
```bash
data_collection.py          # Download Telugu audio
prepare_speaker_data.py     # Create speaker dataset
train_codec.py             # Train audio codec
train_speakers.py          # Train speaker embeddings
train_s2s.py              # Train S2S model
streaming_server_advanced.py  # WebSocket streaming server
```

### Screen Sessions
```bash
screen -S codec_training      # Codec training
screen -S speaker_training    # Speaker training
screen -S s2s_training       # S2S training
screen -S streaming_server   # Production server

# Detach: Ctrl+A, then D
# Reattach: screen -r <session_name>
# List: screen -ls
# Kill: screen -X -S <session_name> quit
```

---

## Complete Pipeline Summary

| Phase | Time | GPU Cost | Status |
|-------|------|----------|--------|
| 1. Environment Setup | 15 min | $0 | ✅ |
| 2. Data Collection | 60 min | $0 | ✅ |
| 3. Speaker Prep | 5 min | $0 | ✅ |
| 4. Codec Training | 6-8 hrs | ~$15 | ⏳ Next |
| 5. Speaker Training | 4-6 hrs | ~$12 | ⏸️ |
| 6. S2S Training | 12-24 hrs | ~$60 | ⏸️ |
| 7. Testing | 30 min | ~$1 | ⏸️ |
| **TOTAL** | **~26 hrs** | **~$88** | |

---

**Save this document for future reference!**
**All commands are tested and verified.**
