# 🚀 RunPod Setup Commands - Telugu Voice AI POC

Complete command reference for setting up and running the Telugu Voice AI system on RunPod.

---

## 📋 Table of Contents

1. [Initial Setup](#1-initial-setup)
2. [Install Dependencies](#2-install-dependencies)
3. [Download Datasets](#3-download-datasets)
4. [Train Codec](#4-train-codec)
5. [Train S2S Transformer](#5-train-s2s-transformer)
6. [Run Demo & Testing](#6-run-demo--testing)
7. [Create Backups](#7-create-backups)
8. [Download Files](#8-download-files)

---

## 1. Initial Setup

```bash
# Connect to RunPod terminal
# Pod Type: GPU Pod (RTX 4090 or A100 recommended)
# Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Navigate to workspace
cd /workspace

# Clone or upload your project
mkdir -p NewProject
cd NewProject
```

---

## 2. Install Dependencies

### System Packages
```bash
apt-get update
apt-get install -y ffmpeg libsndfile1 sox
```

### Python Packages
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy librosa soundfile
pip install einops rotary-embedding-torch
pip install transformers accelerate
pip install fastapi uvicorn python-multipart websockets
pip install tensorboard tqdm pyyaml
pip install datasets huggingface_hub
```

### Optional (for advanced features)
```bash
# Flash Attention (optional, for faster training)
pip install flash-attn --no-build-isolation

# Audio processing
pip install pyloudnorm audioread
```

---

## 3. Download Datasets

### Create Data Directory
```bash
mkdir -p /workspace/telugu_data
cd /workspace/telugu_data
```

### Download OpenSLR Telugu Dataset
```bash
# OpenSLR 66 - Telugu Multi-speaker
wget https://www.openslr.org/resources/66/te_in_female.zip
unzip te_in_female.zip -d openslr/

# Or use Hugging Face datasets
python -c "
from datasets import load_dataset
ds = load_dataset('mozilla-foundation/common_voice_11_0', 'te', split='train[:1000]')
ds.save_to_disk('/workspace/telugu_data/common_voice_te')
"
```

### Prepare Audio Files
```bash
# Convert to 16kHz mono WAV
mkdir -p /workspace/telugu_data/processed
for f in /workspace/telugu_data/openslr/*.wav; do
    ffmpeg -i "$f" -ar 16000 -ac 1 "/workspace/telugu_data/processed/$(basename $f)" -y
done
```

---

## 4. Train Codec

### Create Model Directory
```bash
mkdir -p /workspace/models/codec
mkdir -p /workspace/logs
```

### Start Codec Training
```bash
cd /workspace/NewProject

nohup python train_codec_dac.py \
    --data_dir /workspace/telugu_data \
    --checkpoint_dir /workspace/models/codec \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --experiment_name "telugu_codec_v1" \
    > /workspace/logs/codec_training.log 2>&1 &

# Monitor training
tail -f /workspace/logs/codec_training.log
```

### Check Training Progress
```bash
# View latest logs
tail -50 /workspace/logs/codec_training.log

# Check if model saved
ls -la /workspace/models/codec/
```

---

## 5. Train S2S Transformer

### Create S2S Model Directory
```bash
mkdir -p /workspace/models/s2s
```

### Start S2S Training
```bash
cd /workspace/NewProject

nohup python train_s2s.py \
    --data_dir /workspace/telugu_data \
    --codec_path /workspace/models/codec/best_codec.pt \
    --checkpoint_dir /workspace/models/s2s \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --experiment_name "telugu_s2s_poc_v1" \
    > /workspace/logs/s2s_training.log 2>&1 &

# Monitor training
tail -f /workspace/logs/s2s_training.log
```

---

## 6. Run Demo & Testing

### Test Codec Quality
```bash
cd /workspace/NewProject

python demo_complete_s2s.py \
    --codec_path /workspace/models/codec/best_codec.pt \
    --s2s_path /workspace/models/s2s/s2s_best.pt
```

### Run Real-time Streaming Server
```bash
cd /workspace/NewProject

python realtime_codec_server.py \
    --codec_path /workspace/models/codec/best_codec.pt \
    --host 0.0.0.0 \
    --port 8010
```

### Access in Browser
```
https://[YOUR-POD-ID]-8010.proxy.runpod.net/
```

---

## 7. Create Backups

### Backup Codec Only
```bash
cd /workspace
mkdir -p backup

cp /workspace/models/codec/best_codec.pt backup/
cp /workspace/NewProject/telugu_codec_fixed.py backup/
cp /workspace/NewProject/demo_voice_poc.py backup/
cp /workspace/models/speaker_embeddings.json backup/

tar -czvf telugu_poc_backup.tar.gz backup/
```

### Backup Complete System (Codec + S2S)
```bash
cd /workspace
mkdir -p backup

# Copy all essential files
cp /workspace/NewProject/train_s2s.py backup/
cp /workspace/NewProject/s2s_transformer.py backup/
cp /workspace/models/s2s/s2s_best.pt backup/
cp /workspace/NewProject/telugu_codec_fixed.py backup/
cp /workspace/NewProject/demo_voice_poc.py backup/
cp /workspace/models/speaker_embeddings.json backup/
cp /workspace/models/codec/best_codec.pt backup/

tar -czvf telugu_s2s_complete.tar.gz backup/
```

---

## 8. Download Files

### Method 1: HTTP Server
```bash
cd /workspace
python -m http.server 8010 &

# Access files at:
# https://[YOUR-POD-ID]-8010.proxy.runpod.net/telugu_s2s_complete.tar.gz
```

### Method 2: Hugging Face Hub
```bash
pip install huggingface_hub

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='/workspace/telugu_s2s_complete.tar.gz',
    path_in_repo='telugu_s2s_complete.tar.gz',
    repo_id='YOUR_USERNAME/telugu-voice-ai',
    repo_type='model'
)
"
```

---

## 📊 Quick Reference

| Task | Command |
|------|---------|
| Check GPU | `nvidia-smi` |
| Monitor training | `tail -f /workspace/logs/*.log` |
| List models | `ls -la /workspace/models/*/` |
| Check disk space | `df -h` |
| Kill background job | `pkill -f train_` |
| Check running processes | `ps aux \| grep python` |

---

## ⚠️ Important Notes

1. **Always backup before terminating pod**
2. **Download tar files to local system**
3. **Verify tar file integrity after download**
4. **Use headphones when testing real-time streaming**

---

*Generated: November 26, 2025*
