# Telugu S2S - Complete RunPod Deployment Manual
## Step-by-Step Commands for Web Terminal

---

## 📋 Pre-Deployment: File Cleanup

### Files to Keep (New Architecture)
```
✅ Core Models:
   - telugu_codec.py
   - s2s_transformer.py
   - streaming_server.py

✅ Training Scripts:
   - train_codec.py
   - train_s2s.py
   - data_collection.py

✅ Configuration:
   - config.py
   - data_sources.yaml
   - runpod_config.yaml
   - requirements_new.txt
   - runpod_deploy.sh

✅ Documentation:
   - README.md
   - ARCHITECTURE_DESIGN.md
   - EXECUTIVE_SUMMARY.md
   - TELUGU_S2S_RESEARCH_PLAN.md

✅ Assets:
   - .gitignore
   - static/ (folder)
```

### Files to Delete (Old Pipeline)
```
❌ Old Code:
   - s2s_pipeline.py
   - server.py
   - download_models.py
   - download_telugu.py
   - test_latency.py
   - train_telugu.py
   - train_telugu.sh
   - startup.sh
   - fix_and_run.sh
   - cleanup_old.py
   - requirements.txt

❌ Old Documentation:
   - FINAL_FIXES.txt
   - GPU_RECOMMENDATION.md
   - INSTALLATION_GUIDE.md
   - ISSUE_FIXED.md
   - PERFORMANCE_OPTIMIZATION.md
   - PROJECT_SUMMARY.md
   - QUICK_START.md
   - RUNPOD_FIX_COMMANDS.txt
   - RUNPOD_QUICK_FIX.md
   - UPDATE_GUIDE.md
   - telugu-s2s-windsurf.md
   - telugu_videos.txt
```

---

## 🚀 PHASE 1: Initial Setup (Local Machine)

### Step 1.1: Clean Repository
```bash
# Navigate to project
cd d:\NewProject

# Delete old files (run in PowerShell or Git Bash)
Remove-Item s2s_pipeline.py, server.py, download_models.py, download_telugu.py, test_latency.py, train_telugu.py, train_telugu.sh, startup.sh, fix_and_run.sh, cleanup_old.py, requirements.txt -Force

Remove-Item FINAL_FIXES.txt, GPU_RECOMMENDATION.md, INSTALLATION_GUIDE.md, ISSUE_FIXED.md, PERFORMANCE_OPTIMIZATION.md, PROJECT_SUMMARY.md, QUICK_START.md, RUNPOD_FIX_COMMANDS.txt, RUNPOD_QUICK_FIX.md, UPDATE_GUIDE.md, telugu-s2s-windsurf.md, telugu_videos.txt -Force
```

### Step 1.2: Push Clean Code to GitHub
```bash
# Stage all changes
git add .

# Commit
git commit -m "New Telugu S2S architecture with <150ms latency"

# Push to GitHub
git push origin main
```

---

## 🖥️ PHASE 2: RunPod Setup

### Step 2.1: Create RunPod Account
1. Go to: https://www.runpod.io/
2. Sign up / Log in
3. Add payment method
4. Note your API key

### Step 2.2: Choose GPU Configuration

#### For Training (Codec + S2S Model)
```
Pod Configuration:
├─ GPU: H200 SXM (141GB VRAM)
├─ Template: RunPod PyTorch 2.2
├─ Container: runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04
├─ Disk: 200GB
├─ Expose Ports: 22, 6006, 8888
└─ Cost: $3.89/hour
```

#### For Inference (Production)
```
Pod Configuration:
├─ GPU: RTX A6000 (48GB VRAM)
├─ Template: RunPod Fast Stable Diffusion
├─ Container: runpod/pytorch:2.2.0-py3.10-cuda11.8.0-runtime-ubuntu22.04
├─ Disk: 50GB
├─ Expose Ports: 8000, 8001
└─ Cost: $0.49/hour
```

---

## 🎯 PHASE 3: Training Pipeline (H200 Pod)

### Step 3.1: SSH into H200 Pod
```bash
# Get SSH command from RunPod dashboard
# Example:
ssh root@<POD_ID>-<PORT>.proxy.runpod.net -p <PORT>

# Verify GPU
nvidia-smi
```

### Step 3.2: System Setup
```bash
# Update system
apt-get update && apt-get upgrade -y

# Install system dependencies
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
    screen

echo "✓ System dependencies installed"
```

### Step 3.3: Clone Repository
```bash
# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/devasphn/NewProject.git telugu-s2s

# Enter directory
cd telugu-s2s

# Verify files
ls -la

echo "✓ Repository cloned"
```

### Step 3.4: Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements_new.txt

# Install Flash Attention (critical for speed)
pip install flash-attn --no-build-isolation

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print('Flash Attention: OK')"

echo "✓ Python dependencies installed"
```

### Step 3.5: Setup Environment Variables
```bash
# Create .env file
cat > .env << EOF
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_key_here
CUDA_VISIBLE_DEVICES=0
EOF

# Load environment
export $(cat .env | xargs)

echo "✓ Environment configured"
```

### Step 3.6: Collect Telugu Data
```bash
# Create data directory
mkdir -p /workspace/telugu_data

# Install yt-dlp for YouTube downloads
pip install yt-dlp

# Start data collection (in screen session)
screen -S data_collection

# Inside screen:
python data_collection.py \
    --data_dir /workspace/telugu_data \
    --config data_sources.yaml \
    --max_hours 100

# Detach: Ctrl+A, then D
# Reattach: screen -r data_collection

# Check progress
watch -n 60 du -sh /workspace/telugu_data

echo "✓ Data collection started (1-2 hours)"
```

### Step 3.7: Train Codec (6-8 hours)
```bash
# Wait for data collection to complete
# Then start codec training

# Create checkpoint directory
mkdir -p /workspace/models

# Start training in screen
screen -S codec_training

# Inside screen:
python train_codec.py \
    --data_dir /workspace/telugu_data \
    --checkpoint_dir /workspace/models \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --experiment_name telucodec_production

# Detach: Ctrl+A, then D

# Monitor with TensorBoard
tensorboard --logdir /workspace/models/logs --host 0.0.0.0 --port 6006

# Or monitor with wandb (check wandb.ai)

echo "✓ Codec training started (6-8 hours)"
echo "Monitor: http://<POD_ID>.proxy.runpod.net:6006"
```

### Step 3.8: Train S2S Model (18-24 hours)
```bash
# Wait for codec training to complete
# Check: ls -lh /workspace/models/best_codec.pt

# Start S2S training
screen -S s2s_training

# Inside screen:
python train_s2s.py \
    --data_dir /workspace/telugu_data \
    --codec_path /workspace/models/best_codec.pt \
    --checkpoint_dir /workspace/models \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 5e-5 \
    --experiment_name telugu_s2s_production

# Detach: Ctrl+A, then D

echo "✓ S2S training started (18-24 hours)"
```

### Step 3.9: Verify Training Complete
```bash
# Check models exist
ls -lh /workspace/models/

# Expected files:
# - best_codec.pt (~500MB)
# - s2s_best.pt (~1.2GB)
# - codec_epoch_*.pt (checkpoints)
# - s2s_epoch_*.pt (checkpoints)

# Test codec
python -c "
import torch
from telugu_codec import TeluCodec

codec = TeluCodec()
checkpoint = torch.load('/workspace/models/best_codec.pt')
codec.load_state_dict(checkpoint['model_state'])
print('✓ Codec loads successfully')
print(f'Bitrate: {codec.calculate_bitrate():.2f} kbps')
"

# Test S2S model
python -c "
import torch
from s2s_transformer import TeluguS2STransformer, S2SConfig

config = S2SConfig()
model = TeluguS2STransformer(config)
checkpoint = torch.load('/workspace/models/s2s_best.pt')
model.load_state_dict(checkpoint['model_state'])
print('✓ S2S model loads successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"

echo "✓ Training complete and verified"
```

### Step 3.10: Upload Models to HuggingFace
```bash
# Install HuggingFace Hub
pip install huggingface_hub

# Upload models
python -c "
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create repositories
create_repo('devasphn/telucodec', repo_type='model', exist_ok=True)
create_repo('devasphn/telugu-s2s', repo_type='model', exist_ok=True)

# Upload codec
api.upload_file(
    path_or_fileobj='/workspace/models/best_codec.pt',
    path_in_repo='best_codec.pt',
    repo_id='devasphn/telucodec',
    repo_type='model'
)

# Upload S2S model
api.upload_file(
    path_or_fileobj='/workspace/models/s2s_best.pt',
    path_in_repo='s2s_best.pt',
    repo_id='devasphn/telugu-s2s',
    repo_type='model'
)

print('✓ Models uploaded to HuggingFace')
"

echo "✓ Models backed up to HuggingFace"
```

---

## 🌐 PHASE 4: Production Deployment (A6000 Pod)

### Step 4.1: Create A6000 Pod
```
1. Go to RunPod dashboard
2. Click "New Pod"
3. Select: RTX A6000 (48GB)
4. Template: PyTorch 2.2 Runtime
5. Disk: 50GB
6. Expose ports: 8000, 8001
7. Deploy
```

### Step 4.2: SSH into A6000 Pod
```bash
ssh root@<POD_ID>-<PORT>.proxy.runpod.net -p <PORT>

# Verify GPU
nvidia-smi
```

### Step 4.3: Setup Inference Environment
```bash
# System dependencies
apt-get update && apt-get install -y ffmpeg git vim

# Navigate to workspace
cd /workspace

# Clone repository
git clone https://github.com/devasphn/NewProject.git telugu-s2s
cd telugu-s2s

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_new.txt
pip install flash-attn --no-build-isolation

echo "✓ Inference environment ready"
```

### Step 4.4: Download Trained Models
```bash
# Create models directory
mkdir -p /workspace/models

# Download from HuggingFace
python -c "
from huggingface_hub import hf_hub_download
import os

os.environ['HF_TOKEN'] = 'your_token_here'

# Download codec
codec_path = hf_hub_download(
    repo_id='devasphn/telucodec',
    filename='best_codec.pt',
    local_dir='/workspace/models'
)

# Download S2S model
s2s_path = hf_hub_download(
    repo_id='devasphn/telugu-s2s',
    filename='s2s_best.pt',
    local_dir='/workspace/models'
)

print('✓ Models downloaded')
"

# Verify
ls -lh /workspace/models/
```

### Step 4.5: Test Models Locally
```bash
# Quick test
python -c "
import torch
from telugu_codec import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig

# Load codec
codec = TeluCodec()
codec.load_state_dict(torch.load('/workspace/models/best_codec.pt')['model_state'])
codec.eval()

# Load S2S
config = S2SConfig()
s2s = TeluguS2STransformer(config)
s2s.load_state_dict(torch.load('/workspace/models/s2s_best.pt')['model_state'])
s2s.eval()

# Test inference
dummy_audio = torch.randn(1, 1, 16000)
codes = codec.encode(dummy_audio)
print(f'✓ Codec encoding works: {codes.shape}')

# Test S2S
import time
start = time.time()
encoder_out = s2s.encode(codes, torch.tensor([0]), torch.tensor([0]))
latency = (time.time() - start) * 1000
print(f'✓ S2S encoding latency: {latency:.1f}ms')

if latency < 50:
    print('✅ LATENCY TARGET MET!')
else:
    print('⚠️  Latency above 50ms, optimize further')
"
```

### Step 4.6: Start Production Server
```bash
# Start server in screen
screen -S telugu_s2s_server

# Inside screen:
python streaming_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model_dir /workspace/models \
    --device cuda

# Detach: Ctrl+A, then D

# Check server is running
curl http://localhost:8000/stats

echo "✓ Server started on port 8000"
```

### Step 4.7: Access the System
```bash
# Get your pod URL from RunPod dashboard
# Example: https://<POD_ID>-8000.proxy.runpod.net

# Test WebSocket connection
curl http://<POD_ID>-8000.proxy.runpod.net/

# Open in browser for full UI:
# http://<POD_ID>-8000.proxy.runpod.net
```

---

## ✅ PHASE 5: Verification & Testing

### Step 5.1: Latency Benchmark
```bash
python -c "
import torch
import time
from telugu_codec import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig

# Load models
codec = TeluCodec().cuda()
codec.load_state_dict(torch.load('/workspace/models/best_codec.pt')['model_state'])
codec.eval()

config = S2SConfig()
s2s = TeluguS2STransformer(config).cuda()
s2s.load_state_dict(torch.load('/workspace/models/s2s_best.pt')['model_state'])
s2s.eval()

# Benchmark
latencies = []
for _ in range(10):
    audio = torch.randn(1, 1, 1600).cuda()  # 100ms
    
    torch.cuda.synchronize()
    start = time.time()
    
    # Encode
    codes = codec.encode(audio)
    
    # S2S
    enc_out = s2s.encode(codes, torch.tensor([0]).cuda(), torch.tensor([0]).cuda())
    
    # Generate first token
    for i, chunk in enumerate(s2s.generate_streaming(codes, torch.tensor([0]).cuda(), torch.tensor([0]).cuda())):
        break
    
    # Decode
    response = codec.decode(chunk)
    
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000
    latencies.append(latency)

import numpy as np
print(f'Latency stats (10 runs):')
print(f'  Mean: {np.mean(latencies):.1f}ms')
print(f'  Min: {np.min(latencies):.1f}ms')
print(f'  Max: {np.max(latencies):.1f}ms')
print(f'  Std: {np.std(latencies):.1f}ms')

if np.mean(latencies) < 150:
    print('✅ TARGET LATENCY ACHIEVED (<150ms)')
else:
    print('⚠️  Latency above target')
"
```

### Step 5.2: Quality Test
```bash
# Test emotional speech
python -c "
from s2s_transformer import EMOTION_IDS, SPEAKER_IDS

print('Available emotions:')
for emotion, id in EMOTION_IDS.items():
    print(f'  {id}: {emotion}')

print('\nAvailable speakers:')
for speaker, id in SPEAKER_IDS.items():
    print(f'  {id}: {speaker}')

print('\n✓ All emotions and speakers configured')
"
```

### Step 5.3: Monitor Performance
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Server stats
watch -n 5 curl -s http://localhost:8000/stats

# System resources
htop
```

---

## 📊 PHASE 6: Cost Tracking

### Training Cost (One-time)
```
H200 Training:
├─ Data Collection: 2 hours × $3.89 = $7.78
├─ Codec Training: 8 hours × $3.89 = $31.12
└─ S2S Training: 24 hours × $3.89 = $93.36
───────────────────────────────────────────
Total Training Cost: $132.26
```

### Inference Cost (Ongoing)
```
RTX A6000 Inference:
├─ Per Hour: $0.49
├─ Per Day: $11.76
├─ Per Month: $352.80
└─ Per User-Hour: $0.0049 (100 users/GPU)
```

---

## 🎯 PHASE 7: Performance Targets

### Verify These Metrics:
```bash
# 1. Latency
✓ First audio chunk: <150ms
✓ Streaming factor: <0.8
✓ Per-token generation: <5ms

# 2. Quality
✓ MOS score: >4.0/5
✓ Telugu accuracy: >90%
✓ Emotion recognition: >85%

# 3. Scalability
✓ Concurrent users: >100 per GPU
✓ Throughput: >1000 requests/hour
```

---

## 🔧 PHASE 8: Troubleshooting

### Issue 1: Import Errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements_new.txt --force-reinstall
```

### Issue 2: CUDA Out of Memory
```bash
# Solution: Reduce batch size
# Edit train_codec.py or train_s2s.py
# Change batch_size from 32 to 16 (codec)
# Change batch_size from 8 to 4 (s2s)
```

### Issue 3: Slow Download Speeds
```bash
# Solution: Use aria2c for parallel downloads
apt-get install aria2
# Modify data_collection.py to use aria2c
```

### Issue 4: Model Loading Fails
```bash
# Solution: Verify checkpoint integrity
python -c "
import torch
checkpoint = torch.load('/workspace/models/best_codec.pt')
print(checkpoint.keys())
"
```

---

## 📝 Next Steps After Deployment

### Immediate (Week 1)
- [ ] Deploy on A6000 pod
- [ ] Test with 10 beta users
- [ ] Collect feedback
- [ ] Monitor latency metrics

### Short-term (Month 1)
- [ ] Scale to 100 users
- [ ] Add monitoring dashboard
- [ ] Implement auto-scaling
- [ ] Create API documentation

### Medium-term (Month 3)
- [ ] Voice cloning feature
- [ ] Multi-lingual support
- [ ] Mobile SDK
- [ ] Enterprise features

---

## 🎊 Success Checklist

```
Training Phase:
✓ Data collected (100+ hours)
✓ Codec trained (6-8 hours)
✓ S2S model trained (18-24 hours)
✓ Models uploaded to HuggingFace
✓ Total cost <$150

Deployment Phase:
✓ A6000 pod running
✓ Models downloaded
✓ Server started
✓ Latency <150ms verified
✓ Emotions working (including laughter)
✓ All 4 speakers available

Production Ready:
✓ Accessible via public URL
✓ WebSocket working
✓ Demo UI responsive
✓ Stats endpoint active
✓ Monitoring in place
```

---

## 📞 Support

If you encounter issues:
1. Check screen sessions: `screen -ls`
2. View logs: `journalctl -u telugu-s2s -f`
3. Monitor GPU: `nvidia-smi`
4. Check disk space: `df -h`

---

**You are now ready to deploy the world's fastest Telugu S2S system!** 🚀

**Total Time:** 
- Training: ~36 hours
- Deployment: ~2 hours
- **Total: ~38 hours from zero to production**