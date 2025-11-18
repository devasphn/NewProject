# Quick Command Reference
## Copy-Paste Commands for RunPod Terminal

---

## 🧹 CLEANUP (Local Machine First)

### PowerShell Commands:
```powershell
cd d:\NewProject

# Delete old files in one command
$oldFiles = @(
    "s2s_pipeline.py", "server.py", "download_models.py", 
    "download_telugu.py", "test_latency.py", "train_telugu.py", 
    "train_telugu.sh", "startup.sh", "fix_and_run.sh", 
    "cleanup_old.py", "requirements.txt", "FINAL_FIXES.txt", 
    "GPU_RECOMMENDATION.md", "INSTALLATION_GUIDE.md", 
    "ISSUE_FIXED.md", "PERFORMANCE_OPTIMIZATION.md", 
    "PROJECT_SUMMARY.md", "QUICK_START.md", 
    "RUNPOD_FIX_COMMANDS.txt", "RUNPOD_QUICK_FIX.md", 
    "UPDATE_GUIDE.md", "telugu-s2s-windsurf.md", "telugu_videos.txt"
)
$oldFiles | ForEach-Object { if (Test-Path $_) { Remove-Item $_ -Force } }

# Commit and push
git add .
git commit -m "Clean architecture: Telugu S2S with <150ms latency"
git push origin main
```

---

## 🚀 H200 TRAINING POD

### Initial Setup:
```bash
# System setup
apt-get update && apt-get install -y ffmpeg git vim tmux htop nvtop wget curl build-essential libsndfile1 sox screen

# Clone repo
cd /workspace && git clone https://github.com/devasphn/NewProject.git telugu-s2s && cd telugu-s2s

# Install Python
pip install --upgrade pip && pip install -r requirements_new.txt && pip install flash-attn --no-build-isolation

# Environment
cat > .env << 'EOF'
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
CUDA_VISIBLE_DEVICES=0
EOF
export $(cat .env | xargs)
```

### Data Collection:
```bash
mkdir -p /workspace/telugu_data
screen -S data_collection
python data_collection.py --data_dir /workspace/telugu_data --config data_sources.yaml --max_hours 100
# Ctrl+A, D to detach
```

### Train Codec:
```bash
mkdir -p /workspace/models
screen -S codec_training
python train_codec.py --data_dir /workspace/telugu_data --checkpoint_dir /workspace/models --batch_size 32 --num_epochs 100 --learning_rate 1e-4 --experiment_name telucodec_production
# Ctrl+A, D to detach
```

### Train S2S:
```bash
screen -S s2s_training
python train_s2s.py --data_dir /workspace/telugu_data --codec_path /workspace/models/best_codec.pt --checkpoint_dir /workspace/models --batch_size 8 --num_epochs 200 --learning_rate 5e-5 --experiment_name telugu_s2s_production
# Ctrl+A, D to detach
```

### Upload Models:
```bash
pip install huggingface_hub
python << 'EOF'
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ['HF_TOKEN'])
create_repo('devasphn/telucodec', repo_type='model', exist_ok=True)
create_repo('devasphn/telugu-s2s', repo_type='model', exist_ok=True)

api.upload_file(path_or_fileobj='/workspace/models/best_codec.pt', path_in_repo='best_codec.pt', repo_id='devasphn/telucodec')
api.upload_file(path_or_fileobj='/workspace/models/s2s_best.pt', path_in_repo='s2s_best.pt', repo_id='devasphn/telugu-s2s')
print('✓ Models uploaded')
EOF
```

---

## 🌐 A6000 INFERENCE POD

### Setup:
```bash
# System
apt-get update && apt-get install -y ffmpeg git vim

# Clone
cd /workspace && git clone https://github.com/devasphn/NewProject.git telugu-s2s && cd telugu-s2s

# Install
pip install --upgrade pip && pip install -r requirements_new.txt && pip install flash-attn --no-build-isolation
```

### Download Models:
```bash
mkdir -p /workspace/models
python << 'EOF'
from huggingface_hub import hf_hub_download
import os

os.environ['HF_TOKEN'] = 'hf_your_token_here'

hf_hub_download(repo_id='devasphn/telucodec', filename='best_codec.pt', local_dir='/workspace/models')
hf_hub_download(repo_id='devasphn/telugu-s2s', filename='s2s_best.pt', local_dir='/workspace/models')
print('✓ Models ready')
EOF
```

### Start Server:
```bash
screen -S telugu_s2s_server
python streaming_server.py --host 0.0.0.0 --port 8000 --model_dir /workspace/models --device cuda
# Ctrl+A, D to detach
```

---

## 📊 MONITORING

### Check Training:
```bash
# View screens
screen -ls

# Attach to training
screen -r codec_training
screen -r s2s_training

# GPU usage
watch -n 1 nvidia-smi

# Disk space
df -h /workspace
```

### Check Server:
```bash
# Stats
curl http://localhost:8000/stats

# Logs
screen -r telugu_s2s_server
```

---

## ⚡ BENCHMARK

### Test Latency:
```bash
python << 'EOF'
import torch, time, numpy as np
from telugu_codec import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig

codec = TeluCodec().cuda()
codec.load_state_dict(torch.load('/workspace/models/best_codec.pt')['model_state'])
codec.eval()

config = S2SConfig()
s2s = TeluguS2STransformer(config).cuda()
s2s.load_state_dict(torch.load('/workspace/models/s2s_best.pt')['model_state'])
s2s.eval()

latencies = []
for _ in range(10):
    audio = torch.randn(1, 1, 1600).cuda()
    torch.cuda.synchronize()
    start = time.time()
    codes = codec.encode(audio)
    enc_out = s2s.encode(codes, torch.tensor([0]).cuda(), torch.tensor([0]).cuda())
    for chunk in s2s.generate_streaming(codes, torch.tensor([0]).cuda(), torch.tensor([0]).cuda()):
        break
    response = codec.decode(chunk)
    torch.cuda.synchronize()
    latencies.append((time.time()-start)*1000)

print(f'Mean: {np.mean(latencies):.1f}ms')
print(f'Min: {np.min(latencies):.1f}ms')
print('✅ PASS' if np.mean(latencies)<150 else '❌ FAIL')
EOF
```

---

## 🔧 TROUBLESHOOTING

### Restart Server:
```bash
screen -X -S telugu_s2s_server quit
screen -S telugu_s2s_server
python streaming_server.py --host 0.0.0.0 --port 8000 --model_dir /workspace/models --device cuda
```

### Check Models:
```bash
ls -lh /workspace/models/
python -c "import torch; print(torch.load('/workspace/models/best_codec.pt').keys())"
```

### Free Memory:
```bash
# Kill screens
screen -X -S data_collection quit
screen -X -S codec_training quit

# Clear cache
sync && echo 3 > /proc/sys/vm/drop_caches
```

---

## 📍 Access URLs

```
Training Pod:
- TensorBoard: http://<POD_ID>-6006.proxy.runpod.net
- Jupyter: http://<POD_ID>-8888.proxy.runpod.net

Inference Pod:
- Demo UI: http://<POD_ID>-8000.proxy.runpod.net
- WebSocket: ws://<POD_ID>-8000.proxy.runpod.net/ws
- Stats API: http://<POD_ID>-8000.proxy.runpod.net/stats
```

---

## ✅ Final Checklist

```
Local Machine:
□ Old files deleted
□ Code pushed to GitHub

H200 Training:
□ Data collected (100+ hours)
□ Codec trained (best_codec.pt exists)
□ S2S trained (s2s_best.pt exists)
□ Models uploaded to HuggingFace
□ Cost: ~$130

A6000 Inference:
□ Pod created
□ Models downloaded
□ Server running
□ Latency verified <150ms
□ Demo UI accessible

Production:
□ Emotions working (9 types)
□ Speakers working (4 voices)
□ WebSocket stable
□ Monitoring active
```

---

**⏱️ Total Time: 36 hours training + 2 hours deployment = 38 hours**
**💰 Total Cost: ~$130 training + $0.49/hr inference**

🎉 **You're ready to beat Luna Demo!**