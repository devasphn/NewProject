# ⚡ QUICK START

## RunPod Commands (Copy-Paste)

### 1. Install Dependencies
```bash
apt-get update
apt-get install -y git ffmpeg
```

### 2. Clone Repository
```bash
cd /workspace
git clone https://github.com/devasphn/NewProject.git
cd NewProject
```

### 3. Make Scripts Executable
```bash
chmod +x startup.sh train_telugu.sh
```

### 4. Set HuggingFace Token
```bash
export HF_TOKEN='your_token_here'
```
Get token: https://huggingface.co/settings/tokens

### 5. Run Setup (20-25 min)
```bash
bash startup.sh
```

### 6. Start Server
```bash
python server.py
```

### 7. Access Demo
RunPod → Your Pod → HTTP Service [Port 8000]

---

## ⏱️ Timeline

- Install: 2 min
- Setup: 25 min
- **Total**: ~30 min

---

## 🔧 Common Commands

```bash
# Check GPU
nvidia-smi

# Test latency
python test_latency.py --mode baseline

# Train Telugu
bash train_telugu.sh

# View logs
tail -f logs/server.log
```

---

## ⚠️ Troubleshooting

**HF_TOKEN error:**
```bash
export HF_TOKEN='your_token'
```

**Permission denied:**
```bash
chmod +x startup.sh train_telugu.sh
```

**Port not accessible:**
- Check RunPod ports exposed
- Restart: `python server.py`

---

**Done! See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for details** 📚

