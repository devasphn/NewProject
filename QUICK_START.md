# ⚡ QUICK START - For devasphn

## RunPod Web Terminal Commands

**Copy-paste these commands ONE BY ONE:**

### Step 1: Install & Clone
```bash
apt-get update && apt-get install -y git ffmpeg
```

### Step 2: Get Code
```bash
cd /workspace
git clone https://github.com/devasphn/NewProject.git
cd NewProject
```

### Step 3: Setup
```bash
chmod +x startup.sh train_telugu.sh
export HF_TOKEN='YOUR_TOKEN_HERE'
bash startup.sh
```

**Get token from**: https://huggingface.co/settings/tokens

### Step 4: Start Server
```bash
python server.py
```

### Step 5: Access Demo
- RunPod Dashboard → Your Pod → Connect
- Click "HTTP Service [Port 8000]"
- Demo opens in browser!

---

## Expected Timeline

- **Setup**: 20-30 minutes
- **Models Download**: 15-20 minutes  
- **Testing**: 5 minutes
- **Total**: ~45-60 minutes

---

## After Baseline Testing

### Train Telugu Model
```bash
# 1. Stop server (Ctrl+C)
# 2. Edit download_telugu.py (add YouTube URLs)
# 3. Run training
bash train_telugu.sh
```

---

## Troubleshooting

### "HF_TOKEN not found"
```bash
export HF_TOKEN='your_actual_token'
python download_models.py
```

### "Permission denied"
```bash
chmod +x startup.sh train_telugu.sh
```

### Check GPU
```bash
nvidia-smi
```

---

**That's it! Simple and straightforward!** 🚀
