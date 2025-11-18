# 🚀 GitHub Setup & RunPod Deployment Guide

## Step 1: Create GitHub Repository (5 minutes)

### On GitHub.com:

1. Go to https://github.com/new
2. Repository name: `NewProject`
3. Description: "Telugu Speech-to-Speech Voice Agent - Ultra-Low Latency"
4. Set to: **Private** (recommended) or Public
5. **DO NOT** initialize with README (we have our own files)
6. Click "Create repository"

---

## Step 2: Push Code to GitHub (Local Machine)

```bash
# Navigate to your project directory
cd d:\NewProject

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Telugu S2S Voice Agent"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/NewProject.git

# Push to GitHub
git push -u origin main
```

**If you get branch name error:**
```bash
git branch -M main
git push -u origin main
```

---

## Step 3: Launch RunPod Instance

### On RunPod.io:

1. Go to https://www.runpod.io/console/pods
2. Click **"Deploy"** → **"GPU Cloud"**
3. Select Template: **PyTorch 2.1.0** (or any PyTorch template)
4. Select GPU: **RTX A6000** (48GB VRAM)
5. Configuration:
   - Container Disk: **100 GB**
   - Volume Disk: **150 GB** (for models + data)
   - Volume Mount Path: `/workspace`
6. Click **"Deploy On-Demand"**
7. Wait 2-3 minutes for pod to start
8. Click **"Connect"** → **"Start Web Terminal"**

**Cost**: $0.49/hour for RTX A6000

---

## Step 4: RunPod Commands (Copy-Paste in Web Terminal)

### ⚡ MANUAL SETUP (Step-by-Step)

**Run these commands ONE BY ONE in RunPod Web Terminal:**

#### 1. Install System Dependencies
```bash
apt-get update
apt-get install -y git ffmpeg
```

#### 2. Navigate to Workspace
```bash
cd /workspace
```

#### 3. Clone Your Repository
```bash
git clone https://github.com/devasphn/NewProject.git
```

#### 4. Enter Project Directory
```bash
cd NewProject
```

#### 5. Make Scripts Executable
```bash
chmod +x startup.sh train_telugu.sh
```

#### 6. Set HuggingFace Token (IMPORTANT!)
```bash
# Get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN='your_huggingface_token_here'
```

**Replace `your_huggingface_token_here` with your actual token!**

#### 7. Run Startup Script
```bash
bash startup.sh
```

**This will:**
- Install Python dependencies (5-10 min)
- Create directories
- Download all models (15-20 min)
- Test baseline latency
- Show you how to start server

---

### 📋 AFTER STARTUP SCRIPT COMPLETES

#### 8. Start Server
```bash
python server.py
```

**Server will start on port 8000**

---

### 🔍 If You Need to Verify GPU or Install More Packages

#### Check GPU
```bash
nvidia-smi
# Should show RTX A6000 with 48GB memory
```

#### Install Additional Packages
```bash
pip install package_name
```

#### Test Latency Manually
```bash
python test_latency.py --mode baseline
```

---

## Step 5: Access Your Demo

1. In RunPod dashboard, find your pod
2. Click **"Connect"** → **"HTTP Service [Port 8000]"**
3. Browser demo opens automatically
4. Click **"Start Conversation"**
5. Allow microphone access
6. **Speak in Telugu!**

---

## Step 6: Train Telugu Model (Optional)

### After baseline testing, train on Telugu data:

#### 1. Add Telugu Video URLs

Edit `download_telugu.py`:
```python
urls = [
    "https://www.youtube.com/watch?v=ACTUAL_VIDEO_ID_1",
    "https://www.youtube.com/watch?v=ACTUAL_VIDEO_ID_2",
    # Add 15-20 Telugu video URLs
]
```

See `telugu_videos.txt` for guidance on finding videos.

#### 2. Stop Server
```bash
# Press Ctrl+C in terminal
```

#### 3. Run Training
```bash
bash train_telugu.sh
```

This will:
- Download Telugu data (~2 hours)
- Train model (~3-4 hours on A6000)
- Test Telugu latency

#### 4. Restart Server with Telugu Model
```bash
python server.py
```

---

## 📊 Expected Results

### Baseline (Before Telugu Training):
- **Latency**: 250-350ms
- **Telugu Recognition**: 70-80% (using Whisper's multilingual)
- **Response Quality**: Good

### After Telugu Training:
- **Latency**: 300-400ms (slightly higher but within target)
- **Telugu Recognition**: 85-90%
- **Response Quality**: Excellent
- **Accent**: Natural Telugu

---

## 🎯 Demo Checklist for MD

Before presenting to MD, verify:

- [ ] Server starts without errors
- [ ] Browser demo loads
- [ ] Microphone works
- [ ] Telugu speech recognized
- [ ] Audio response plays
- [ ] **Latency <400ms** ✅
- [ ] Metrics display correctly
- [ ] No crashes during 3+ test runs

---

## 🔧 Troubleshooting

### "HF_TOKEN not found"
```bash
export HF_TOKEN='your_token_here'
python download_models.py
```

### "CUDA out of memory"
```bash
# Reduce batch size in config.py
TRAINING_BATCH_SIZE = 2  # Instead of 4
```

### "Port 8000 not accessible"
- Check RunPod dashboard → Ports → Ensure 8000 is exposed
- Try port 8080 instead (edit config.py)

### "WebSocket connection failed"
- Refresh browser
- Check server logs for errors
- Restart server

---

## 💰 Cost Tracking

### RTX A6000 Costs:
- Setup + Testing: 2-3 hours = **$1.50**
- Telugu Training: 4-5 hours = **$2.50**
- Demo Day: 2 hours = **$1.00**
- **Total**: ~**$5-6** for complete POC

### Storage:
- **$2/month** for 150GB volume

---

## 🎉 You're Ready!

Once setup completes, you'll have:
- ✅ Working Telugu S2S voice agent
- ✅ <400ms latency
- ✅ WebSocket streaming
- ✅ Browser demo
- ✅ Ready for MD presentation

**Server URL**: `http://[your-runpod-id]-8000.proxy.runpod.net`

---

## 📝 Quick Commands Reference

```bash
# Start server
python server.py

# Test latency
python test_latency.py --mode baseline
python test_latency.py --mode telugu

# Check GPU
nvidia-smi

# View logs
tail -f logs/server.log

# Train Telugu
bash train_telugu.sh
```

---

## 🚀 Ready to Deploy!

Follow these steps exactly and you'll have a working demo in 3-4 hours!
