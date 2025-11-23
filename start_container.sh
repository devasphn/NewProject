#!/bin/bash
# Telugu S2S Production Setup - RunPod Start Container Command
# Copy-paste this ENTIRE script into RunPod Template "Start Script" field

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Telugu S2S Production Setup Starting..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg sox libsox-fmt-all curl wget git-lfs > /dev/null 2>&1

# Install Node.js 20 (LTS) for web UI
echo "📦 Installing Node.js 20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
apt-get install -y -qq nodejs > /dev/null 2>&1

# Navigate to workspace
cd /workspace

# Clone repository
echo "📥 Cloning repository..."
if [ ! -d "NewProject" ]; then
    git clone https://github.com/devasphn/NewProject.git
fi

cd NewProject

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements_new.txt > /dev/null 2>&1

# Install Flash Attention (critical for speed)
echo "⚡ Installing Flash Attention (this may take 5-10 minutes)..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install audio processing tools
echo "🎵 Installing audio tools..."
pip install yt-dlp ffmpeg-python pydub soundfile librosa > /dev/null 2>&1

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p /workspace/telugu_data/{raw,processed,train,val,test}
mkdir -p /workspace/models/{codec,s2s,speaker}
mkdir -p /workspace/checkpoints/{codec,s2s,speaker}
mkdir -p /workspace/logs

# Verify installations
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation Verification:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -c "import torch; print(f'PyTorch:       {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "CUDA Device:    N/A"
python -c "from flash_attn import flash_attn_func; print('Flash Attn:     ✓')" 2>/dev/null || echo "Flash Attn:     ✗ FAILED"
python -c "import torchaudio; print(f'TorchAudio:     {torchaudio.__version__}')"
python -c "import whisperx; print('WhisperX:       ✓')" 2>/dev/null || echo "WhisperX:       ✗ (optional)"
echo "Node.js:        $(node --version)"
echo "npm:            $(npm --version)"
echo "ffmpeg:         $(ffmpeg -version | head -n1 | awk '{print $3}')"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Project Location: /workspace/NewProject"
echo "📁 Data Directory:   /workspace/telugu_data"
echo "📁 Models Directory: /workspace/models"
echo ""
echo "🔑 Next Steps:"
echo "   1. huggingface-cli login"
echo "   2. wandb login"
echo "   3. python download_telugu_data_PRODUCTION.py"
echo ""
echo "⏱️  Total setup time: ~15-20 minutes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
