#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  RUNPOD COMPLETE SETUP SCRIPT
#  Run this ONCE when you start a new RunPod instance
#  Tested on: RunPod H200 (141GB VRAM), Ubuntu 22.04
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  RUNPOD SETUP - Luna-Equivalent S2S AI"
echo "  This will install all dependencies and prepare the environment"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: System Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 1: Installing System Dependencies                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

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
    libsndfile1-dev \
    sox \
    libsox-dev \
    screen \
    nano \
    unzip \
    p7zip-full \
    aria2 \
    pigz

echo "✅ System dependencies installed"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Python Environment
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 2: Setting Up Python Environment                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1
echo "📦 Installing PyTorch with CUDA 12.1..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install main requirements
echo "📦 Installing project requirements..."
pip install \
    einops>=0.7.0 \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    safetensors>=0.4.0 \
    datasets>=2.14.0 \
    huggingface_hub>=0.19.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    pydub>=0.25.0 \
    tqdm>=4.66.0 \
    tensorboard>=2.15.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    websockets>=12.0 \
    pyyaml>=6.0.1 \
    rich>=13.0.0 \
    colorama>=0.4.6

# Install Flash Attention (optional but recommended)
echo "📦 Installing Flash Attention..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention install failed (optional)"

echo "✅ Python environment ready"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Clone Repository
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 3: Cloning Project Repository                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

cd /workspace

if [ ! -d "NewProject" ]; then
    git clone https://github.com/devasphn/NewProject.git
    echo "✅ Repository cloned"
else
    cd NewProject
    git pull
    echo "✅ Repository updated"
fi

cd /workspace/NewProject

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Create Data Directories
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 4: Creating Data Directory Structure                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

mkdir -p /workspace/data/{english,hindi,telugu}/{librispeech,indicvoices,kathbath,commonvoice,openslr,augmented}
mkdir -p /workspace/checkpoints_codec
mkdir -p /workspace/checkpoints_s2s
mkdir -p /workspace/logs

echo "✅ Directory structure created"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Verify Installation
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 5: Verifying Installation                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import torchaudio
print(f"TorchAudio: {torchaudio.__version__}")

from transformers import WavLMModel
print("✅ WavLM (for semantic distillation) available")

print("\n✅ All dependencies verified!")
EOF

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ✅ SETUP COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Login to HuggingFace: huggingface-cli login"
echo "  2. Download data: bash download_6000h_data.sh"
echo "  3. Download HF data: python download_huggingface_data.py"
echo "  4. Start training: python train_codec_production.py --help"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
