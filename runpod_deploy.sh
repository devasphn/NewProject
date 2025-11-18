#!/bin/bash
# ============================================================================
# Telugu S2S System - Complete RunPod Deployment Script
# Ultra-low latency (<150ms) with emotional speech and laughter
# ============================================================================

set -e  # Exit on error

echo "============================================================"
echo "Telugu S2S Deployment Script"
echo "Target: <150ms latency with emotional speech"
echo "============================================================"

# Configuration
WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/telugu-s2s"
DATA_DIR="$WORKSPACE/telugu_data"
MODEL_DIR="$WORKSPACE/models"
LOG_DIR="$WORKSPACE/logs"

# Check GPU
echo "[1/10] Checking GPU..."
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "✓ GPU: $GPU_NAME"

# Setup directories
echo "[2/10] Setting up directories..."
mkdir -p $DATA_DIR $MODEL_DIR $LOG_DIR
cd $WORKSPACE

# Clone repository
echo "[3/10] Cloning repository..."
if [ ! -d "$PROJECT_DIR" ]; then
    git clone https://github.com/devasphn/telugu-s2s.git
fi
cd $PROJECT_DIR

# Install system dependencies
echo "[4/10] Installing system dependencies..."
apt-get update && apt-get install -y \
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
    sox

# Install Python packages
echo "[5/10] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements_new.txt

# Install Flash Attention (if on H200/A6000)
if [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
    echo "Installing Flash Attention for $GPU_NAME..."
    pip install flash-attn --no-build-isolation
fi

# Download Telugu data
echo "[6/10] Downloading Telugu data (this may take 1-2 hours)..."
python data_collection.py \
    --data_dir $DATA_DIR \
    --sources "raw_talks,news" \
    --max_hours 100

echo "Data collection complete. Total size:"
du -sh $DATA_DIR

# Train or download models based on GPU
if [[ "$GPU_NAME" == *"H200"* ]]; then
    echo "============================================================"
    echo "H200 detected - Starting training pipeline"
    echo "============================================================"
    
    # Phase 1: Train codec
    echo "[7/10] Training TeluCodec (6-8 hours)..."
    python train_codec.py \
        --data_dir $DATA_DIR \
        --checkpoint_dir $MODEL_DIR \
        --batch_size 32 \
        --num_epochs 100 \
        --experiment_name "telucodec_production"
    
    # Phase 2: Train S2S model
    echo "[8/10] Training S2S Transformer (18-24 hours)..."
    python train_s2s.py \
        --data_dir $DATA_DIR \
        --codec_path "$MODEL_DIR/best_codec.pt" \
        --checkpoint_dir $MODEL_DIR \
        --batch_size 8 \
        --num_epochs 200 \
        --experiment_name "telugu_s2s_production"
    
    echo "✓ Training complete!"
    
elif [[ "$GPU_NAME" == *"A6000"* ]] || [[ "$GPU_NAME" == *"4090"* ]]; then
    echo "============================================================"
    echo "Inference GPU detected - Downloading pre-trained models"
    echo "============================================================"
    
    # Download pre-trained models from HuggingFace
    echo "[7/10] Downloading pre-trained models..."
    python -c "
from huggingface_hub import snapshot_download
import os

# Download codec
snapshot_download(
    repo_id='devasphn/telucodec',
    local_dir='$MODEL_DIR/codec',
    token=os.environ.get('HF_TOKEN')
)

# Download S2S model
snapshot_download(
    repo_id='devasphn/telugu-s2s',
    local_dir='$MODEL_DIR/s2s',
    token=os.environ.get('HF_TOKEN')
)
print('✓ Models downloaded')
"
fi

# Optimize models for inference
echo "[9/10] Optimizing models for inference..."
python -c "
import torch
from telugu_codec import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig

# Load models
codec = TeluCodec()
s2s_config = S2SConfig()
s2s = TeluguS2STransformer(s2s_config)

# Compile for faster inference
if hasattr(torch, 'compile'):
    codec = torch.compile(codec, mode='reduce-overhead')
    s2s = torch.compile(s2s, mode='reduce-overhead')
    print('✓ Models compiled')

# Export to TorchScript
codec_scripted = torch.jit.script(codec)
torch.jit.save(codec_scripted, '$MODEL_DIR/codec_scripted.pt')
print('✓ Codec exported to TorchScript')
"

# Start the server
echo "[10/10] Starting Telugu S2S server..."
echo "============================================================"
echo "Server Configuration:"
echo "  - Host: 0.0.0.0"
echo "  - Port: 8000"
echo "  - GPU: $GPU_NAME"
echo "  - Target latency: <150ms"
echo "============================================================"

# Create systemd service for auto-restart
cat > /etc/systemd/system/telugu-s2s.service <<EOF
[Unit]
Description=Telugu S2S Streaming Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
Environment="PATH=/usr/local/bin:/usr/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="HF_TOKEN=$HF_TOKEN"
ExecStart=/usr/bin/python3 $PROJECT_DIR/streaming_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable telugu-s2s
systemctl start telugu-s2s

echo "============================================================"
echo "✅ Deployment Complete!"
echo "============================================================"
echo ""
echo "Access points:"
echo "  - WebSocket: ws://$(hostname -I | awk '{print $1}'):8000/ws"
echo "  - Demo UI: http://$(hostname -I | awk '{print $1}'):8000"
echo "  - Stats API: http://$(hostname -I | awk '{print $1}'):8000/stats"
echo ""
echo "Monitor logs:"
echo "  journalctl -u telugu-s2s -f"
echo ""
echo "Performance targets:"
echo "  - First audio: <150ms"
echo "  - Streaming: Real-time"
echo "  - Emotions: 9 including laughter"
echo "  - Speakers: 4 distinct voices"
echo ""
echo "============================================================"
echo "Beating Luna Demo with in-house Telugu technology! 🚀"
echo "============================================================"