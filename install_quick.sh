#!/bin/bash
# Quick installation script - CONFLICT-FREE
# Run this to install all packages without errors

set -e  # Exit on error

echo "================================================"
echo "Telugu S2S - Quick Install (Conflict-Free)"
echo "================================================"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install main requirements
echo "Installing core packages..."
pip install -r requirements_new.txt

# Install Flash Attention separately
echo "Installing Flash Attention (critical for <150ms latency)..."
pip install flash-attn --no-build-isolation || echo "Warning: Flash Attention failed (optional)"

# Install DeepSpeed (H200 only)
if nvidia-smi | grep -q "H200\|H100"; then
    echo "H200/H100 detected - Installing DeepSpeed..."
    pip install deepspeed==0.14.0 || echo "Warning: DeepSpeed failed (optional)"
fi

# Verify installations
echo ""
echo "================================================"
echo "Verifying installations..."
echo "================================================"

python -c "import torch; print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import librosa; print('✓ Librosa: Installed')"
python -c "import fastapi; print('✓ FastAPI: Installed')"
python -c "import wandb; print('✓ WandB: Installed')"

# Try to import optional packages
python -c "import flash_attn; print('✓ Flash Attention: Installed')" 2>/dev/null || echo "⚠ Flash Attention: Not installed (optional)"
python -c "import deepspeed; print('✓ DeepSpeed: Installed')" 2>/dev/null || echo "⚠ DeepSpeed: Not installed (optional)"

echo ""
echo "================================================"
echo "✅ Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Set up environment variables (see DEPLOYMENT_MANUAL_V2.md)"
echo "2. Start data collection or download models"
echo ""
