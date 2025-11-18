#!/bin/bash
# Startup Script for Telugu S2S Voice Agent
# For RunPod RTX A6000 Web Terminal
# 
# Prerequisites (run manually before this script):
# - apt-get update && apt-get install -y ffmpeg git
# - cd /workspace
# - git clone https://github.com/devasphn/NewProject.git
# - cd NewProject
# - bash startup.sh

set -e  # Exit on error

echo "=================================================="
echo "Telugu S2S Voice Agent - Startup Script"
echo "GPU: RTX A6000 (48GB)"
echo "GitHub: devasphn/NewProject"
echo "=================================================="

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Verify we're in the right directory
echo -e "${BLUE}[1/7] Verifying directory...${NC}"
if [ ! -f "config.py" ]; then
    echo -e "${RED}❌ Error: config.py not found!${NC}"
    echo "Make sure you're in the NewProject directory"
    echo "Run: cd /workspace/NewProject"
    exit 1
fi
echo -e "${GREEN}✓ Directory verified${NC}"

# Step 2: Verify GPU
echo -e "${BLUE}[2/7] Verifying GPU...${NC}"
nvidia-smi
echo -e "${GREEN}✓ GPU verified: RTX A6000${NC}"

# Step 3: Install Python dependencies
echo -e "${BLUE}[3/7] Installing Python packages (5-10 minutes)...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Python packages installed${NC}"

# Step 4: Create necessary directories
echo -e "${BLUE}[4/7] Creating directories...${NC}"
mkdir -p models
mkdir -p telugu_data
mkdir -p outputs
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Step 5: Check HuggingFace token
echo -e "${BLUE}[5/7] Checking HuggingFace token...${NC}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}⚠ Warning: HF_TOKEN not set!${NC}"
    echo "Llama model download may fail without it"
    echo "Set it with: export HF_TOKEN='your_token_here'"
    echo "Get token from: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
fi

# Step 6: Download pre-trained models
echo -e "${BLUE}[6/7] Downloading models (15-20 minutes)...${NC}"
python download_models.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Model download failed!${NC}"
    echo "Check the error above and fix before continuing"
    exit 1
fi
echo -e "${GREEN}✓ Models downloaded${NC}"

# Step 7: Test latency (baseline before Telugu training)
echo -e "${BLUE}[7/7] Testing baseline latency...${NC}"
python test_latency.py --mode baseline
echo -e "${GREEN}✓ Baseline latency tested${NC}"

# Done!
echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "To start the server:"
echo "  python server.py"
echo ""
echo "To access demo:"
echo "  1. Go to RunPod dashboard"
echo "  2. Click 'Connect' → 'HTTP Service [Port 8000]'"
echo "  3. Browser demo will open"
echo ""
echo "To train Telugu model:"
echo "  1. Edit download_telugu.py (add YouTube URLs)"
echo "  2. Run: bash train_telugu.sh"
echo "  3. Restart server: python server.py"
echo ""
echo "=================================================="
echo ""
echo "Ready to start server? (python server.py)"
