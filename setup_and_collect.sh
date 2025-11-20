#!/bin/bash
# PRODUCTION Telugu Data Collection - Complete Setup Script
# Run this in RunPod web terminal to set everything up and start collection

set -e  # Exit on error

echo "========================================================================"
echo "TELUGU CODEC - PRODUCTION DATA COLLECTION SETUP"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_green() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_red() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Check disk space
echo "Step 1: Checking disk space..."
df -h | grep -E "Filesystem|/workspace"
echo ""

FREE_SPACE=$(df /workspace | awk 'NR==2 {print $4}' | sed 's/G//')
if (( $(echo "$FREE_SPACE < 50" | bc -l) )); then
    print_yellow "WARNING: Less than 50GB free space!"
    echo "You need at least 50GB for collection."
    echo "Continue anyway? (yes/no)"
    read -r response
    if [[ "$response" != "yes" ]]; then
        echo "Exiting."
        exit 1
    fi
else
    print_green "Sufficient disk space available: ${FREE_SPACE}GB"
fi

# Step 2: Update system
echo ""
echo "Step 2: Updating system packages..."
apt-get update -qq
print_green "System updated"

# Step 3: Install ffmpeg
echo ""
echo "Step 3: Installing ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    print_green "ffmpeg already installed"
else
    apt-get install -y ffmpeg > /dev/null 2>&1
    print_green "ffmpeg installed"
fi

# Step 4: Install/upgrade yt-dlp
echo ""
echo "Step 4: Installing yt-dlp (YouTube downloader)..."
pip install --upgrade --quiet yt-dlp
print_green "yt-dlp installed"

# Step 5: Verify installations
echo ""
echo "Step 5: Verifying installations..."
yt-dlp --version > /dev/null 2>&1 && print_green "yt-dlp works" || print_red "yt-dlp failed"
ffmpeg -version > /dev/null 2>&1 && print_green "ffmpeg works" || print_red "ffmpeg failed"

# Step 6: Navigate to project
echo ""
echo "Step 6: Setting up project directory..."
cd /workspace/NewProject || {
    print_red "NewProject directory not found!"
    echo "Please ensure you're in the correct directory."
    exit 1
}
print_green "In project directory: $(pwd)"

# Step 7: Pull latest files
echo ""
echo "Step 7: Pulling latest files from GitHub..."
git pull origin main
print_green "Files updated"

# Step 8: Verify required files
echo ""
echo "Step 8: Verifying required files..."
required_files=(
    "data_sources_PRODUCTION.yaml"
    "download_telugu_data_PRODUCTION.py"
    "calculate_data_requirements.py"
    "PRODUCTION_DATA_COLLECTION_GUIDE.md"
)

all_files_present=true
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_green "$file exists"
    else
        print_red "$file missing!"
        all_files_present=false
    fi
done

if [[ "$all_files_present" == false ]]; then
    print_red "Some files are missing. Please check your git pull."
    exit 1
fi

# Step 9: Show data requirements
echo ""
echo "========================================================================"
echo "DATA REQUIREMENTS CALCULATION"
echo "========================================================================"
python calculate_data_requirements.py

# Step 10: Ask user to confirm
echo ""
echo "========================================================================"
echo "READY TO START COLLECTION"
echo "========================================================================"
echo ""
echo "This will:"
echo "  - Download 180GB of YouTube videos"
echo "  - Extract audio to 16kHz mono"
echo "  - Take approximately 5-6 days"
echo "  - Result in 350-400 hours of Telugu speech"
echo "  - 15+ speakers with diverse accents"
echo ""
echo "Output directory: /workspace/telugu_data_production"
echo ""
echo "Do you want to start collection now? (yes/no)"
read -r response

if [[ "$response" == "yes" ]]; then
    echo ""
    echo "========================================================================"
    echo "STARTING COLLECTION..."
    echo "========================================================================"
    echo ""
    echo "You can monitor progress in another terminal with:"
    echo "  tail -f /workspace/telugu_data_production/collection_log_*.txt"
    echo ""
    echo "Or check statistics with:"
    echo "  cat /workspace/telugu_data_production/collection_stats.json"
    echo ""
    echo "Collection starting in 5 seconds..."
    sleep 5
    
    # Start collection
    python download_telugu_data_PRODUCTION.py \
        --config data_sources_PRODUCTION.yaml \
        --output /workspace/telugu_data_production
    
    echo ""
    echo "========================================================================"
    echo "COLLECTION COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Process audio: python process_audio.py"
    echo "  2. Prepare dataset: python prepare_speaker_data.py"
    echo "  3. Train codec: python train_codec_dac.py"
    echo ""
    
else
    echo ""
    echo "Collection cancelled. You can start it later with:"
    echo ""
    echo "  python download_telugu_data_PRODUCTION.py \\"
    echo "      --config data_sources_PRODUCTION.yaml \\"
    echo "      --output /workspace/telugu_data_production"
    echo ""
fi

print_green "Setup complete!"
