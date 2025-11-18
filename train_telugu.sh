#!/bin/bash
# Telugu Training Workflow

echo "================================"
echo "Telugu Training Pipeline"
echo "================================"

# Step 1: Download Telugu data
echo "[1/4] Downloading Telugu data..."
python download_telugu.py
if [ $? -ne 0 ]; then
    echo "❌ Download failed!"
    exit 1
fi

# Step 2: Train Telugu model
echo "[2/4] Training Telugu model..."
python train_telugu.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 3: Test latency
echo "[3/4] Testing Telugu model latency..."
python test_latency.py --mode telugu

# Step 4: Done
echo "[4/4] Complete!"
echo ""
echo "✓ Telugu model trained and tested"
echo "Restart server to use Telugu model:"
echo "  python server.py"
