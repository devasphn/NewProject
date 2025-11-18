#!/bin/bash
# Quick Fix Script - Update packages and retry setup

echo "=========================================="
echo "Fixing Llama Download Issue"
echo "=========================================="

# Step 1: Pull latest code
echo "[1/3] Pulling latest code from GitHub..."
git pull origin main
if [ $? -ne 0 ]; then
    echo "⚠ Warning: Git pull failed (might be no changes)"
fi

# Step 2: Update packages
echo "[2/3] Updating transformers and accelerate..."
pip install --upgrade transformers==4.45.0 accelerate==0.33.0 --quiet
echo "✓ Packages updated"

# Step 3: Run setup
echo "[3/3] Running setup script..."
bash startup.sh

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
