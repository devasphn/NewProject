#!/bin/bash
# Run Fixed Speaker Preparation Script

echo "========================================"
echo "Telugu S2S - Speaker Preparation (FIXED)"
echo "========================================"
echo ""

# Step 1: Clean up old output
echo "Step 1: Cleaning up old output..."
rm -rf /workspace/speaker_data
echo "✓ Cleaned"
echo ""

# Step 2: Run fixed script
echo "Step 2: Running fixed speaker preparation..."
cd /workspace/NewProject

python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data

echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo ""

# Step 3: Verify outputs
echo "Files created:"
ls -lh /workspace/speaker_data/
echo ""

echo "Metadata:"
cat /workspace/speaker_data/metadata.json | python -m json.tool
echo ""

echo "Split counts:"
echo "  Train: $(cat /workspace/speaker_data/train_split.json | grep -o '"speaker_id"' | wc -l)"
echo "  Val: $(cat /workspace/speaker_data/val_split.json | grep -o '"speaker_id"' | wc -l)"
echo "  Test: $(cat /workspace/speaker_data/test_split.json | grep -o '"speaker_id"' | wc -l)"
echo ""

echo "Training set speaker distribution:"
cat /workspace/speaker_data/train_split.json | grep -o '"speaker_id": [0-9]' | sort | uniq -c
echo ""

echo "========================================"
echo "✓ Speaker preparation complete!"
echo "========================================"
