#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#   PRODUCTION CODEC TRAINING - READY TO RUN
#   Uses ALL downloaded multilingual data
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd /workspace/NewProject

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "   PRODUCTION CODEC TRAINING"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK DATA
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📊 Checking downloaded data..."

# Count audio files
TOTAL_FILES=0
for dir in data/*/; do
    if [ -d "$dir" ]; then
        COUNT=$(find "$dir" -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) 2>/dev/null | wc -l)
        if [ $COUNT -gt 0 ]; then
            echo "   $dir: $COUNT files"
            TOTAL_FILES=$((TOTAL_FILES + COUNT))
        fi
    fi
done

echo ""
echo "   TOTAL: $TOTAL_FILES audio files"
echo ""

if [ $TOTAL_FILES -lt 100 ]; then
    echo "⚠️ Not enough audio files! Please download data first."
    echo "   Run: bash download_remaining_data.sh"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════
echo "📦 Installing dependencies..."

pip install torch torchaudio transformers einops tensorboard tqdm numpy -q

# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINE BATCH SIZE BASED ON GPU
# ═══════════════════════════════════════════════════════════════════════════════
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

echo ""
echo "🖥️ GPU: $GPU_NAME ($GPU_MEM MB)"

# Set batch size based on GPU
if [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
    BATCH_SIZE=64
    echo "   Using batch_size=64 (High-end GPU)"
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    BATCH_SIZE=48
    echo "   Using batch_size=48 (A100)"
elif [[ "$GPU_NAME" == *"A40"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
    BATCH_SIZE=32
    echo "   Using batch_size=32 (A40/A6000)"
elif [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"3090"* ]]; then
    BATCH_SIZE=24
    echo "   Using batch_size=24 (RTX 3090/4090)"
else
    BATCH_SIZE=16
    echo "   Using batch_size=16 (Default)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DATA DIRS LIST
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIRS=""
for dir in data/*/; do
    if [ -d "$dir" ]; then
        COUNT=$(find "$dir" -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) 2>/dev/null | wc -l)
        if [ $COUNT -gt 0 ]; then
            DATA_DIRS="$DATA_DIRS $dir"
        fi
    fi
done

echo ""
echo "📁 Using data directories:$DATA_DIRS"

# ═══════════════════════════════════════════════════════════════════════════════
# START TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "🚀 STARTING TRAINING..."
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

python train_codec_production.py \
    --data_dirs $DATA_DIRS \
    --batch_size $BATCH_SIZE \
    --num_epochs 100 \
    --gen_lr 1e-4 \
    --disc_lr 1e-4 \
    --checkpoint_dir checkpoints_production \
    --num_workers 4

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "✅ TRAINING COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════════════"
