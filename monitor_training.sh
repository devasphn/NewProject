#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#   TRAINING MONITOR - GPU & System Stats
#   Run in a separate terminal: bash monitor_training.sh
# ═══════════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  TRAINING MONITOR"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found!"
    exit 1
fi

# Print GPU info once
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Continuous monitoring
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  LIVE MONITORING (Ctrl+C to stop)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        CODEC TRAINING MONITOR                                ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # GPU Stats
    echo "🎮 GPU STATUS:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU Util: %s%%  |  Mem: %s/%s MB (%s%%)  |  Temp: %s°C  |  Power: %sW\n", $1, $3, $4, $2, $5, $6}'
    echo ""
    
    # CPU & Memory
    echo "🖥️  SYSTEM STATUS:"
    echo "   CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%"
    echo "   RAM: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
    echo "   Disk: $(df -h /workspace | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
    echo ""
    
    # Training Process
    echo "📈 TRAINING PROCESS:"
    if pgrep -f "train_codec_production" > /dev/null; then
        PID=$(pgrep -f "train_codec_production" | head -1)
        echo "   Status: ✅ RUNNING (PID: $PID)"
        echo "   Runtime: $(ps -o etime= -p $PID 2>/dev/null || echo 'N/A')"
    else
        echo "   Status: ❌ NOT RUNNING"
    fi
    echo ""
    
    # Checkpoints
    echo "💾 CHECKPOINTS:"
    if [ -d "/workspace/NewProject/checkpoints_production" ]; then
        LATEST=$(ls -t /workspace/NewProject/checkpoints_production/*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "   Latest: $(basename $LATEST)"
            echo "   Size: $(du -h $LATEST | cut -f1)"
            echo "   Time: $(stat -c %y $LATEST 2>/dev/null | cut -d'.' -f1)"
        else
            echo "   No checkpoints yet"
        fi
    fi
    echo ""
    
    # TensorBoard info
    echo "📊 TENSORBOARD:"
    if [ -d "/workspace/NewProject/checkpoints_production/logs" ]; then
        echo "   Logs: $(ls /workspace/NewProject/checkpoints_production/logs/ 2>/dev/null | tail -1)"
        echo "   View: tensorboard --logdir=/workspace/NewProject/checkpoints_production/logs --port=6006"
    fi
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "  Updated: $(date '+%Y-%m-%d %H:%M:%S')  |  Press Ctrl+C to stop"
    echo "───────────────────────────────────────────────────────────────────────────────"
    
    sleep 5
done
