# 📊 Storage Calculator - Telugu S2S Project

## Current Data: 80GB Raw Video

### Data Breakdown

#### Phase 1: Raw Data (Already Downloaded)
```
Raw YouTube Videos:           80 GB
├── Video streams:            ~60 GB
└── Audio streams:            ~20 GB
```

#### Phase 2: Extracted Audio
```
Extracted WAV files (16kHz):  ~80 GB
├── Mono, 16-bit PCM
├── 16,000 Hz sample rate
└── Lossless quality
```

#### Phase 3: Processed & Split Data
```
Train set (80%):              ~64 GB
Validation set (10%):         ~8 GB
Test set (10%):               ~8 GB
─────────────────────────────────
Total processed:              ~80 GB
```

**Note:** Processed = extracted audio, so total audio storage is ~80GB, not 160GB

---

## Model Storage During Training

### Codec Training
```
Model checkpoints (every 5 epochs): ~5 GB
  ├── Encoder:                      ~100 MB
  ├── Decoder:                      ~100 MB
  ├── VQ codebooks:                 ~50 MB
  └── Discriminators:               ~200 MB
  └── × 20 checkpoints              = ~5 GB

Best model + EMA:                   ~500 MB
Optimizer state:                    ~1 GB
Gradient checkpoints:               ~2 GB
Training artifacts:                 ~1.5 GB
─────────────────────────────────────────
Codec training total:               ~10 GB
```

### S2S Training
```
Model checkpoints (every 5 epochs): ~10 GB
  ├── Encoder (Conformer):          ~200 MB
  ├── Decoder (Transformer):        ~300 MB
  ├── Speaker embeddings:           ~50 MB
  └── × 20 checkpoints              = ~10 GB

Best model + EMA:                   ~1 GB
Optimizer state:                    ~2 GB
Gradient checkpoints:               ~5 GB
Training artifacts:                 ~2 GB
─────────────────────────────────────────
S2S training total:                 ~20 GB
```

### Speaker Training
```
Speaker embeddings:                 ~500 MB
Training checkpoints:               ~2 GB
─────────────────────────────────────────
Speaker training total:             ~2.5 GB
```

---

## Logs & Monitoring

### WandB Logs
```
Training metrics:                   ~5 GB
  ├── Loss curves
  ├── Audio samples
  ├── Spectrograms
  └── System metrics
```

### TensorBoard
```
TensorBoard logs:                   ~3 GB
  ├── Scalars
  ├── Histograms
  └── Embeddings
```

### Text Logs
```
Python logs (JSON, txt):            ~2 GB
```

**Total logs:**                     ~10 GB

---

## Processing & Temporary Files

### Data Processing
```
Temporary audio chunks:             ~10 GB
FFmpeg cache:                       ~5 GB
WhisperX transcripts:               ~500 MB
Speaker diarization:                ~500 MB
──────────────────────────────────────
Processing artifacts:               ~16 GB
```

### System Cache
```
Pip cache:                          ~5 GB
HuggingFace cache:                  ~10 GB
──────────────────────────────────────
System cache:                       ~15 GB
```

---

## 🎯 TOTAL STORAGE CALCULATION

### Container Disk (System + Models + Processing)
```
System & Base Image:                ~30 GB (from pytorch:2.2.0)
Python packages:                    ~15 GB
Project code:                       ~100 MB
Raw video data:                     ~80 GB
Extracted audio:                    ~80 GB
Model checkpoints:                  ~32.5 GB (codec + s2s + speaker)
Logs & monitoring:                  ~10 GB
Processing artifacts:               ~16 GB
System cache:                       ~15 GB
Buffer (safety):                    ~20 GB
─────────────────────────────────────────────
TOTAL NEEDED:                       ~299 GB
```

### Volume Disk (Long-term Storage + Backups)
```
Model backups:                      ~50 GB
Data backups:                       ~80 GB
Exported models (ONNX, TRT):        ~10 GB
Test outputs:                       ~10 GB
Additional data collection:         ~200 GB (future)
Buffer:                             ~150 GB
─────────────────────────────────────────────
TOTAL NEEDED:                       ~500 GB
```

---

## ✅ RECOMMENDATIONS

### Scenario 1: Tight Budget (Minimum)
```
Container Disk: 300 GB              ⚠️  TIGHT (96% used)
Volume Disk:    500 GB              ✅  Good
```
- **Risk:** Very little buffer for unexpected files
- **Works if:** You're careful about disk usage
- **Monitor:** `watch -n 60 df -h`

### Scenario 2: Recommended (Balanced)
```
Container Disk: 400 GB              ✅  Comfortable (75% used)
Volume Disk:    500 GB              ✅  Good
```
- **Pros:** 25% buffer for safety
- **Cost:** +$0.30/hour (~$7/day extra)
- **Best for:** Production training

### Scenario 3: Optimal (Generous)
```
Container Disk: 500 GB              ✅  Spacious (60% used)
Volume Disk:    1 TB                ✅  Excellent
```
- **Pros:** No storage worries, room for experiments
- **Cost:** +$0.60/hour (~$14/day extra)
- **Best for:** Long-term projects, multiple experiments

---

## 💾 Storage Monitoring Commands

### Check Disk Usage
```bash
# Overall disk usage
df -h

# Container disk
df -h /workspace

# Volume disk (if mounted)
df -h /runpod-volume

# Detailed directory sizes
du -sh /workspace/*

# Find largest files
find /workspace -type f -size +1G -exec ls -lh {} \; | awk '{ print $9 ": " $5 }'

# Clean up if needed
rm -rf /workspace/NewProject/.git  # ~50MB
pip cache purge                     # ~5GB
```

### Set Up Alerts
```bash
# Add to ~/.bashrc
echo 'alias checkdisk="df -h | grep workspace"' >> ~/.bashrc

# Or create a monitoring script
cat > /workspace/check_storage.sh << 'EOF'
#!/bin/bash
USAGE=$(df -h /workspace | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $USAGE -gt 90 ]; then
    echo "⚠️  WARNING: Disk usage at ${USAGE}%"
else
    echo "✅ Disk usage: ${USAGE}%"
fi
EOF
chmod +x /workspace/check_storage.sh
```

---

## 🗑️ Clean Up Strategies (If Running Low)

### Safe to Delete During Training
```bash
# Old pip cache
pip cache purge                     # ~5GB

# Old HuggingFace cache (careful!)
rm -rf ~/.cache/huggingface/hub/*   # ~10GB (will re-download if needed)

# Temporary processing files
rm -rf /tmp/*                       # ~2-5GB

# Old checkpoints (keep only best + last 3)
find /workspace/checkpoints -name "*.pt" -mtime +7 -delete
```

### Safe to Delete After Training
```bash
# Raw video files (keep extracted audio only)
rm -rf /workspace/telugu_data/raw/*.mp4  # ~60GB

# Old training logs (compress or upload to cloud)
tar -czf logs_backup.tar.gz /workspace/logs
# Then delete originals after backing up
```

### NEVER Delete
```bash
# ❌ /workspace/telugu_data/processed/  (extracted audio)
# ❌ /workspace/models/*/best_model.pt  (best models)
# ❌ /workspace/NewProject/              (code)
# ❌ /workspace/checkpoints/*/latest.pt (resume training)
```

---

## 📈 Storage Usage Timeline

### Day 1-2: Data Collection
```
Start:    30 GB  (system + packages)
After:    190 GB (+ raw videos + extracted audio)
```

### Day 3-7: Codec Training
```
Start:    190 GB
After:    210 GB (+ codec models + checkpoints)
```

### Day 8-14: S2S Training
```
Start:    210 GB
After:    240 GB (+ s2s models + checkpoints)
```

### Day 15+: Testing & Deployment
```
Final:    260 GB (+ logs + test outputs)
```

**Peak Usage:** ~260-280 GB (leaves 20-40GB buffer with 300GB container)

---

## 🚨 Emergency: Running Out of Space

### If you get "No space left on device"

1. **Check what's using space:**
   ```bash
   du -sh /workspace/* | sort -rh | head -20
   ```

2. **Quick wins:**
   ```bash
   pip cache purge              # ~5GB
   rm -rf /tmp/*                # ~2-5GB
   rm /workspace/*.mp4          # Raw videos if audio extracted
   ```

3. **Delete old checkpoints:**
   ```bash
   # Keep only last 5 checkpoints
   cd /workspace/checkpoints/codec
   ls -t *.pt | tail -n +6 | xargs rm --
   ```

4. **Upload to cloud and delete:**
   ```bash
   # Upload to HuggingFace
   huggingface-cli upload Devakumar868/telugu-codec-poc \
     /workspace/checkpoints/codec/checkpoint_epoch_50.pt

   # Then delete local copy
   rm /workspace/checkpoints/codec/checkpoint_epoch_50.pt
   ```

---

## ✅ FINAL ANSWER

### For 80GB Data:
```
┌─────────────────────────────────────────┐
│  RECOMMENDED RUNPOD CONFIGURATION       │
├─────────────────────────────────────────┤
│  Container Disk: 400 GB  ✅ (safe)      │
│  Volume Disk:    500 GB  ✅ (excellent) │
│                                         │
│  Alternative (minimum):                 │
│  Container Disk: 300 GB  ⚠️  (tight)    │
│  Volume Disk:    500 GB  ✅ (excellent) │
└─────────────────────────────────────────┘
```

**Cost difference:** ~$7/day extra for 400GB vs 300GB  
**Worth it?** YES - avoids storage panic during training

---

## 📞 Quick Reference

**Check space:**
```bash
df -h | grep workspace
```

**Clean up safely:**
```bash
pip cache purge && rm -rf /tmp/*
```

**Monitor in real-time:**
```bash
watch -n 60 "df -h | grep workspace"
```

**Emergency backup:**
```bash
# Upload critical files before cleaning
huggingface-cli upload Devakumar868/telugu-codec-poc /workspace/models
```

---

**Ready for deployment!** 🚀
