# 🎯 PRODUCTION CODEC - COMPLETE GUIDE

## ✅ License Verification

### WavLM License: **MIT (Commercial Use ALLOWED!)**

```
Source: https://github.com/microsoft/unilm/blob/master/LICENSE

The MIT License (MIT)
Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

**✅ You CAN use WavLM for:**
- Commercial products
- Production deployments
- Your company's voice AI
- No attribution required (but citation appreciated)

---

## 📊 Production Codec Features

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **Hybrid Architecture** | CNN + 4 Transformer layers | Best of both worlds |
| **50Hz Frame Rate** | Down from 200Hz | 4x fewer tokens for S2S |
| **Semantic Distillation** | WavLM teacher | Better S2S understanding |
| **Variable Codebooks** | [2048, 1024×7] | Semantic + Acoustic separation |
| **Snake Activation** | DAC-style | Better audio modeling |
| **Causal Convolutions** | Streaming ready | Real-time inference |
| **Multi-scale Spectral Loss** | 512, 1024, 2048 FFT | Better reconstruction |
| **GAN Training** | DAC Discriminator | High quality audio |

### Technical Specifications

```
Sample Rate:        16,000 Hz
Frame Rate:         50 Hz (16000 / 320)
Bitrate:            ~4.5 kbps
Quantizers:         8 layers
Codebook Sizes:     [2048, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
Hidden Dimension:   1024
Transformer Layers: 4 (encoder) + 4 (decoder)
Total Parameters:   ~80-100M
```

---

## 📁 DATASETS REQUIRED

### 🎯 Minimum for Production: 500+ hours mixed audio

### Tier 1: FREE Telugu Data (No Attribution Required)

| Dataset | Hours | Language | License | Download Method |
|---------|-------|----------|---------|-----------------|
| **Kathbath** | ~140h | Telugu | CC0 | HuggingFace (agreement needed) |
| **OpenSLR SLR66** | 10h | Telugu | CC-BY-4.0 | wget |
| **OpenSLR SLR103** | 40h | Telugu | Free | wget |
| **Common Voice Telugu** | 20h | Telugu | CC-0 | HuggingFace |
| **IndicTTS Telugu** | 9h | Telugu | CC-BY-4.0 | GitHub |

### Tier 2: Multilingual Data (Recommended for Better Codec)

| Dataset | Hours | Languages | License | Download Method |
|---------|-------|-----------|---------|-----------------|
| **LibriSpeech** | 960h | English | CC-BY-4.0 | wget |
| **Common Voice Hindi** | 50h | Hindi | CC-0 | HuggingFace |
| **Common Voice Tamil** | 30h | Tamil | CC-0 | HuggingFace |
| **LJSpeech** | 24h | English | Public Domain | wget |
| **VCTK** | 44h | English (multi-speaker) | CC-BY-4.0 | wget |

### Tier 3: SEA Languages (For Multilingual Support)

| Dataset | Hours | Languages | License | Download Method |
|---------|-------|-----------|---------|-----------------|
| **Common Voice Thai** | 20h | Thai | CC-0 | HuggingFace |
| **Common Voice Indonesian** | 15h | Indonesian | CC-0 | HuggingFace |
| **Common Voice Vietnamese** | 10h | Vietnamese | CC-0 | HuggingFace |
| **FLEURS** | Multi | 102 languages | CC-BY-4.0 | HuggingFace |

### Storage Requirements

```
Tier 1 (Telugu only):      ~25 GB
Tier 2 (+ Multilingual):   ~150 GB  
Tier 3 (+ SEA):            ~200 GB
Total with buffer:         ~250 GB
```

---

## 🚀 DOWNLOAD COMMANDS

### Step 1: Setup RunPod Environment

```bash
# Connect to RunPod and go to workspace
cd /workspace

# Clone your project
git clone <your-repo-url> NewProject
cd NewProject

# Install dependencies
pip install torch torchaudio transformers einops tensorboard tqdm
pip install datasets huggingface_hub  # For HuggingFace datasets
```

### Step 2: Download Datasets

```bash
# Create data directories
mkdir -p data/telugu data/english data/hindi data/tamil data/sea

# ═══════════════════════════════════════════════════════════════
# TIER 1: TELUGU DATA
# ═══════════════════════════════════════════════════════════════

# OpenSLR SLR66 (10h Telugu, direct download)
wget -P data/telugu https://www.openslr.org/resources/66/te_in_female.zip
wget -P data/telugu https://www.openslr.org/resources/66/te_in_male.zip
cd data/telugu && unzip "*.zip" && rm *.zip && cd ../..

# OpenSLR SLR103 - MUCS Challenge (40h Telugu)
wget -P data/telugu https://www.openslr.org/resources/103/Telugu.zip
cd data/telugu && unzip Telugu.zip && rm Telugu.zip && cd ../..

# IndicTTS Telugu (9h)
git clone https://github.com/AI4Bharat/Indic-TTS data/telugu/indic_tts

# ═══════════════════════════════════════════════════════════════
# TIER 2: ENGLISH & OTHER INDIAN LANGUAGES
# ═══════════════════════════════════════════════════════════════

# LibriSpeech (960h English)
# Download train-clean-100 first (smaller, faster)
wget -P data/english https://www.openslr.org/resources/12/train-clean-100.tar.gz
cd data/english && tar -xzf train-clean-100.tar.gz && rm *.tar.gz && cd ../..

# LJSpeech (24h English, single speaker)
wget -P data/english https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
cd data/english && tar -xjf LJSpeech-1.1.tar.bz2 && rm *.tar.bz2 && cd ../..

# ═══════════════════════════════════════════════════════════════
# HUGGINGFACE DATASETS (Kathbath, Common Voice)
# ═══════════════════════════════════════════════════════════════
```

### Step 3: Download HuggingFace Datasets (Python)

Create `download_hf_datasets.py`:

```python
"""Download HuggingFace datasets for codec training"""

from datasets import load_dataset
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import os

# Create directories
os.makedirs("data/kathbath", exist_ok=True)
os.makedirs("data/common_voice_te", exist_ok=True)
os.makedirs("data/common_voice_hi", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# KATHBATH TELUGU (~140h)
# NOTE: You need to accept the license on HuggingFace first!
# Visit: https://huggingface.co/datasets/ai4bharat/Kathbath
# ═══════════════════════════════════════════════════════════════

print("📥 Downloading Kathbath Telugu...")
try:
    kathbath = load_dataset("ai4bharat/Kathbath", "telugu", split="train", trust_remote_code=True)
    
    for i, sample in enumerate(tqdm(kathbath, desc="Saving Kathbath")):
        audio = sample['audio']
        path = f"data/kathbath/kathbath_{i:06d}.wav"
        sf.write(path, audio['array'], audio['sampling_rate'])
        
        if i >= 50000:  # Limit to avoid timeout
            break
except Exception as e:
    print(f"⚠️ Kathbath error: {e}")
    print("   Make sure you accepted the license on HuggingFace!")

# ═══════════════════════════════════════════════════════════════
# COMMON VOICE TELUGU (~20h)
# ═══════════════════════════════════════════════════════════════

print("\n📥 Downloading Common Voice Telugu...")
try:
    cv_te = load_dataset("mozilla-foundation/common_voice_16_1", "te", split="train", trust_remote_code=True)
    
    for i, sample in enumerate(tqdm(cv_te, desc="Saving CV Telugu")):
        audio = sample['audio']
        path = f"data/common_voice_te/cv_te_{i:06d}.wav"
        sf.write(path, audio['array'], audio['sampling_rate'])
except Exception as e:
    print(f"⚠️ Common Voice Telugu error: {e}")

# ═══════════════════════════════════════════════════════════════
# COMMON VOICE HINDI (~50h)
# ═══════════════════════════════════════════════════════════════

print("\n📥 Downloading Common Voice Hindi...")
try:
    cv_hi = load_dataset("mozilla-foundation/common_voice_16_1", "hi", split="train", trust_remote_code=True)
    
    for i, sample in enumerate(tqdm(cv_hi, desc="Saving CV Hindi")):
        audio = sample['audio']
        path = f"data/common_voice_hi/cv_hi_{i:06d}.wav"
        sf.write(path, audio['array'], audio['sampling_rate'])
        
        if i >= 30000:  # Limit
            break
except Exception as e:
    print(f"⚠️ Common Voice Hindi error: {e}")

print("\n✅ Download complete!")
```

Run:
```bash
python download_hf_datasets.py
```

---

## 🏋️ TRAINING COMMANDS

### Option A: Train from Scratch (RECOMMENDED for Production)

```bash
# Full production training
python train_codec_production.py \
    --data_dirs data/telugu data/english data/hindi \
    --batch_size 16 \
    --num_epochs 200 \
    --gen_lr 1e-4 \
    --disc_lr 1e-4 \
    --checkpoint_dir checkpoints_production \
    --num_workers 4
```

### Option B: Resume from Existing Checkpoint

```bash
# If you want to use your existing codec as starting point
python train_codec_production.py \
    --data_dirs data/telugu data/english data/hindi \
    --batch_size 16 \
    --num_epochs 200 \
    --checkpoint_dir checkpoints_production \
    --resume checkpoints/best_codec.pt
```

### Training Parameters by GPU

| GPU | VRAM | Batch Size | Workers |
|-----|------|------------|---------|
| RTX 3090 | 24GB | 16 | 4 |
| RTX 4090 | 24GB | 24 | 4 |
| A100 40GB | 40GB | 32 | 8 |
| A100 80GB | 80GB | 48 | 8 |

---

## ❓ SHOULD YOU TRAIN FROM SCRATCH?

### Answer: **YES, train from scratch for production!**

| Reason | Explanation |
|--------|-------------|
| **Architecture changed** | New codec has Transformer layers, 50Hz frame rate |
| **Codebook sizes changed** | First layer is 2048 now |
| **Semantic layer added** | WavLM distillation is new |
| **More data** | Training on multilingual data |

### Your Existing 785MB Codec
- ❌ Different architecture (no Transformer)
- ❌ 200Hz frame rate (too high)
- ❌ No semantic layer
- ❌ Single codebook size (all 1024)

**Verdict: Start fresh with new production codec!**

---

## 📋 TRAINING CHECKLIST

### Before Training

- [ ] RunPod pod created with 800GB+ volume
- [ ] All dependencies installed
- [ ] Downloaded Tier 1 Telugu data (~25GB)
- [ ] Downloaded Tier 2 English data (~100GB)
- [ ] Accepted Kathbath license on HuggingFace
- [ ] Test codec can run: `python codec_production.py`

### During Training

- [ ] Monitor TensorBoard: `tensorboard --logdir checkpoints_production/logs`
- [ ] Check GPU usage: `nvidia-smi`
- [ ] Watch for codebook collapse (check logs)
- [ ] Validate audio quality every 10 epochs

### Training Timeline (Estimated)

| Phase | Epochs | Time (A100) | Expected Loss |
|-------|--------|-------------|---------------|
| Initial | 0-20 | ~6 hours | ~1.0 → 0.3 |
| GAN starts | 20-50 | ~10 hours | ~0.3 → 0.15 |
| Fine-tuning | 50-100 | ~15 hours | ~0.15 → 0.08 |
| Final | 100-200 | ~30 hours | ~0.08 → 0.05 |

**Total: ~60 hours on A100** (~$150 at $2.5/hr)

---

## 🔍 VALIDATION TESTS

### Test 1: Basic Functionality
```bash
python codec_production.py
# Should print "All tests passed!"
```

### Test 2: Audio Quality Check
```python
import torch
import torchaudio
from codec_production import ProductionCodec

# Load codec
codec = ProductionCodec().cuda()
codec.load_state_dict(torch.load("checkpoints_production/best_codec.pt")['codec_state_dict'])
codec.eval()

# Load test audio
audio, sr = torchaudio.load("test_audio.wav")
audio = torchaudio.transforms.Resample(sr, 16000)(audio)
audio = audio.unsqueeze(0).cuda()

# Encode/decode
with torch.no_grad():
    codes = codec.encode(audio)
    recon = codec.decode(codes)

# Save and compare
torchaudio.save("reconstructed.wav", recon.squeeze(0).cpu(), 16000)
print("Compare test_audio.wav and reconstructed.wav by listening!")
```

### Test 3: SNR Check
```python
# Calculate SNR
signal = audio.squeeze()
noise = signal - recon.squeeze().cpu()
snr = 10 * torch.log10((signal**2).mean() / (noise**2).mean())
print(f"SNR: {snr.item():.1f} dB")
# Target: > 15 dB is good, > 20 dB is excellent
```

---

## 🚀 NEXT STEPS AFTER CODEC

1. **Validate codec quality** (SNR > 15dB, good listening test)
2. **Generate conversation pairs** using trained codec
3. **Train S2S model** on conversation pairs
4. **Test end-to-end pipeline**

---

## 📁 FILES CREATED

| File | Purpose |
|------|---------|
| `codec_production.py` | Production codec architecture |
| `train_codec_production.py` | Training script |
| `PRODUCTION_CODEC_GUIDE.md` | This guide |

---

## ⚠️ IMPORTANT NOTES

1. **WavLM will auto-download** on first run (~400MB)
2. **Semantic distillation is optional** - training works without it
3. **50Hz frame rate** means 4x fewer tokens for S2S
4. **First codebook (2048)** captures semantics better
5. **GAN training starts at epoch 5** - this is intentional for stability

---

## 💡 TIPS

1. **Start small**: First train on 100 hours, validate, then scale up
2. **Monitor codebook usage**: All codes should be used
3. **Listen to samples**: Numbers don't tell the full story
4. **Save checkpoints frequently**: GPU time is expensive!
