# 🎯 TELUGU 200+ HOURS - COMPLETE DOWNLOAD PLAN

## 📊 Telugu Data Sources Summary

| Source | Hours | License | Access |
|--------|-------|---------|--------|
| **OpenSLR 66** | ~10h | CC-BY-4.0 | ✅ Direct wget |
| **Kathbath Telugu** | ~140h | **CC0** | HuggingFace (accept) |
| **IndicVoices Telugu** | ~300h+ | CC-BY-4.0 | HuggingFace (accept) |
| **Common Voice Telugu** | ~20h | CC-0 | HuggingFace |
| **TOTAL** | **~470+ hours** | | |

---

## 🚀 STEP-BY-STEP COMMANDS

### Step 1: OpenSLR 66 (Already done - ~10h)
```bash
# You already have this!
ls data/telugu/
```

### Step 2: Download Kathbath Telugu (~140h) - CC0 LICENSE!

**First: Accept license at HuggingFace**
1. Go to: https://huggingface.co/datasets/ai4bharat/Kathbath
2. Click "Agree and access repository"
3. Login with your HuggingFace account

**Then run:**
```bash
pip install datasets huggingface_hub soundfile tqdm

# Login to HuggingFace
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

**Download script:**
```python
# Save as: download_kathbath_telugu.py
from datasets import load_dataset
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import os

output_dir = Path("data/kathbath_telugu")
output_dir.mkdir(parents=True, exist_ok=True)

print("📥 Downloading Kathbath Telugu (~140 hours)...")

dataset = load_dataset(
    "ai4bharat/Kathbath",
    "telugu",
    split="train",
    streaming=True,
    trust_remote_code=True
)

count = 0
for sample in tqdm(dataset, desc="Saving Telugu"):
    try:
        audio = sample['audio']
        filename = output_dir / f"kathbath_te_{count:08d}.wav"
        sf.write(str(filename), audio['array'], audio['sampling_rate'])
        count += 1
    except Exception as e:
        continue

print(f"✅ Saved {count} Telugu audio files!")
```

Run:
```bash
python download_kathbath_telugu.py
```

### Step 3: Download IndicVoices Telugu (~300h+)

**First: Accept license at HuggingFace**
1. Go to: https://huggingface.co/datasets/ai4bharat/IndicVoices
2. Click "Agree and access repository"

**Download script:**
```python
# Save as: download_indicvoices_telugu.py
from datasets import load_dataset
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

output_dir = Path("data/indicvoices_telugu")
output_dir.mkdir(parents=True, exist_ok=True)

print("📥 Downloading IndicVoices Telugu (~300+ hours)...")

dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "telugu",
    split="train",
    streaming=True,
    trust_remote_code=True
)

count = 0
for sample in tqdm(dataset, desc="Saving Telugu"):
    try:
        audio = sample['audio']
        filename = output_dir / f"indicvoices_te_{count:08d}.wav"
        sf.write(str(filename), audio['array'], audio['sampling_rate'])
        count += 1
    except Exception as e:
        continue

print(f"✅ Saved {count} Telugu audio files!")
```

Run:
```bash
python download_indicvoices_telugu.py
```

### Step 4: Download Common Voice Telugu (~20h)

```python
# Save as: download_cv_telugu.py
from datasets import load_dataset
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

output_dir = Path("data/commonvoice_telugu")
output_dir.mkdir(parents=True, exist_ok=True)

print("📥 Downloading Common Voice Telugu (~20 hours)...")

dataset = load_dataset(
    "mozilla-foundation/common_voice_16_1",
    "te",
    split="train",
    streaming=True,
    trust_remote_code=True
)

count = 0
for sample in tqdm(dataset, desc="Saving Telugu"):
    try:
        audio = sample['audio']
        filename = output_dir / f"cv_te_{count:08d}.wav"
        sf.write(str(filename), audio['array'], audio['sampling_rate'])
        count += 1
    except Exception as e:
        continue

print(f"✅ Saved {count} Telugu audio files!")
```

Run:
```bash
python download_cv_telugu.py
```

---

## 🖥️ H200 vs A40 COMPARISON

| GPU | VRAM | FP16 TFLOPS | Price/hr | Training Speed |
|-----|------|-------------|----------|----------------|
| A40 | 48GB | 37.4 | ~$1.00 | 1x (baseline) |
| **H200** | **141GB** | **989** | ~$4-5 | **~10-15x faster!** |
| H100 | 80GB | 756 | ~$3-4 | ~8-10x faster |
| A100-80GB | 80GB | 312 | ~$2-3 | ~3-4x faster |

### Recommendation: **YES, switch to H200!**

**Why H200:**
- 141GB VRAM = can train with **batch_size=64+**
- 989 TFLOPS = **10-15x faster** than A40
- Faster training = **less total cost** despite higher hourly rate

**Cost comparison for 60 hours training:**
- A40: 60h × $1.00 = $60 (but takes 60 hours)
- H200: 6h × $4.50 = $27 (10x faster, **saves $33 + 54 hours!**)

---

## 📋 QUICK COMMANDS FOR RUNPOD

### Before switching GPU:

```bash
# Save your progress - download data to /workspace (persistent)
cd /workspace/NewProject
ls data/  # Check what you have

# Your data in /workspace survives pod termination!
```

### After getting H200:

```bash
# Everything in /workspace is still there!
cd /workspace/NewProject

# Install dependencies
pip install torch torchaudio transformers einops tensorboard tqdm datasets huggingface_hub soundfile

# Login to HuggingFace (for Kathbath & IndicVoices)
huggingface-cli login

# Download Telugu data (200+ hours)
python download_kathbath_telugu.py
python download_indicvoices_telugu.py
python download_cv_telugu.py

# Start training with larger batch size on H200!
python train_codec_production.py \
    --data_dirs data/telugu data/kathbath_telugu data/indicvoices_telugu data/commonvoice_telugu \
    --batch_size 64 \
    --num_epochs 100 \
    --checkpoint_dir checkpoints_production
```

---

## 🎯 FINAL TELUGU DATA BREAKDOWN

After all downloads:

```
data/
├── telugu/                    # OpenSLR 66 (~10h)
│   ├── te_in_female/
│   └── te_in_male/
├── kathbath_telugu/           # Kathbath (~140h) 
│   └── *.wav
├── indicvoices_telugu/        # IndicVoices (~300h)
│   └── *.wav
└── commonvoice_telugu/        # Common Voice (~20h)
    └── *.wav

TOTAL TELUGU: ~470 hours! ✅
```

---

## ⚡ TRAINING TIME ESTIMATE

| GPU | Batch Size | Epochs | Time | Cost |
|-----|------------|--------|------|------|
| A40 | 16 | 100 | ~80h | ~$80 |
| **H200** | 64 | 100 | **~8h** | **~$40** |

**H200 saves you 72 hours and $40!**
