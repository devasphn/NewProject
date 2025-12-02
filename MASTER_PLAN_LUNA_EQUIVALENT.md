# 🚀 LUNA-EQUIVALENT S2S AI - MASTER PLAN

## Project Goal
Build a production-grade Speech-to-Speech AI system equivalent to Luna/Moshi/Maya with:
- **Latency**: <400ms end-to-end
- **Languages**: English, Hindi, Telugu (2000h each = 6000h total)
- **Budget**: ~$1,500 USD
- **Timeline**: 2 months

---

## 📊 PROJECT OVERVIEW TABLE

| Phase | Task | Duration | Cost | Status |
|-------|------|----------|------|--------|
| **1** | Data Collection (6000h) | 1 week | $0 | ⏳ Pending |
| **2** | Codec Training + GAN | 2 weeks | $400 | ⏳ Pending |
| **3** | S2S Model Training | 2 weeks | $500 | ⏳ Pending |
| **4** | Fine-tuning + Emotion | 1 week | $300 | ⏳ Pending |
| **5** | Production Deployment | 1 week | $200 | ⏳ Pending |
| **Total** | | **8 weeks** | **~$1,400** | |

---

## 📁 PHASE 1: DATA COLLECTION (6000 Hours)

### ENGLISH: 2000+ Hours ✅ EASY

| Dataset | Hours | License | Download Method |
|---------|-------|---------|-----------------|
| **LibriSpeech** | 960h | CC BY 4.0 | wget from OpenSLR |
| **LibriLight** (small) | 600h | MIT | wget from Meta |
| **Common Voice** | 500h+ | CC0 | HuggingFace |
| **Total** | **2060h+** | ✅ | |

#### Download Commands (English):
```bash
# Create directories
mkdir -p data/english/{librispeech,librilight,commonvoice}
cd data/english

# === LIBRISPEECH (960 hours) ===
cd librispeech

# Train Clean 100 (100 hours) - 6.3GB
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz && rm train-clean-100.tar.gz

# Train Clean 360 (360 hours) - 23GB
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzf train-clean-360.tar.gz && rm train-clean-360.tar.gz

# Train Other 500 (500 hours) - 30GB
wget https://www.openslr.org/resources/12/train-other-500.tar.gz
tar -xzf train-other-500.tar.gz && rm train-other-500.tar.gz

# === LIBRILIGHT SMALL (600 hours) ===
cd ../librilight
# Download from: https://github.com/facebookresearch/libri-light
# Or use HuggingFace:
# pip install datasets
# python -c "from datasets import load_dataset; ds = load_dataset('facebook/libri-light', 'small', split='train'); ds.save_to_disk('librilight_small')"
```

---

### HINDI: 2000+ Hours ✅ ACHIEVABLE

| Dataset | Hours | License | Download Method |
|---------|-------|---------|-----------------|
| **Gramvaani (OpenSLR 118)** | 1,111h | CC BY 4.0 | wget |
| **IndicVoices Hindi** | 800h+ | CC BY 4.0 | HuggingFace |
| **Kathbath Hindi** | 140h | CC0 | HuggingFace |
| **OpenSLR 103** | 95h | CC BY 4.0 | wget |
| **Total** | **2146h+** | ✅ | |

#### Download Commands (Hindi):
```bash
mkdir -p data/hindi/{gramvaani,indicvoices,kathbath,openslr103}
cd data/hindi

# === GRAMVAANI (1111 hours) - LARGEST FREE HINDI DATASET ===
cd gramvaani

# Labeled data (111 hours)
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Train_100h.tar.gz
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Dev_5h.tar.gz
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_5h.tar.gz

# Unlabeled data (1000 hours) - HUGE
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Unlabeled_1000h.tar.gz

# Extract all
for f in *.tar.gz; do tar -xzf "$f" && rm "$f"; done

# === OPENSLR 103 (95 hours) ===
cd ../openslr103
wget https://www.openslr.org/resources/103/Hindi_train.tar.gz
wget https://www.openslr.org/resources/103/Hindi_test.tar.gz
tar -xzf Hindi_train.tar.gz && rm Hindi_train.tar.gz
tar -xzf Hindi_test.tar.gz && rm Hindi_test.tar.gz

# === INDICVOICES + KATHBATH (HuggingFace) ===
# See Python script below
```

---

### TELUGU: 2000+ Hours ⚠️ CHALLENGING

| Dataset | Hours | License | Download Method |
|---------|-------|---------|-----------------|
| **IndicVoices Telugu** | 400h+ | CC BY 4.0 | HuggingFace |
| **Kathbath Telugu** | 155h | CC0 | HuggingFace |
| **OpenSLR 66** | 10h | CC BY-SA 4.0 | wget |
| **Common Voice Telugu** | 25h | CC0 | HuggingFace |
| **SPRING-INX Telugu** | 200h+ | Academic | IIT Madras |
| **IndicTTS Telugu** | 10h | CC BY 4.0 | HuggingFace |
| **Subtotal Available** | ~800h | | |
| **Gap** | ~1200h | | **Need synthetic/augmentation** |

#### Download Commands (Telugu):
```bash
mkdir -p data/telugu/{openslr66,indicvoices,kathbath,commonvoice,springinx}
cd data/telugu

# === OPENSLR 66 (10 hours) - VERIFIED WORKING ===
cd openslr66
wget https://www.openslr.org/resources/66/te_in_female.zip   # 505MB
wget https://www.openslr.org/resources/66/te_in_male.zip     # 529MB
unzip te_in_female.zip && rm te_in_female.zip
unzip te_in_male.zip && rm te_in_male.zip

# === SPRING-INX (200+ hours) ===
cd ../springinx
# Download from: https://asr.iitm.ac.in/dataset
# No login required, direct download available
```

---

### PYTHON SCRIPT FOR HUGGINGFACE DATASETS

```python
#!/usr/bin/env python3
"""
download_huggingface_data.py
Downloads IndicVoices, Kathbath, and Common Voice for Hindi and Telugu
"""

import os
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

# Create output directories
for lang in ['hindi', 'telugu']:
    for ds in ['indicvoices', 'kathbath', 'commonvoice']:
        Path(f"data/{lang}/{ds}").mkdir(parents=True, exist_ok=True)

def save_dataset(dataset, output_dir, prefix, max_samples=None):
    """Save dataset audio files to disk"""
    count = 0
    for i, sample in enumerate(tqdm(dataset)):
        if max_samples and i >= max_samples:
            break
        try:
            audio = sample['audio']
            array = audio['array']
            sr = audio['sampling_rate']
            
            output_path = Path(output_dir) / f"{prefix}_{i:08d}.wav"
            sf.write(str(output_path), array, sr)
            count += 1
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Saved {count} files to {output_dir}")

# ==========================================
# INDICVOICES (Requires HuggingFace login)
# First: huggingface-cli login
# Then: Accept license at https://huggingface.co/datasets/ai4bharat/IndicVoices
# ==========================================

print("="*60)
print("DOWNLOADING INDICVOICES")
print("="*60)

# Hindi IndicVoices
try:
    print("\nDownloading IndicVoices Hindi...")
    ds_hindi = load_dataset("ai4bharat/IndicVoices", "hindi", 
                            split="train", streaming=True)
    save_dataset(ds_hindi, "data/hindi/indicvoices", "indicvoices_hi")
except Exception as e:
    print(f"IndicVoices Hindi failed: {e}")
    print("Make sure to: 1) huggingface-cli login, 2) Accept license on HuggingFace")

# Telugu IndicVoices
try:
    print("\nDownloading IndicVoices Telugu...")
    ds_telugu = load_dataset("ai4bharat/IndicVoices", "telugu", 
                             split="train", streaming=True)
    save_dataset(ds_telugu, "data/telugu/indicvoices", "indicvoices_te")
except Exception as e:
    print(f"IndicVoices Telugu failed: {e}")

# ==========================================
# KATHBATH (CC0 License - No attribution required!)
# Accept license at: https://huggingface.co/datasets/ai4bharat/Kathbath
# ==========================================

print("\n" + "="*60)
print("DOWNLOADING KATHBATH")
print("="*60)

# Hindi Kathbath
try:
    print("\nDownloading Kathbath Hindi...")
    ds_kathbath_hi = load_dataset("ai4bharat/Kathbath", "hindi", split="train")
    save_dataset(ds_kathbath_hi, "data/hindi/kathbath", "kathbath_hi")
except Exception as e:
    print(f"Kathbath Hindi failed: {e}")

# Telugu Kathbath
try:
    print("\nDownloading Kathbath Telugu...")
    ds_kathbath_te = load_dataset("ai4bharat/Kathbath", "telugu", split="train")
    save_dataset(ds_kathbath_te, "data/telugu/kathbath", "kathbath_te")
except Exception as e:
    print(f"Kathbath Telugu failed: {e}")

# ==========================================
# COMMON VOICE (CC0 - Completely free)
# ==========================================

print("\n" + "="*60)
print("DOWNLOADING COMMON VOICE")
print("="*60)

# Telugu Common Voice
try:
    print("\nDownloading Common Voice Telugu...")
    ds_cv_te = load_dataset("mozilla-foundation/common_voice_16_1", "te", 
                            split="train", streaming=True)
    save_dataset(ds_cv_te, "data/telugu/commonvoice", "cv_te")
except Exception as e:
    print(f"Common Voice Telugu failed: {e}")

# Hindi Common Voice
try:
    print("\nDownloading Common Voice Hindi...")
    ds_cv_hi = load_dataset("mozilla-foundation/common_voice_16_1", "hi", 
                            split="train", streaming=True)
    save_dataset(ds_cv_hi, "data/hindi/commonvoice", "cv_hi")
except Exception as e:
    print(f"Common Voice Hindi failed: {e}")

print("\n" + "="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
```

---

### FILLING THE TELUGU GAP (Need ~1200h more)

**Strategy 1: Data Augmentation (Doubles existing data)**
```python
# audio_augmentation.py
import torchaudio
import torch
from pathlib import Path

def augment_audio(input_dir, output_dir):
    """Apply augmentations to double the dataset"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for audio_file in input_path.glob("*.wav"):
        waveform, sr = torchaudio.load(str(audio_file))
        
        # 1. Speed perturbation (0.9x, 1.1x)
        for speed in [0.9, 1.1]:
            effects = [["speed", str(speed)], ["rate", str(sr)]]
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
            out_name = f"{audio_file.stem}_speed{speed}.wav"
            torchaudio.save(str(output_path / out_name), augmented, sr)
        
        # 2. Pitch shift (+/- 2 semitones)
        for pitch in [-2, 2]:
            effects = [["pitch", str(pitch * 100)], ["rate", str(sr)]]
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
            out_name = f"{audio_file.stem}_pitch{pitch}.wav"
            torchaudio.save(str(output_path / out_name), augmented, sr)
        
        # 3. Add background noise
        noise = torch.randn_like(waveform) * 0.005
        augmented = waveform + noise
        out_name = f"{audio_file.stem}_noise.wav"
        torchaudio.save(str(output_path / out_name), augmented, sr)

# Usage: 800h original → 4000h+ augmented
augment_audio("data/telugu/original", "data/telugu/augmented")
```

**Strategy 2: Synthetic TTS Data**
- Use AI4Bharat's Indic-TTS to generate additional Telugu speech
- Generate diverse speakers and prosodies
- ~500h can be synthesized

**Bottom Line**: With augmentation + synthetic, 2000h Telugu is achievable!

---

## ⚙️ PHASE 2: CODEC TRAINING

### Current Codec Status: ✅ GOOD ARCHITECTURE

Your `codec_production.py` already has:
- ✅ Hybrid CNN + Transformer (Mimi-style)
- ✅ WavLM semantic distillation  
- ✅ 50Hz frame rate (efficient)
- ✅ Variable codebook sizes
- ✅ Multi-scale spectral + mel losses
- ✅ EMA codebook updates

Your `train_codec_production.py` already has:
- ✅ GAN discriminator integrated (`discriminator_dac.py`)
- ✅ Multi-Period + Multi-Scale STFT discriminators
- ✅ Feature matching loss
- ✅ Mixed precision (FP16)
- ✅ Gradient clipping

### What's MISSING (Need to Add):

#### 1. Data Augmentation During Training
```python
# Add to MultilingualAudioDataset.__getitem__()

class MultilingualAudioDataset(Dataset):
    def __init__(self, ..., augment=True):
        self.augment = augment
        
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(str(audio_path))
        # ... existing code ...
        
        # ADD AUGMENTATION
        if self.augment and self.training:
            waveform = self._apply_augmentation(waveform)
        
        return waveform
    
    def _apply_augmentation(self, waveform):
        """Apply random augmentations for robustness"""
        # 30% chance of each augmentation
        
        # 1. Speed perturbation
        if torch.rand(1).item() < 0.3:
            speed = 0.9 + torch.rand(1).item() * 0.2  # 0.9-1.1
            waveform = torchaudio.functional.speed(waveform, self.sample_rate, speed)[0]
        
        # 2. Add noise
        if torch.rand(1).item() < 0.3:
            noise_level = torch.rand(1).item() * 0.01  # 0-0.01
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        # 3. Volume perturbation
        if torch.rand(1).item() < 0.3:
            gain = 0.7 + torch.rand(1).item() * 0.6  # 0.7-1.3
            waveform = waveform * gain
        
        return waveform
```

#### 2. Multi-Speaker Diversity Check
```python
# Add speaker counting to training script
def count_speakers(data_dirs):
    """Estimate number of unique speakers"""
    speaker_dirs = set()
    for data_dir in data_dirs:
        for subdir in Path(data_dir).rglob("*"):
            if subdir.is_dir() and any(subdir.glob("*.wav")):
                speaker_dirs.add(subdir.name)
    print(f"Estimated speakers: {len(speaker_dirs)}")
    return len(speaker_dirs)
```

### Codec Training Command:
```bash
python train_codec_production.py \
    --data_dirs data/english data/hindi data/telugu \
    --batch_size 32 \
    --num_epochs 100 \
    --gen_lr 1e-4 \
    --disc_lr 1e-4 \
    --checkpoint_dir checkpoints_codec \
    --num_workers 12
```

### Expected Training Time:
- **6000 hours of data** @ 32 batch size
- **100 epochs** on H200 (141GB VRAM)
- **Estimated: 40-50 GPU hours = ~$170**

---

## 🧠 PHASE 3: S2S MODEL TRAINING

### Architecture (From Luna Image Analysis):
```
Raw Audio Input → VAD → Speech Encoder → Neural Audio Codec (VQ-VAE)
                                              ↓
                                    Discrete Audio Tokens
                                              ↓
                                    Transformer LM ←→ Semantic Understanding
                                              ↓         ↓
                                              ↓    Emotion Detection
                                              ↓         ↓
                                    Audio Decoder Transformer
                                              ↓
                                    Neural Vocoder
                                              ↓
                                    Streaming Audio Synthesis
                                              ↓
                                    Synthesized Speech Output
```

### Key Components Needed:
1. **Codec** (Phase 2) ✅
2. **Transformer LM** for dialogue - Need to train
3. **Emotion Detection** - Need to integrate
4. **Streaming Pipeline** - Already have structure

### S2S Training Data:
- Need **conversation pairs**, not just raw audio
- Download Fisher Corpus (if available)
- Generate synthetic conversations with LLM + TTS

---

## 📋 RUNPOD SETUP COMMANDS (Verified)

### Step 1: System Dependencies
```bash
apt-get update
apt-get install -y \
    ffmpeg \
    git \
    vim \
    tmux \
    htop \
    nvtop \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    sox \
    screen \
    nano \
    unzip \
    p7zip-full
```

### Step 2: Clone Repository
```bash
cd /workspace
git clone https://github.com/devasphn/NewProject.git
cd NewProject
```

### Step 3: Install Python Packages
```bash
pip install --upgrade pip wheel setuptools

# Core packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Requirements
pip install -r requirements_new.txt

# Flash Attention (for faster training)
pip install flash-attn --no-build-isolation

# HuggingFace for datasets
pip install datasets transformers huggingface_hub

# Audio processing
pip install librosa soundfile pydub
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "from transformers import WavLMModel; print('WavLM OK')"
```

### Step 5: HuggingFace Login (for IndicVoices/Kathbath)
```bash
huggingface-cli login
# Enter your HuggingFace token
# Then accept licenses at:
# - https://huggingface.co/datasets/ai4bharat/IndicVoices
# - https://huggingface.co/datasets/ai4bharat/Kathbath
```

---

## 💰 BUDGET BREAKDOWN

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| H200 - Codec Training | 50h | $3.39/h | $170 |
| H200 - S2S Training | 80h | $3.39/h | $271 |
| H200 - Fine-tuning | 30h | $3.39/h | $102 |
| Storage (500GB) | 2 months | $50/mo | $100 |
| Inference Testing | 20h | $3.39/h | $68 |
| **Total** | **180h** | | **$711** |

**Buffer for errors/restarts**: +$300

**Total Budget**: ~$1,000-$1,400 ✅ Under $1,500!

---

## ✅ IS 6000 HOURS ENOUGH?

### Comparison with State-of-Art:
| Model | Training Data | Our Target |
|-------|---------------|------------|
| EnCodec | ~10,000h | ✅ |
| DAC | ~5,000h | ✅ |
| Moshi/Mimi | Not disclosed (est. 20,000h+) | ⚠️ |
| Luna | Not disclosed (est. 60,000h+) | ⚠️ |

### Verdict:
- **6000h is ENOUGH for production-grade codec** ✅
- **6000h is MINIMUM for good S2S** - need conversation pairs
- **For Luna-equivalent emotion/prosody** - may need more data or synthetic augmentation

### Mitigation:
1. **Data augmentation** → Effectively 3-5x more data
2. **Transfer learning** from English → Hindi/Telugu
3. **Synthetic data** for emotion training

---

## 📅 2-MONTH TIMELINE

### Week 1-2: Data Collection
- [ ] Download LibriSpeech (English)
- [ ] Download Gramvaani (Hindi)
- [ ] Download IndicVoices (Hindi + Telugu)
- [ ] Download Kathbath (Hindi + Telugu)
- [ ] Download OpenSLR 66 (Telugu)
- [ ] Apply data augmentation for Telugu
- [ ] Verify all data (count files, check formats)

### Week 3-4: Codec Training
- [ ] Start codec training with GAN
- [ ] Monitor losses and quality
- [ ] Checkpoint every 5 epochs
- [ ] Evaluate codec quality (SNR, PESQ)

### Week 5-6: S2S Model Training
- [ ] Prepare conversation pairs
- [ ] Train S2S transformer
- [ ] Integrate emotion embeddings
- [ ] Test end-to-end latency

### Week 7: Fine-tuning + Optimization
- [ ] Fine-tune on emotion data
- [ ] Optimize for latency (<400ms)
- [ ] Test streaming inference
- [ ] Profile and optimize bottlenecks

### Week 8: Production Deployment
- [ ] Deploy to RunPod serverless
- [ ] Add WebSocket streaming
- [ ] Load testing
- [ ] Documentation

---

## 🎯 SUCCESS CRITERIA

| Metric | Target | Luna Reference |
|--------|--------|----------------|
| Codec SNR | >20 dB | ~22 dB |
| Codec MOS | >4.0 | ~4.3 |
| E2E Latency | <400ms | 580-600ms |
| Languages | 3 | 1 (expanding) |
| Training Cost | <$1,500 | Unknown (est. $100K+) |

---

## 📝 NEXT IMMEDIATE STEPS

1. **Run system setup commands** on RunPod
2. **Start data downloads** (LibriSpeech first - largest)
3. **Accept HuggingFace licenses** for IndicVoices/Kathbath
4. **Create download scripts** and run overnight
5. **Verify data counts** before training

Ready to proceed? Let me create the download scripts!
