# Data Collection: 1000+ Hours Per Language

## Target Languages: English, Hindi, Telugu

---

## ENGLISH: 1000+ Hours (EASY - Multiple Sources)

### Option 1: LibriSpeech (1,000 hours) - RECOMMENDED
```bash
# Download all LibriSpeech (960 hours clean + noisy)
mkdir -p data/english/librispeech
cd data/english/librispeech

# Clean data (460 hours)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz

# Other data (500 hours)  
wget https://www.openslr.org/resources/12/train-other-500.tar.gz

# Extract
for f in *.tar.gz; do tar -xzf "$f"; done
```
**Total: ~960 hours**

### Option 2: LibriLight (60,000 hours) - For scaling
```bash
# Unlabeled but huge dataset
# Download from: https://github.com/facebookresearch/libri-light
# Subsets: small (577h), medium (5,193h), large (53,924h)
```

### Option 3: LibriHeavy (50,000 hours)
```bash
# With punctuation and casing
# Download from: https://github.com/k2-fsa/libriheavy
```

**English Status: ✅ EASY to get 1000+ hours**

---

## HINDI: 1000+ Hours (ACHIEVABLE)

### Source 1: Gramvaani Hindi (1,111 hours) - OpenSLR 118
```bash
mkdir -p data/hindi/gramvaani
cd data/hindi/gramvaani

# Download labeled data (111 hours)
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Train_100h.tar.gz
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Dev_5h.tar.gz
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_5h.tar.gz

# Download unlabeled data (1000 hours)
wget https://asr.iitm.ac.in/Gramvaani/NEW/GV_Unlabeled_1000h.tar.gz

tar -xzf *.tar.gz
```
**Total: ~1,111 hours**

### Source 2: IndicVoices Hindi (800+ hours)
```python
from datasets import load_dataset

# Requires HuggingFace login & license acceptance
# Accept at: https://huggingface.co/datasets/ai4bharat/IndicVoices

dataset = load_dataset("ai4bharat/IndicVoices", "hindi", split="train", streaming=True)
```
**Estimated: 800-1000 hours**

### Source 3: Kathbath Hindi (~140 hours)
```python
from datasets import load_dataset

# Accept at: https://huggingface.co/datasets/ai4bharat/Kathbath
dataset = load_dataset("ai4bharat/Kathbath", "hindi", split="train")
```
**Total: ~140 hours**

### Source 4: OpenSLR 103 MUCS Hindi (95 hours)
```bash
mkdir -p data/hindi/openslr103
cd data/hindi/openslr103

wget https://www.openslr.org/resources/103/Hindi_train.zip
wget https://www.openslr.org/resources/103/Hindi_test.zip

unzip *.zip
```
**Total: ~95 hours**

### Source 5: Common Voice Hindi (~50 hours)
```python
from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "hi", split="train")
```

**Hindi Total Available: ~2,000+ hours ✅**

---

## TELUGU: 1000+ Hours (CHALLENGING)

### Current Available Sources

| Source | Hours | License | Download |
|--------|-------|---------|----------|
| IndicVoices Telugu | ~300h | CC-BY-4.0 | HuggingFace |
| Kathbath Telugu | ~140h | CC0 | HuggingFace |
| OpenSLR 66 | ~10h | CC-BY-SA | wget |
| Common Voice Telugu | ~20h | CC0 | HuggingFace |
| **Total Available** | **~470h** | | |

### Source 1: IndicVoices Telugu (300+ hours)
```python
from datasets import load_dataset
import soundfile as sf
from pathlib import Path
import gc

output_dir = Path("data/telugu/indicvoices")
output_dir.mkdir(parents=True, exist_ok=True)

# Accept license at: https://huggingface.co/datasets/ai4bharat/IndicVoices
dataset = load_dataset("ai4bharat/IndicVoices", "telugu", split="train", streaming=True)

count = 0
for sample in dataset:
    try:
        audio = sample['audio']
        array = audio['array']
        sr = audio['sampling_rate']
        
        sf.write(str(output_dir / f"indicvoices_te_{count:08d}.wav"), array, sr)
        count += 1
        
        if count % 1000 == 0:
            print(f"Downloaded {count} files")
            gc.collect()
    except:
        continue
```

### Source 2: Kathbath Telugu (140 hours)
```python
from datasets import load_dataset

# Accept at: https://huggingface.co/datasets/ai4bharat/Kathbath
dataset = load_dataset("ai4bharat/Kathbath", "telugu", split="train")
```

### Source 3: OpenSLR 66 (10 hours)
```bash
mkdir -p data/telugu/openslr66
cd data/telugu/openslr66

wget https://www.openslr.org/resources/66/te_in_female.zip
wget https://www.openslr.org/resources/66/te_in_male.zip

unzip *.zip
```

### Source 4: Common Voice Telugu (20 hours)
```python
from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "te", split="train")
```

### Gap Analysis for Telugu
```
Required: 1000 hours
Available: ~470 hours
Gap: ~530 hours
```

### Options to Fill Telugu Gap:

1. **IndicVoices-R (Read speech)** - Additional Telugu data
   ```python
   dataset = load_dataset("ai4bharat/indicvoices_r", "telugu")
   ```

2. **YouTube Scraping** (risky, quality varies)
   - Telugu news channels
   - Telugu podcasts
   - Educational content

3. **Synthetic Data Generation**
   - Use existing TTS to generate more Telugu audio
   - Not ideal but can supplement

4. **Partner with AI4Bharat** for additional data

**Telugu Status: ⚠️ ~470h available, need creative solutions for 1000h**

---

## COMBINED DOWNLOAD SCRIPT

```bash
#!/bin/bash
# download_production_data.sh

set -e

echo "============================================"
echo "PRODUCTION DATA DOWNLOAD (3000+ hours)"
echo "============================================"

# ============ ENGLISH (1000h) ============
echo ""
echo "📥 ENGLISH: LibriSpeech (960 hours)"
mkdir -p data/english/librispeech
cd data/english/librispeech

wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c https://www.openslr.org/resources/12/train-other-500.tar.gz

for f in *.tar.gz; do tar -xzf "$f" && rm "$f"; done
cd ../../..

# ============ HINDI (1100h) ============
echo ""
echo "📥 HINDI: Gramvaani (1111 hours)"
mkdir -p data/hindi/gramvaani
cd data/hindi/gramvaani

wget -c https://asr.iitm.ac.in/Gramvaani/NEW/GV_Train_100h.tar.gz
wget -c https://asr.iitm.ac.in/Gramvaani/NEW/GV_Unlabeled_1000h.tar.gz

for f in *.tar.gz; do tar -xzf "$f" && rm "$f"; done
cd ../../..

# ============ TELUGU (OpenSLR only - use Python for HF) ============
echo ""
echo "📥 TELUGU: OpenSLR 66 (10 hours)"
mkdir -p data/telugu/openslr66
cd data/telugu/openslr66

wget -c https://www.openslr.org/resources/66/te_in_female.zip
wget -c https://www.openslr.org/resources/66/te_in_male.zip

unzip -o *.zip && rm *.zip
cd ../../..

echo ""
echo "============================================"
echo "✅ OpenSLR downloads complete!"
echo ""
echo "Next: Run Python scripts for HuggingFace data:"
echo "  python download_indicvoices.py --lang telugu"
echo "  python download_indicvoices.py --lang hindi"
echo "  python download_kathbath.py --lang telugu"
echo "============================================"
```

---

## Storage Requirements

| Language | Hours | Est. Size (16kHz mono) |
|----------|-------|------------------------|
| English | 1000h | ~120 GB |
| Hindi | 1100h | ~130 GB |
| Telugu | 500h | ~60 GB |
| **Total** | **2600h** | **~310 GB** |

Add 50% buffer: **~500 GB recommended**

---

## Is 3000 Hours Enough for Production?

### Comparison with State-of-Art
| Model | Training Data | Quality |
|-------|---------------|---------|
| EnCodec | ~10,000h | Excellent |
| DAC | ~5,000h | Excellent |
| Mimi/Moshi | Not disclosed | Excellent |
| SoundStream | ~4,000h | Good |

### Research Findings
- **Codec training**: 1000-2000h per language is sufficient
- **S2S training**: Requires conversational pairs, not raw hours
- **Quality threshold**: >500h gives diminishing returns for codecs

### Verdict: **YES, 3000h (1000h x 3 languages) is sufficient for production-grade codec**

For S2S model, you need **conversational pairs**, not just raw audio!

---

## Action Items

1. [ ] Accept HuggingFace licenses:
   - https://huggingface.co/datasets/ai4bharat/IndicVoices
   - https://huggingface.co/datasets/ai4bharat/Kathbath

2. [ ] Download English (LibriSpeech) - 960h

3. [ ] Download Hindi (Gramvaani + IndicVoices) - 1100h+

4. [ ] Download Telugu (all sources) - ~470h

5. [ ] Consider data augmentation for Telugu gap
