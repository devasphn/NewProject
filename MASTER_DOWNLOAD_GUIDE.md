# 🚀 MASTER DATA DOWNLOAD GUIDE - 10,000+ HOURS

## 📊 Available Data Summary

| Tier | Source | Hours | Languages | License |
|------|--------|-------|-----------|---------|
| **1** | OpenSLR (Direct) | ~400h | 9 Indian langs | CC-BY-4.0 |
| **2** | LibriSpeech | ~1,000h | English | CC-BY-4.0 |
| **3** | MLS | ~50,000h | 8 languages | CC-BY-4.0 |
| **4** | IndicVoices | ~19,550h | 22 Indian langs | CC-BY-4.0 |
| **5** | Kathbath | ~1,684h | 12 Indian langs | **CC0** |
| | **TOTAL** | **~72,600+h** | | |

---

## 🔧 STEP 1: Fix RunPod Environment

```bash
# SSH into RunPod and run these FIRST:
apt-get update
apt-get install -y unzip wget aria2 p7zip-full

# Go to your project
cd /workspace/NewProject
```

---

## 🔧 STEP 2: Direct Downloads (OpenSLR + LibriSpeech)

### Quick Start - Get ~1,500 hours immediately:

```bash
# Create directories
mkdir -p data/{telugu,tamil,kannada,malayalam,marathi,gujarati,bengali,hindi,odia,english}

# ═══════════════════════════════════════════════════════════════
# TELUGU (~10 hours) - OpenSLR 66 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/telugu https://www.openslr.org/resources/66/te_in_female.zip
wget -c -P data/telugu https://www.openslr.org/resources/66/te_in_male.zip
cd data/telugu && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# TAMIL (~15 hours) - OpenSLR 65 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/tamil https://www.openslr.org/resources/65/ta_in_female.zip
wget -c -P data/tamil https://www.openslr.org/resources/65/ta_in_male.zip
cd data/tamil && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# KANNADA (~10 hours) - OpenSLR 79 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/kannada https://www.openslr.org/resources/79/kn_in_female.zip
wget -c -P data/kannada https://www.openslr.org/resources/79/kn_in_male.zip
cd data/kannada && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# MALAYALAM (~10 hours) - OpenSLR 63 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/malayalam https://www.openslr.org/resources/63/ml_in_female.zip
wget -c -P data/malayalam https://www.openslr.org/resources/63/ml_in_male.zip
cd data/malayalam && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# MARATHI (~10 hours) - OpenSLR 64 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/marathi https://www.openslr.org/resources/64/mr_in_female.zip
wget -c -P data/marathi https://www.openslr.org/resources/64/mr_in_male.zip
cd data/marathi && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# GUJARATI (~10 hours) - OpenSLR 78 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/gujarati https://www.openslr.org/resources/78/gu_in_female.zip
wget -c -P data/gujarati https://www.openslr.org/resources/78/gu_in_male.zip
cd data/gujarati && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# BENGALI (~10 hours) - OpenSLR 53 ✅ VERIFIED  
# ═══════════════════════════════════════════════════════════════
wget -c -P data/bengali https://www.openslr.org/resources/53/bn_in_female.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/bn_in_male.zip
cd data/bengali && unzip -o "*.zip" && rm *.zip && cd ../..

# ═══════════════════════════════════════════════════════════════
# HINDI (~95 hours) - MUCS OpenSLR 103 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/hindi https://www.openslr.org/resources/103/Hindi_train.tar.gz
wget -c -P data/hindi https://www.openslr.org/resources/103/Hindi_test.tar.gz
cd data/hindi && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# ═══════════════════════════════════════════════════════════════
# ODIA (~95 hours) - MUCS OpenSLR 103 ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/odia https://www.openslr.org/resources/103/Odia_train.tar.gz
wget -c -P data/odia https://www.openslr.org/resources/103/Odia_test.tar.gz
cd data/odia && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# ═══════════════════════════════════════════════════════════════
# ENGLISH - LibriSpeech (~1,000 hours) ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/english https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c -P data/english https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c -P data/english https://www.openslr.org/resources/12/train-other-500.tar.gz
cd data/english && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# ═══════════════════════════════════════════════════════════════
# LJSpeech (24 hours - High Quality) ✅ VERIFIED
# ═══════════════════════════════════════════════════════════════
wget -c -P data/english https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
cd data/english && tar -xjf *.tar.bz2 && rm *.tar.bz2 && cd ../..
```

**Total from direct downloads: ~1,300 hours**

---

## 🔧 STEP 3: MLS - Massive Multilingual Data (~5,000+ hours)

```bash
mkdir -p data/{german,french,spanish,dutch,italian}

# German (~2,000 hours) - OPUS compressed
wget -c -P data/german https://dl.fbaipublicfiles.com/mls/mls_german_opus.tar.gz
cd data/german && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# French (~1,100 hours)
wget -c -P data/french https://dl.fbaipublicfiles.com/mls/mls_french_opus.tar.gz
cd data/french && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# Spanish (~900 hours)
wget -c -P data/spanish https://dl.fbaipublicfiles.com/mls/mls_spanish_opus.tar.gz
cd data/spanish && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# Dutch (~1,500 hours)
wget -c -P data/dutch https://dl.fbaipublicfiles.com/mls/mls_dutch_opus.tar.gz
cd data/dutch && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..

# Italian (~200 hours)
wget -c -P data/italian https://dl.fbaipublicfiles.com/mls/mls_italian_opus.tar.gz
cd data/italian && tar -xzf *.tar.gz && rm *.tar.gz && cd ../..
```

**Total from MLS: ~5,700 hours**

---

## 🔧 STEP 4: HuggingFace Datasets (Kathbath + IndicVoices)

### 4.1 Setup HuggingFace Access

```bash
# Install packages
pip install datasets huggingface_hub soundfile tqdm

# Login to HuggingFace (required for gated datasets)
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### 4.2 Accept Dataset Licenses

**IMPORTANT: You MUST accept these licenses on HuggingFace website:**

1. **Kathbath** (1,684h): https://huggingface.co/datasets/ai4bharat/Kathbath
   - Click "Agree and access repository"
   
2. **IndicVoices** (19,550h): https://huggingface.co/datasets/ai4bharat/IndicVoices
   - Click "Agree and access repository"

### 4.3 Download Kathbath (~1,684 hours)

```bash
python download_kathbath.py
```

### 4.4 Download IndicVoices (~19,550 hours)

```bash
python download_indicvoices.py
```

---

## 📊 FINAL DATA SUMMARY

| Category | Hours | Storage |
|----------|-------|---------|
| Indian Languages (OpenSLR) | ~350h | ~40GB |
| English (LibriSpeech + LJ) | ~1,024h | ~100GB |
| European (MLS) | ~5,700h | ~150GB |
| Kathbath (HuggingFace) | ~1,684h | ~200GB |
| IndicVoices (HuggingFace) | ~19,550h | ~2TB+ |
| **TOTAL** | **~28,300h** | **~2.5TB** |

---

## 🎯 RECOMMENDED STRATEGY

### For 10,000+ hours (Production-Grade):

```bash
# Day 1: Direct downloads (~1,300h) - 2-3 hours
# Run all wget commands above

# Day 1-2: MLS European (~5,700h) - 4-6 hours
# Download German, French, Spanish

# Day 2-3: Kathbath (~1,684h) - 3-4 hours
python download_kathbath.py

# Day 3-5: IndicVoices (selective ~2,000h) - 6-8 hours
# Edit download_indicvoices.py to limit samples
python download_indicvoices.py
```

**Total: ~10,700 hours in 3-5 days!**

---

## 🏋️ START TRAINING

Once you have 1,000+ hours downloaded:

```bash
# Test codec
python codec_production.py

# Start training
python train_codec_production.py \
    --data_dirs data/telugu data/tamil data/hindi data/english data/german \
    --batch_size 16 \
    --num_epochs 200 \
    --checkpoint_dir checkpoints_production
```

---

## ⚠️ STORAGE REQUIREMENTS

| Data Amount | Disk Space Needed |
|-------------|-------------------|
| 1,000 hours | ~100 GB |
| 5,000 hours | ~500 GB |
| 10,000 hours | ~1 TB |
| 20,000+ hours | ~2 TB+ |

**Recommendation: Use RunPod with 1TB+ volume disk for 10,000+ hours**

---

## ❌ INCORRECT URLs (DO NOT USE)

These URLs from previous guides are **WRONG**:

```bash
# ❌ WRONG - Telugu.zip doesn't exist at SLR103
wget https://www.openslr.org/resources/103/Telugu.zip

# ✅ CORRECT - SLR103 has Hindi, Marathi, Odia only
wget https://www.openslr.org/resources/103/Hindi_train.tar.gz
```

---

## 🔍 VERIFY DOWNLOADS

```bash
# Count audio files
find data -name "*.wav" | wc -l
find data -name "*.flac" | wc -l
find data -name "*.mp3" | wc -l

# Check disk usage
du -sh data/*
```
