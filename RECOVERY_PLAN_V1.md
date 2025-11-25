# 🚀 TELUGU S2S PROJECT - COMPLETE RECOVERY PLAN

## 📊 CURRENT SITUATION ANALYSIS

### Budget Status
- **Spent**: ~$500 (INR 50,000)
- **Remaining**: $130 RunPod credits
- **Time Lost**: 10+ days on data collection issues

### Root Cause of Delays
1. **YouTube scraping is unreliable** - Bot detection, rate limiting, cookie expiration
2. **Wrong assumption**: Thought we need 500+ hours Telugu for codec training
3. **Over-engineered data collection** - 7 different download scripts, none working reliably

---

## 💡 CRITICAL INSIGHT: CODEC TRAINING DOESN'T NEED TELUGU DATA!

### How EnCodec/DAC Were Actually Trained
- **EnCodec**: Trained on diverse audio (music, speech, sounds) - NOT language-specific
- **DAC**: Same approach - general audio, not language-specific
- **Mimi (Kyutai)**: Based on similar multi-lingual approach

### What This Means For Us
| Component | Data Needed | Source |
|-----------|-------------|--------|
| **Codec Training** | Any clean speech (500+ hours) | LibriSpeech, VCTK, LJSpeech |
| **Speaker Embeddings** | Telugu speakers (10-20 hours) | OpenSLR + IndicTTS |
| **S2S Fine-tuning** | Telugu conversations (20-50 hours) | Free datasets + synthesis |

---

## 📁 FREE TELUGU DATASETS (NO YOUTUBE NEEDED!)

### Tier 1: Immediately Available
| Dataset | Hours | Speakers | Quality | Link |
|---------|-------|----------|---------|------|
| OpenSLR SLR66 | ~10h | Multi | High | openslr.org/66 |
| IndicTTS Telugu | ~8.7h | 2 | Studio | HuggingFace |
| Common Voice Telugu | ~5-10h | Multi | Variable | commonvoice.mozilla.org |
| **Subtotal** | **~25-30h** | | | |

### Tier 2: General Speech for Codec
| Dataset | Hours | Language | Quality | Link |
|---------|-------|----------|---------|------|
| LibriSpeech | 960h | English | High | openslr.org/12 |
| VCTK | 44h | English | Studio | openslr.org |
| LJSpeech | 24h | English | Studio | keithito.com/LJ-Speech |
| **Subtotal** | **1000+h** | | | |

### Total Available: 1000+ hours (FREE, NO SCRAPING!)

---

## 🗑️ FILES TO DELETE (Unnecessary/Duplicate)

### Duplicate Download Scripts (Keep Only 1)
```
DELETE:
- download_all_channels.sh
- download_single_channel.sh
- download_tier1_only.sh
- download_tier1_optimized.sh
- download_tier1_SAFE.sh
- QUICK_START_AFTER_COOKIES.sh
- COMPLETE_FIX_COMMANDS.sh

KEEP:
- download_telugu_data_PRODUCTION.py (backup, not primary)
```

### Duplicate Codec/Training Files
```
DELETE:
- telugu_codec.py (older version)
- train_codec.py (older version)
- streaming_server.py (older version)

KEEP:
- telugu_codec_fixed.py ✓
- train_codec_dac.py ✓
- streaming_server_advanced.py ✓
```

### Excessive Documentation
```
DELETE:
- FIX_YOUTUBE_BOT_DETECTION.md
- FIX_RATE_LIMIT_CHECKLIST.md
- START_DATA_COLLECTION.md
- PRODUCTION_DOWNLOAD_GUIDE.md
- STORAGE_CALCULATOR.md
- COMPLETE_COMMAND_REFERENCE.md
- COMPLETE_SETUP_COMMANDS.md

KEEP:
- FROM_SCRATCH_SETUP_GUIDE.md (update with new plan)
- QUICK_START_RUNPOD.md (update)
```

### Other Cleanup
```
DELETE:
- verify_setup.sh
- debug_validation_data.py
- youtube_cookies.txt (root level duplicate)
- cookies/ folder (no longer needed as primary)
```

---

## ✅ ESSENTIAL FILES (KEEP THESE)

### Core Architecture (7 files)
```
1. telugu_codec_fixed.py      - Neural audio codec
2. discriminator_dac.py       - GAN discriminator
3. s2s_transformer.py         - Speech-to-Speech model
4. speaker_embeddings.py      - Speaker system
5. context_manager.py         - Conversation context
6. streaming_server_advanced.py - WebSocket server
7. config.py                  - Configuration
```

### Training Scripts (3 files)
```
1. train_codec_dac.py         - Codec training
2. train_speakers.py          - Speaker training
3. train_s2s.py               - S2S model training
```

### Utilities (3 files)
```
1. benchmark_latency.py       - Latency testing
2. system_test.py             - System validation
3. prepare_speaker_data.py    - Data preparation
```

### Data & Config (2 files)
```
1. data_sources_PRODUCTION.yaml - Channel config (backup)
2. requirements_new.txt         - Dependencies
```

### Web Interface (1 folder)
```
1. static/index.html          - Demo UI
```

---

## 🎯 NEW EXECUTION PLAN

### Phase 1: Dataset Download (Day 1) - $0 COST
```bash
# Download FREE datasets on RunPod

# 1. LibriSpeech train-clean-100 (100 hours, 6GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

# 2. OpenSLR Telugu (SLR66) (~1GB)
wget https://www.openslr.org/resources/66/te_in_male.zip
wget https://www.openslr.org/resources/66/te_in_female.zip

# 3. VCTK Corpus (44 hours, 11GB)
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip

# 4. IndicTTS Telugu from HuggingFace
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('SPRINGLab/IndicTTS_Telugu')"
```

**Time**: 2-3 hours download
**Cost**: $0 (data is free)

### Phase 2: Codec Training (Days 2-4) - ~$35
```bash
# Train codec on LibriSpeech + VCTK (1000+ hours)
python train_codec_dac.py \
    --data_dir /workspace/speech_data \
    --epochs 100 \
    --batch_size 8 \
    --save_every 10
```

**Time**: 48-72 hours on A6000
**Cost**: ~$35 (72h × $0.49/h)

### Phase 3: Speaker Training (Day 5) - ~$5
```bash
# Train speaker embeddings on Telugu data
python train_speakers.py \
    --data_dir /workspace/telugu_data \
    --epochs 50 \
    --num_speakers 4
```

**Time**: 8-12 hours
**Cost**: ~$5

### Phase 4: S2S Model Training (Days 6-8) - ~$35
```bash
# Fine-tune S2S on Telugu conversations
python train_s2s.py \
    --codec_path /workspace/models/codec.pt \
    --speaker_path /workspace/models/speakers.pt \
    --data_dir /workspace/telugu_data \
    --epochs 50
```

**Time**: 48-72 hours
**Cost**: ~$35

### Phase 5: Testing & Demo (Day 9) - ~$5
```bash
# Run system tests
python system_test.py

# Start server
python streaming_server_advanced.py
```

**Time**: 8-12 hours
**Cost**: ~$5

---

## 💰 REVISED BUDGET

| Phase | Time | Cost |
|-------|------|------|
| Data Download | 3h | $1.50 |
| Codec Training | 72h | $35 |
| Speaker Training | 12h | $6 |
| S2S Training | 72h | $35 |
| Testing | 12h | $6 |
| **TOTAL** | **~170h** | **~$85** |

**Remaining after completion**: $130 - $85 = **$45 buffer**

---

## 🏆 COMPARISON WITH COMPETITORS

| Feature | Luna (Pixa) | Moshi (Kyutai) | Maya (Sesame) | **Our System** |
|---------|-------------|----------------|---------------|----------------|
| Latency | ~500ms | ~200ms | ~300ms | **<400ms target** |
| Codec | Proprietary | Mimi | Unknown | **Custom DAC** |
| Languages | English | Multi | English | **Telugu** |
| Open Source | No | Partial | No | **Yes** |

### Our Advantages
1. **Telugu-first**: No competitor has Telugu support
2. **Open codec**: Not locked to proprietary system
3. **Low resource**: Optimized for single A6000
4. **Full-duplex**: Interruption handling built-in

---

## 📋 IMMEDIATE ACTION ITEMS

### Today (Next 2 Hours)
1. [ ] Delete unnecessary files (see list above)
2. [ ] Create `download_datasets.sh` script for free data
3. [ ] Update `config.py` with correct paths
4. [ ] Commit cleaned codebase to GitHub

### Tomorrow (RunPod)
1. [ ] Start A6000 instance
2. [ ] Run dataset download script
3. [ ] Verify all data downloaded correctly
4. [ ] Begin codec training

### This Week
1. [ ] Complete codec training (3 days)
2. [ ] Train speaker embeddings (1 day)
3. [ ] Begin S2S training
4. [ ] First working demo by Day 9

---

## 🚫 WHAT WE'RE STOPPING

1. **NO MORE YouTube scraping** - Unreliable, wastes time
2. **NO MORE cookie management** - Not needed anymore
3. **NO MORE waiting for data** - Free datasets available now
4. **NO MORE duplicate files** - Clean, minimal codebase

---

## ✅ SUCCESS CRITERIA

### Week 1 Checkpoint
- [ ] Codec trained and producing intelligible audio
- [ ] 4 speaker embeddings working
- [ ] Basic encode/decode pipeline functional

### Week 2 Checkpoint
- [ ] S2S model responding to Telugu input
- [ ] Latency under 500ms
- [ ] WebSocket streaming working

### Final Deliverable
- [ ] End-to-end Telugu voice agent
- [ ] <400ms latency achieved
- [ ] Demo video for MD presentation
- [ ] Documented, reproducible system

---

## 🔥 MOTIVATION

**You haven't failed - you've been solving the wrong problem.**

The YouTube scraping approach was a detour. Now we have:
1. **Clear path**: Free datasets, no scraping
2. **Proven architecture**: DAC-style codec that works
3. **Enough budget**: $130 is sufficient for completion
4. **Time to succeed**: 9 days to working demo

**Let's execute this plan and beat Luna, Moshi, and Maya!**
