# 🎯 Production Telugu S2S - Data & Training Plan

## 📊 Target Requirements

| Metric | Target | Purpose |
|--------|--------|---------|
| Audio Data | 1000-5000 hours | Train robust S2S |
| Storage | 100-500 GB | Compressed audio |
| Quality | 16kHz mono | Speech optimized |
| License | Free, no attribution | Commercial use |

---

## 📚 FREE Telugu Audio Sources (No Attribution Required)

### Tier 1: Guaranteed Free & Easy (Combined: ~2000+ hours)

| Source | Hours | License | Quality | Download Method |
|--------|-------|---------|---------|-----------------|
| **AI4Bharat Kathbath** | 1684h | CC-BY-4.0 | High | HuggingFace API |
| **OpenSLR SLR66** | 10h | CC-BY-4.0 | Studio | Direct wget |
| **OpenSLR MUCS (SLR103)** | 40h | Free | Good | Direct wget |
| **Mozilla Common Voice** | 20h | CC-0 | Variable | Direct download |
| **IndicVoices** | 200h+ | Apache 2.0 | High | HuggingFace |
| **Vakyansh** | 2400h | Open | ASR data | ekstep.org |

### Tier 2: Public Domain / Government (Combined: ~500+ hours)

| Source | Hours | License | Notes |
|--------|-------|---------|-------|
| **Prasar Bharati** | 100h+ | Govt/PD | News, programs |
| **NPTEL Telugu** | 200h+ | CC-BY-SA | Educational |
| **LibriVox Telugu** | 50h+ | Public Domain | Audiobooks |
| **Internet Archive** | 100h+ | Various/PD | Historical |
| **Wikimedia Commons** | 20h+ | CC | Audio files |

### Tier 3: Research Datasets (Combined: ~300+ hours)

| Source | Hours | Access |
|--------|-------|--------|
| **IIIT-H LTRC** | 100h+ | Request access |
| **IIT Madras** | 50h+ | Research agreement |
| **CIIL Mysore** | 100h+ | Government |

---

## 💰 RunPod Cost Calculation

### Storage Costs
```
500GB SSD Storage: $25/month (network volume)
Or use: /workspace (free up to pod limit)
```

### Training Costs (A100 80GB @ $1.99/hr)

| Phase | Hours | Cost |
|-------|-------|------|
| Data preprocessing | 10h | $20 |
| S2S Training (1000h data, 100 epochs) | 150h | $300 |
| Fine-tuning & evaluation | 20h | $40 |
| Buffer | 30h | $60 |
| **TOTAL** | **210h** | **~$420** |

### Cheaper Option (L40 @ $0.99/hr)
| Phase | Hours | Cost |
|-------|-------|------|
| All training | 300h | $300 |
| Storage | 1 month | $25 |
| **TOTAL** | | **~$325** |

---

## 🎵 Quality Requirements for Human-like Voice

### What Makes Voice Natural:
1. **Prosody** - Pitch variation, rhythm, stress patterns
2. **Emotion** - Happy, sad, neutral, excited tones  
3. **Breathing** - Natural pauses, breath sounds
4. **Speaker consistency** - Same voice character
5. **Disfluencies** - Natural "uh", "um" (optional)

### Data Quality Checklist:
- [ ] Multiple speakers (50+) for variety
- [ ] Emotional range in recordings
- [ ] Conversational style (not just read text)
- [ ] Clean audio (no background noise)
- [ ] Natural speaking rate

---

## 📥 Download Strategy (No Cookies/No YouTube)

### Phase 1: HuggingFace Datasets (Largest & Easiest)
```bash
# Kathbath - 1684 hours (BEST for conversation!)
huggingface-cli download ai4bharat/kathbath --local-dir data/kathbath

# IndicVoices - Multi-speaker Telugu
huggingface-cli download ai4bharat/indicvoices --local-dir data/indicvoices

# Common Voice
huggingface-cli download mozilla-foundation/common_voice_16_1 \
    --config te --local-dir data/common_voice
```

### Phase 2: Direct Downloads (OpenSLR)
```bash
# SLR66 - Studio quality Telugu
wget https://www.openslr.org/resources/66/te_in_female.zip
wget https://www.openslr.org/resources/66/te_in_male.zip

# MUCS Telugu
wget https://www.openslr.org/resources/103/te_in.zip
```

### Phase 3: Vakyansh (ekstep)
```bash
# 2400 hours Telugu ASR data
# Download from: https://ekstep.org/language-datasets/
```

---

## 🏗️ Architecture for Production S2S

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION S2S ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Audio In (16kHz) ──► YOUR CODEC ──► Codes [8 quantizers]      │
│                           │                                     │
│                           ▼                                     │
│           ┌───────────────────────────────────┐                │
│           │      AUDIO LANGUAGE MODEL         │                │
│           │      (Transformer 500M-1B)        │                │
│           │                                   │                │
│           │  Features:                        │                │
│           │  - Multi-speaker embeddings       │                │
│           │  - Emotion tokens                 │                │
│           │  - Prosody modeling               │                │
│           │  - Streaming generation           │                │
│           └───────────────────────────────────┘                │
│                           │                                     │
│                           ▼                                     │
│  Audio Out ◄── YOUR CODEC ◄── Response Codes                  │
│                                                                 │
│  Target: <200ms first-token, <500ms full response              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 Training Curriculum

### Stage 1: Reconstruction (Codec validation)
- Input: Audio → Codes → Audio
- Ensure codec preserves quality

### Stage 2: Next-token prediction (Audio LM)
- Train on all Telugu audio (unsupervised)
- Learn Telugu speech patterns

### Stage 3: Conversation fine-tuning
- Train on Q&A pairs
- Learn to generate responses

### Stage 4: Speaker/Emotion conditioning
- Add speaker embeddings
- Add emotion tokens

---

## ⏱️ Timeline

| Week | Task | GPU Hours |
|------|------|-----------|
| 1 | Download & preprocess all data | 20h |
| 2 | Train Audio LM (unsupervised) | 80h |
| 3 | Generate conversation pairs | 20h |
| 4 | Train S2S for conversation | 60h |
| 5 | Fine-tune & evaluate | 30h |
| **Total** | | **210h (~$420)** |

---

## ✅ Action Items

1. [ ] Download Kathbath (1684 hours) - PRIORITY
2. [ ] Download OpenSLR datasets
3. [ ] Download IndicVoices
4. [ ] Preprocess all audio to 16kHz mono
5. [ ] Encode all audio with YOUR codec
6. [ ] Train Audio LM on all data
7. [ ] Generate conversation pairs
8. [ ] Train S2S for conversation
9. [ ] Add speaker/emotion conditioning
10. [ ] Optimize for streaming inference
