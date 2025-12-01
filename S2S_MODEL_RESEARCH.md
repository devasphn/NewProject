# S2S Voice AI Research: Moshi, Maya, EVI

## Executive Summary

All three leading S2S voice AI systems (Moshi, Maya/CSM, EVI) use **conversational training data** and similar architectural patterns. Your approach of training with conversational pairs is **correct**.

---

## 1. MOSHI (Kyutai Labs)

### Architecture
```
Audio Input (24kHz) 
    ↓
Mimi Codec (encodes to 12.5Hz tokens)
    ↓
Temporal Transformer (7B parameters)
    ↓
Depth Transformer (for inter-codebook)
    ↓
Text Tokens (inner monologue) + Audio Tokens
    ↓
Mimi Decoder
    ↓
Audio Output (24kHz)
```

### Key Features
- **Full-duplex**: Models TWO audio streams simultaneously (user + AI)
- **Latency**: 160-200ms theoretical
- **Codec**: Mimi (12.5Hz, uses WavLM distillation)
- **Model size**: 7B parameters

### Training Data
- **Pre-training data**: NOT RELEASED
- **Architecture**: Uses interleaved audio + text (inner monologue)
- **Conversational**: YES - models both sides of conversation

### What We Learn
> Moshi uses **paired conversational data** where it models both the user's speech and its own response simultaneously.

---

## 2. MAYA / CSM (Sesame AI)

### Architecture
```
Interleaved Text (T) + Audio (A) tokens
    ↓
Backbone Transformer (Llama-based)
    ↓ (predicts zeroth codebook)
Decoder Transformer (smaller)
    ↓ (predicts codebooks 1 to N-1)
Mimi Tokenizer (12.5Hz)
    ↓
Reconstructed Audio
```

### Key Features
- **Two-stage**: Backbone for semantics, Decoder for acoustics
- **Tokenizer**: Uses Mimi (same as Moshi)
- **Frame rate**: 12.5Hz
- **RVQ codebooks**: Split at zeroth level

### Training Approach (from their paper)
> "Training samples are structured as **alternating interleaved patterns of text and audio**, with speaker identity encoded directly in the text representation."

### What We Learn
> Sesame trains on **interleaved conversation turns** with speaker identity. They use paired conversational data.

---

## 3. EVI (Hume AI)

### Architecture
- **Type**: Voice-to-voice foundation model
- **Latency**: Subsecond response times
- **Features**: Emotional intelligence, tone understanding

### Key Features (from their blog)
> "EVI 2 is trained to maintain characters and personalities that are fun and interesting to interact with."

### Training Approach
- Uses emotional/prosodic training
- Trained on conversational interactions
- Personality and character consistency

### What We Learn
> Hume trains on **emotional conversational data** with personality consistency.

---

## 4. Common Patterns Across All Three

| Feature | Moshi | Maya/CSM | EVI |
|---------|-------|----------|-----|
| Conversational pairs | ✅ | ✅ | ✅ |
| Audio codec | Mimi (12.5Hz) | Mimi (12.5Hz) | Custom |
| Text integration | Inner monologue | Interleaved | Unknown |
| Full-duplex | ✅ | ✅ | ✅ |
| Low latency | 200ms | ~300ms | <1s |

### Key Insight
**ALL three systems train on conversational data, not just raw audio!**

---

## 5. Training Data Requirements

### For Codec
- Raw audio (any language)
- 1000+ hours per language
- Mixed speakers, accents, qualities
- **Does NOT need transcripts or pairs**

### For S2S Model
- **Conversational pairs required**
- Format: (User utterance, AI response)
- Multiple turns per conversation
- Speaker identity labels
- **Transcripts helpful but not mandatory**

### Estimated Data Needs
| Component | Data Type | Amount |
|-----------|-----------|--------|
| Codec | Raw audio | 3000h total (1000h x 3 langs) |
| S2S | Conversation pairs | 10,000-50,000 pairs |
| S2S | Total audio hours | 500-1000h of dialogues |

---

## 6. How to Create Conversational Pairs

### Option 1: Existing Dialogue Datasets
```python
# Fisher Corpus (English conversations)
# CallHome (Phone conversations)
# DSTC (Dialogue State Tracking Challenge)
```

### Option 2: Synthetic Generation
```python
# Use LLM to generate conversation scripts
# Use TTS to synthesize both sides
# Advantage: Control over content & quality
```

### Option 3: Role-based Recording
```
# Record human conversations with:
# - Person A asks questions (user role)
# - Person B responds (AI role)
# - Multiple speakers for variety
```

### Option 4: Extract from Podcasts/Interviews
```python
# Download interview podcasts
# Use speaker diarization
# Segment into Q&A pairs
```

---

## 7. Recommended Architecture for Your S2S

Based on research, here's the recommended approach:

```
┌─────────────────────────────────────────┐
│           YOUR S2S ARCHITECTURE         │
├─────────────────────────────────────────┤
│                                         │
│   User Audio                            │
│       ↓                                 │
│   Production Codec (50Hz)               │
│       ↓                                 │
│   Audio Tokens [8 codebooks]            │
│       ↓                                 │
│   Conformer Encoder (6 layers)          │
│       ↓                                 │
│   Cross-attention with Context          │
│       ↓                                 │
│   Transformer Decoder (6 layers)        │
│       ↓                                 │
│   Audio Tokens [8 codebooks]            │
│       ↓                                 │
│   Production Codec Decoder              │
│       ↓                                 │
│   Response Audio                        │
│                                         │
└─────────────────────────────────────────┘
```

### Key Differences from Current
1. **Add conversation history** (previous turns)
2. **Add speaker embeddings** (user vs AI)
3. **Train on paired data** (not just audio)

---

## 8. Training Strategy

### Phase 1: Codec Training
```
Data: 3000h raw audio (English + Hindi + Telugu)
Duration: 40-60 hours on H200
Output: Production codec that works for all 3 languages
```

### Phase 2: S2S Pre-training
```
Data: 500h of dialogue audio + pairs
Duration: 60-80 hours on H200
Output: Basic conversation capability
```

### Phase 3: S2S Fine-tuning
```
Data: High-quality conversation pairs
Duration: 20-40 hours on H200
Output: Production-grade responses
```

---

## 9. Verdict: Is Your Approach Correct?

### ✅ YES - Conversational pairs are REQUIRED

Based on research:
- Moshi uses paired conversational data
- Sesame CSM uses interleaved conversation turns
- EVI trains on emotional conversations

### Your MD is RIGHT
> Training S2S with conversational pairs is the correct approach. All leading S2S models do this.

### Data Priority
1. **Codec**: Raw audio (1000h per language) ✅
2. **S2S**: Conversation pairs (10,000+ pairs) ⚠️ Need to create/collect

---

## 10. Next Steps

1. [ ] Train codec on 3000h raw audio
2. [ ] Collect/create conversation pairs:
   - Fisher Corpus (English)
   - Hindi dialogue datasets
   - Telugu dialogue synthesis
3. [ ] Train S2S on conversation pairs
4. [ ] Fine-tune for personality/style

---

## References

1. Moshi: https://github.com/kyutai-labs/moshi
2. Sesame CSM: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
3. Hume EVI: https://www.hume.ai/blog/introducing-evi2
4. Mimi Codec: https://huggingface.co/kyutai/mimi
