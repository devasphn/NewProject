# 🎯 Training S2S Model for Conversation

## Current State

You have:
- ✅ `best_codec.pt` - Audio codec (encode/decode audio to codes)
- ✅ `s2s_best.pt` - S2S transformer (trained for RECONSTRUCTION)

What's missing:
- ❌ S2S trained for CONVERSATION (input → response)

---

## 🔄 The Difference

### Reconstruction (What you trained)
```
Input:  "నమస్కారం" audio → codes [1,2,3,4...]
Output: "నమస్కారం" audio → codes [1,2,3,4...]  (SAME!)
```

### Conversation (What you need)
```
Input:  "నమస్కారం" audio → codes [1,2,3,4...]
Output: "నమస్కారం! ఎలా ఉన్నారు?" audio → codes [5,6,7,8...]  (DIFFERENT!)
```

---

## 📊 Training Data Required

### Option A: Parallel Conversation Audio
```
data/
├── conversations/
│   ├── conv_001/
│   │   ├── user.wav      # User's question
│   │   └── assistant.wav # Assistant's response
│   ├── conv_002/
│   │   ├── user.wav
│   │   └── assistant.wav
│   └── ... (1000+ pairs minimum)
```

**Where to get this:**
1. **Record yourself**: Ask questions, record responses
2. **TTS + LLM**: Generate synthetic pairs
   - Use LLM to generate Q&A text pairs
   - Use TTS to synthesize both sides
3. **Existing datasets**: Search for Telugu dialogue datasets

### Option B: Synthetic Data Generation (Recommended for POC)
```python
# Generate training data using LLM + TTS
questions = [
    "నమస్కారం",
    "మీ పేరు ఏమిటి?",
    "ఈ రోజు వాతావరణం ఎలా ఉంది?",
    # ... 1000+ questions
]

for q in questions:
    # Generate response using LLM
    response = llm.generate(q)
    
    # Synthesize both to audio
    q_audio = tts.synthesize(q)
    r_audio = tts.synthesize(response)
    
    # Encode to codes
    q_codes = codec.encode(q_audio)
    r_codes = codec.encode(r_audio)
    
    # Save as training pair
    save_pair(q_codes, r_codes)
```

---

## 🔧 Modified Training Script

```python
# train_s2s_conversation.py

class ConversationDataset(Dataset):
    """Dataset of (input_codes, response_codes) pairs"""
    
    def __init__(self, data_dir: str, codec):
        self.pairs = []
        
        for conv_dir in Path(data_dir).glob("conv_*"):
            user_audio = load_audio(conv_dir / "user.wav")
            asst_audio = load_audio(conv_dir / "assistant.wav")
            
            # Encode to codes using YOUR codec
            user_codes = codec.encode(user_audio)
            asst_codes = codec.encode(asst_audio)
            
            self.pairs.append((user_codes, asst_codes))
    
    def __getitem__(self, idx):
        input_codes, target_codes = self.pairs[idx]
        return {
            "input_codes": input_codes,   # User's audio codes
            "target_codes": target_codes  # Assistant's response codes
        }

# Training loop
for batch in dataloader:
    input_codes = batch["input_codes"]    # [B, Q, T1]
    target_codes = batch["target_codes"]  # [B, Q, T2]
    
    # Forward pass
    output = model(input_codes, target_codes[:, :, :-1])
    
    # Loss: predict response codes given input codes
    loss = F.cross_entropy(
        output.view(-1, vocab_size),
        target_codes[:, :, 1:].reshape(-1)
    )
    
    loss.backward()
    optimizer.step()
```

---

## 📈 Training Strategy

### Phase 1: Synthetic Data (1-2 days)
1. Generate 1000+ Q&A text pairs using LLM
2. Synthesize to audio using Edge TTS
3. Encode using YOUR codec
4. Train S2S for 10-20 epochs

### Phase 2: Real Data (Optional, for quality)
1. Record real Telugu conversations
2. Augment with synthetic data
3. Fine-tune the model

### Phase 3: Evaluation
1. Test with unseen questions
2. Measure response quality
3. Measure latency

---

## 🚀 Quick Start Script

```bash
# 1. Generate synthetic training data
python generate_conversation_data.py \
    --num_pairs 1000 \
    --output_dir data/conversations \
    --codec best_codec.pt

# 2. Train S2S for conversation
python train_s2s_conversation.py \
    --data_dir data/conversations \
    --codec best_codec.pt \
    --epochs 20 \
    --output s2s_conversation.pt

# 3. Test the trained model
python realtime_s2s_agent.py \
    --codec best_codec.pt \
    --s2s s2s_conversation.pt
```

---

## ⏱️ Expected Results

| Metric | Reconstruction | Conversation |
|--------|---------------|--------------|
| Input | Same audio | Question audio |
| Output | Same audio | Response audio |
| Latency | ~70ms | ~100-200ms |
| Training data | Any audio | Q&A pairs |
| Training time | 4-6 hours | 8-12 hours |

---

## 📋 Summary

**To build full S2S conversation:**

1. **Generate data**: Create (question, answer) audio pairs
2. **Modify training**: Train S2S to predict answer codes from question codes
3. **Fine-tune**: Use real conversation data for better quality

**The fastest path:**
- Use synthetic data (LLM + TTS generated)
- Train for 10-20 epochs
- Test and iterate

Would you like me to create the `generate_conversation_data.py` script?
