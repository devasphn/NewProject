# Telugu Ultra-Low Latency S2S Architecture (<150ms)
## Beating Luna Demo by Pixa AI

---

## 🎯 Project Goals

### Performance Targets
- **Latency**: <150ms (first audio chunk)
- **Streaming**: Real-time audio generation
- **Quality**: Native Telugu with emotional expression
- **Speakers**: 2 male + 2 female with distinct accents
- **Emotions**: Laughter, excitement, empathy, surprise

### Key Differentiators vs Luna Demo
1. **Custom Neural Codec**: 16KHz optimized for Telugu phonemes
2. **Streaming Transformer**: Parallel encoding/decoding
3. **Emotion Tokens**: Embedded emotional control (laugh, cry, etc.)
4. **Zero-Shot Voice Cloning**: Speaker adaptation in <100ms

---

## 🏗️ System Architecture

### Overview: Dual-Stream Processing
```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (User Side)                       │
├─────────────────────────────────────────────────────────────┤
│ Audio Input → WebAudio API → Opus Encode → WebSocket        │
│                                                              │
│ WebSocket → Opus Decode → WebAudio API → Audio Output       │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
             [WebSocket]            [WebSocket]
                   │                      │
┌──────────────────▼──────────────────────▼───────────────────┐
│                    GPU Server (RunPod)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Streaming Encoder Pipeline                 │  │
│  │                                                       │  │
│  │  Opus Decode → Resample → Telugu Codec Encoder       │  │
│  │       ↓            ↓              ↓                   │  │
│  │    [16KHz]     [16KHz]     [Discrete Tokens]         │  │
│  │                                   ↓                   │  │
│  └───────────────────────────────────┼───────────────────┘  │
│                                      │                       │
│  ┌───────────────────────────────────▼───────────────────┐  │
│  │         Streaming S2S Transformer Core                │  │
│  │                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐                 │  │
│  │  │  Encoder    │───▶│  Decoder    │                 │  │
│  │  │  (Conformer)│    │  (GPT-style)│                 │  │
│  │  └─────────────┘    └─────────────┘                 │  │
│  │                                                       │  │
│  │  Features:                                            │  │
│  │  • Sliding window attention (100ms chunks)           │  │
│  │  • KV-cache for ultra-fast generation                │  │
│  │  • Emotion embeddings (laugh, excitement)            │  │
│  │  • Speaker embeddings (4 Telugu voices)              │  │
│  │                                                       │  │
│  └───────────────────────────────────┼───────────────────┘  │
│                                      │                       │
│  ┌───────────────────────────────────▼───────────────────┐  │
│  │           Streaming Decoder Pipeline                  │  │
│  │                                                       │  │
│  │  [Response Tokens] → Telugu Codec Decoder → [16KHz]   │  │
│  │                              ↓                        │  │
│  │                     Opus Encode → WebSocket           │  │
│  │                                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Components

### 1. Custom Telugu Neural Codec (TeluCodec)

#### Architecture: Residual Vector Quantization (RVQ)
```python
class TeluCodec:
    """
    Custom neural codec optimized for Telugu speech at 16KHz
    Based on Encodec but optimized for Telugu phonemes
    """
    
    # Encoder: Conv1D Stack
    encoder_layers = [
        Conv1d(1, 32, kernel_size=7, stride=1),   # 16000Hz
        Conv1d(32, 64, kernel_size=7, stride=2),  # 8000Hz
        Conv1d(64, 128, kernel_size=7, stride=2), # 4000Hz
        Conv1d(128, 256, kernel_size=7, stride=2),# 2000Hz
        Conv1d(256, 512, kernel_size=7, stride=5),# 400Hz
        Conv1d(512, 1024, kernel_size=3, stride=2)# 200Hz (80x compression)
    ]
    
    # Quantizer: 8-level RVQ
    quantizer = ResidualVectorQuantizer(
        dim=1024,
        codebook_size=1024,      # 10-bit codes
        num_quantizers=8,         # 8 levels
        commitment_weight=0.25,
        orthogonal_reg_weight=0.1,
        threshold_ema_dead_code=2  # Replace dead codes
    )
    
    # Decoder: Transposed Conv1D Stack (mirror of encoder)
    decoder_layers = [...] # Reverse of encoder
    
    # Bitrate: 200Hz * 8 quantizers * 10 bits = 16kbps
    # Latency: <10ms encoding, <10ms decoding
```

#### Training Strategy
- **Dataset**: 100 hours Telugu speech (mixed emotions)
- **Loss**: Reconstruction + Perceptual + Adversarial
- **GPU**: H200 (8 hours training)
- **Optimization**: Flash Attention, Mixed Precision (BF16)

### 2. Streaming S2S Transformer

#### Encoder: Conformer Architecture
```python
class StreamingEncoder:
    """
    Conformer-based encoder with sliding window attention
    Processes 100ms chunks with 20ms lookahead
    """
    
    def __init__(self):
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=768,
                depth=12,
                heads=12,
                conv_kernel_size=31,  # 100ms receptive field
                use_flash_attn=True,
                use_rotary_emb=True,  # RoPE for positions
            ) for _ in range(12)
        ])
        
        # Sliding window for streaming
        self.window_size = 100  # ms
        self.lookahead = 20     # ms
        self.chunk_size = 20    # Process every 20ms
    
    def forward(self, tokens, speaker_emb, emotion_emb):
        # Combine embeddings
        x = tokens + speaker_emb + emotion_emb
        
        # Process through conformer blocks
        for layer in self.layers:
            x = layer(x, mask=self.sliding_mask)
        
        return x
```

#### Decoder: GPT-style with KV Cache
```python
class StreamingDecoder:
    """
    GPT-style decoder optimized for streaming generation
    Uses KV-cache for O(1) token generation
    """
    
    def __init__(self):
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=768,
                heads=12,
                mlp_ratio=4,
                use_flash_attn=True,
                use_kv_cache=True  # Critical for speed
            ) for _ in range(12)
        ])
        
        # KV Cache for ultra-fast generation
        self.kv_cache = KVCache(
            num_layers=12,
            max_seq_len=4096,
            batch_size=1,
            num_heads=12,
            head_dim=64
        )
        
    def generate_streaming(self, encoder_output):
        """Generate tokens with <5ms per token"""
        while not self.is_complete():
            # Use cached KV for O(1) generation
            next_token = self.forward_cached(encoder_output)
            yield next_token  # Stream immediately
```

### 3. Emotion & Speaker Control

#### Emotion Tokens
```python
EMOTION_TOKENS = {
    "[NEUTRAL]": 0,
    "[HAPPY]": 1,
    "[LAUGH]": 2,      # Natural laughter
    "[EXCITED]": 3,    # Enthusiasm
    "[EMPATHY]": 4,    # Compassionate
    "[SURPRISE]": 5,   # Shocked/amazed
    "[THINKING]": 6,   # Contemplative
    "[Telugu_ACCENT_HEAVY]": 7,  # Strong Telugu accent
    "[TELUGU_ACCENT_MILD]": 8,   # Mild Telugu accent
}
```

#### Speaker Embeddings
```python
SPEAKERS = {
    "male_1": "Young professional (25-30)",     # Clear, energetic
    "male_2": "Mature narrator (35-45)",        # Deep, authoritative
    "female_1": "Young conversational (22-28)", # Friendly, expressive
    "female_2": "Professional anchor (30-40)",  # Clear, confident
}
```

---

## 📊 Performance Optimizations

### 1. Kernel Fusion & Graph Optimization
```python
# Compile model with TorchScript for 30% speedup
model = torch.jit.script(model)

# Use CUDA graphs for deterministic execution paths
with torch.cuda.graph():
    output = model(input)

# Fuse operations with TorchDynamo
model = torch.compile(model, mode="reduce-overhead")
```

### 2. Memory Optimization
```python
# Pin memory for faster CPU-GPU transfer
dataloader = DataLoader(
    dataset,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

# Use gradient checkpointing during training
model.gradient_checkpointing_enable()
```

### 3. Inference Optimization
```python
# INT8 quantization for encoder (2x speedup)
encoder = torch.quantization.quantize_dynamic(
    encoder, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
)

# Keep decoder in FP16 for quality
decoder = decoder.half()

# Use Flash Attention 2 for 4x attention speedup
from flash_attn import flash_attn_func
```

---

## 🔄 Latency Breakdown

### Target: <150ms End-to-End

| Component | Latency | Cumulative | Notes |
|-----------|---------|------------|-------|
| Browser audio capture | 10ms | 10ms | WebAudio ScriptProcessor |
| Opus encode (browser) | 5ms | 15ms | Hardware accelerated |
| WebSocket transfer | 5ms | 20ms | Local network |
| Opus decode (server) | 3ms | 23ms | |
| Telugu Codec encode | 10ms | 33ms | Custom optimized |
| S2S Encoder | 20ms | 53ms | Sliding window |
| S2S Decoder (first token) | 15ms | 68ms | KV-cached |
| Telugu Codec decode | 10ms | 78ms | Streaming |
| Opus encode (server) | 3ms | 81ms | |
| WebSocket transfer | 5ms | 86ms | |
| Opus decode (browser) | 5ms | 91ms | |
| Audio playback buffer | 20ms | 111ms | WebAudio |
| **Safety margin** | 39ms | **150ms** | **TOTAL TARGET** |

### Streaming Continuation
After first chunk (150ms), subsequent audio streams in real-time:
- Generate 20ms audio every 20ms
- Continuous playback with no gaps
- Adaptive jitter buffer for network variation

---

## 🎓 Key Innovations

### 1. Dual-Path Processing
- Encode and decode simultaneously
- Parallel processing reduces latency

### 2. Predictive Pre-generation
- Start generating likely responses before user finishes
- Discard if actual input differs
- Saves 50-100ms on common phrases

### 3. Adaptive Bitrate
- Reduce codec bitrate during silence
- Increase quality during speech
- Dynamic quality/latency tradeoff

### 4. Telugu-Specific Optimizations
- Phoneme-aware codec design
- Telugu morphology in tokenizer
- Accent-preserving quantization

---

## 💰 Cost Analysis

### Training Phase (One-time)
| Component | Hours | GPU | Cost |
|-----------|-------|-----|------|
| Codec Training | 8 | H200 | $32 |
| S2S Model Training | 24 | H200 | $96 |
| Fine-tuning | 4 | H200 | $16 |
| Testing | 2 | A6000 | $1 |
| **TOTAL** | **38** | | **$145** |

### Inference (Production)
| GPU | Cost/hr | Users | Cost/user-hr |
|-----|---------|-------|--------------|
| RTX A6000 | $0.49 | 100 | $0.0049 |
| RTX 4090 | $0.39 | 50 | $0.0078 |
| H100 PCIe | $2.49 | 500 | $0.0050 |

---

## 🎯 Comparison with Luna Demo

| Feature | Luna Demo | Our System | Winner |
|---------|-----------|------------|--------|
| Latency | ~200ms | <150ms | **Ours ✓** |
| Language | English | Telugu+English | **Ours ✓** |
| Emotions | Basic | Laughter+6 emotions | **Ours ✓** |
| Speakers | 1 | 4 (2M+2F) | **Ours ✓** |
| Streaming | Yes | Yes | Tie |
| Voice Clone | Yes | Yes | Tie |
| Open Source | No | Yes | **Ours ✓** |

---

## 🚀 Implementation Timeline

### Week 1: Foundation (Days 1-7)
- Day 1-2: Data collection pipeline
- Day 3-4: Codec architecture implementation
- Day 5-6: S2S model architecture
- Day 7: Integration testing

### Week 2: Training (Days 8-14)
- Day 8-9: Codec training (H200)
- Day 10-12: S2S model training (H200)
- Day 13: Fine-tuning & optimization
- Day 14: Deployment preparation

### Week 3: Production (Days 15-21)
- Day 15-16: RunPod deployment
- Day 17-18: WebSocket server optimization
- Day 19-20: Browser client refinement
- Day 21: Launch & monitoring

---

## ✅ Success Criteria

### Performance
- [ ] First audio chunk <150ms
- [ ] Streaming at real-time factor <0.8
- [ ] 100+ concurrent users on single A6000

### Quality
- [ ] MOS score >4.0/5
- [ ] Telugu accent accuracy >90%
- [ ] Emotion recognition >85%

### Business
- [ ] Total training cost <$150
- [ ] Inference cost <$0.005/user-hour
- [ ] Better than Luna Demo benchmarks