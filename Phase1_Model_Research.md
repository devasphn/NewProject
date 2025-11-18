# Phase 1: Model Research & Selection
## Ultra-Low Latency Telugu S2S Voice Agent

---

## Executive Summary

**Recommended Model: Moshi by Kyutai Labs** ✅
- **Latency**: 200ms on L4 GPU (well under 500ms target)
- **License**: Apache 2.0 (100% commercial-free)
- **Architecture**: Full-duplex streaming S2S
- **GPU**: L4 recommended (optimal cost-performance)

---

## 1. End-to-End S2S Models Analysis

### 1.1 Moshi (Kyutai Labs) - **RECOMMENDED** ✅

#### Overview
Moshi is the breakthrough full-duplex S2S model achieving 160-200ms latency with production-ready performance. Released in 2024, it eliminates traditional VAD→ASR→LLM→TTS pipelines.

#### Technical Specifications
- **Model Size**: 7B parameters (Temporal Transformer) + Depth Transformer
- **Audio Codec**: Mimi (80ms latency, 1.1 kbps bandwidth)
- **Frame Rate**: 12.5 Hz (80ms frames)
- **Latency**: 160ms theoretical, 200ms practical on L4 GPU
- **License**: Apache 2.0 (fully open source, commercial-friendly)
- **Training Data**: 100K+ hours multilingual speech
- **Repository**: https://github.com/kyutai-labs/moshi

#### Architecture Innovation
```
User Audio → Mimi Encoder (80ms) → Temporal Transformer (7B) → 
Inner Monologue (text tokens) → Depth Transformer → 
Mimi Decoder (80ms) → Agent Audio Output

Simultaneous Processing:
├─ User Stream (continuous input)
└─ Agent Stream (continuous output)
```

#### Key Features
1. **Full-Duplex**: Processes user input and generates output simultaneously
2. **Inner Monologue**: Predicts text tokens before acoustic tokens (improves linguistic quality)
3. **Streaming**: Frame-by-frame processing (no need for complete utterance)
4. **No Explicit VAD**: Implicit speech detection via frame-level state tokens
5. **Multi-task**: Built-in ASR and TTS capabilities as byproducts

#### Performance Metrics
- Latency on L4 GPU: 200ms
- Latency on A40 GPU: 180ms
- Audio Quality: 24kHz output (high fidelity)
- Compression: 1.1 kbps (ultra-efficient)
- Concurrent users on L4: 8-12 sessions

#### Mimi Codec (Included)
- **Sampling Rate**: 24 kHz
- **Latency**: 80ms per direction (160ms total)
- **Bandwidth**: 1.1 kbps (extremely efficient)
- **Architecture**: Transformer-based encoder-decoder
- **Quality**: Distillation from WavLM (preserves semantic + acoustic info)

#### Why Moshi for This Project
1. ✅ Achieves <500ms latency (200ms + 150ms overhead = 350ms total)
2. ✅ Apache 2.0 license (100% commercial-free, no restrictions)
3. ✅ Full-duplex eliminates turn-based delays
4. ✅ Proven deployment on L4/A40 GPUs (RunPod compatible)
5. ✅ Active development and community support
6. ✅ Can be fine-tuned on Telugu datasets via LoRA
7. ✅ No external API dependencies

#### Limitations
- Trained primarily on English/French (requires Telugu fine-tuning)
- Maximum session length ~10 minutes (WebSocket stability)
- GPU memory: ~16GB VRAM minimum (L4 has 24GB)

---

### 1.2 Alternative Models (Not Recommended)

#### A. SpeechGPT2
**Status**: Research prototype
- **Pros**: Emotion/style control, 750 bps ultra-compression, 7B model
- **Cons**: Higher latency (~40ms per token × 25 tokens/sec = 1 second), no full-duplex
- **License**: Unclear/research-only
- **Verdict**: ❌ Exceeds latency target + unclear licensing

#### B. Meta Seamless Communication
**Status**: Production-ready
- **Pros**: 100+ languages, robust translation
- **Cons**: 2-second latency (4x our target), translation-focused not conversational
- **License**: CC-BY-NC 4.0 (NON-COMMERCIAL ONLY)
- **Verdict**: ❌ License incompatible + excessive latency

#### C. OpenAI Realtime API
**Status**: Commercial service
- **Pros**: Production-ready, low latency, high quality
- **Cons**: Proprietary, API costs ($0.06/min), no self-hosting
- **Verdict**: ❌ Violates "zero external API costs" requirement

#### D. VALL-E-X
**Status**: Research project
- **Pros**: Zero-shot voice cloning, multilingual support
- **Cons**: Microsoft Research (non-commercial license), TTS-only (not S2S)
- **Verdict**: ❌ License incompatible + not end-to-end

#### E. Google AudioPaLM
**Status**: Research paper
- **Pros**: Multimodal, translation
- **Cons**: Not open-sourced, no public weights
- **Verdict**: ❌ Not available

---

## 2. VAD Solutions

Since Moshi has implicit VAD, we need lightweight VAD only for optional preprocessing/bandwidth optimization.

### 2.1 Silero VAD - **RECOMMENDED** ✅

#### Specifications
- **Latency**: 5-10ms per frame
- **Accuracy**: 98%+ (TPR at 5% FPR)
- **Model Size**: 2MB (ONNX)
- **License**: MIT (commercial-friendly)
- **CPU Usage**: ~2% on modern CPUs
- **Supported Rates**: 8kHz, 16kHz
- **Training**: 6000+ languages (handles Telugu)

#### Why Silero VAD
- Ultra-lightweight (2MB model, runs on CPU)
- Trained on massive multilingual dataset
- WebSocket-compatible (16kHz PCM)
- Can run parallel to Moshi for bandwidth optimization
- MIT license (no restrictions)

#### Use Cases in Our Architecture
1. **Optional**: Pre-filter silence before sending to Moshi (saves bandwidth)
2. **Monitoring**: Track speech activity for analytics
3. **Fallback**: If Moshi implicit VAD fails in edge cases

### 2.2 Alternative VAD Options

#### A. WebRTC VAD
- **Pros**: Ultra-lightweight (~1ms latency), zero dependencies
- **Cons**: Low accuracy (50% TPR at 5% FPR), struggles with noise
- **Verdict**: ❌ Too inaccurate for production

#### B. TEN VAD
- **Pros**: C++ implementation, very fast, 2024 release
- **Cons**: Newer (less battle-tested than Silero)
- **Verdict**: ⚠️ Consider if C++ performance critical

#### C. Cobra VAD (Picovoice)
- **Pros**: Best accuracy (98.9% TPR), very efficient
- **Cons**: Commercial license ($0.0025/minute after free tier)
- **Verdict**: ❌ Violates "zero cost" requirement

---

## 3. Model Selection Verdict

### Final Decision: Moshi + Silero VAD

**Primary Model**: Moshi (full-duplex S2S)
- Handles end-to-end speech processing
- 200ms latency on L4 GPU
- Apache 2.0 license
- Fine-tune on Telugu data

**Optional VAD**: Silero VAD
- Pre-filter silence (bandwidth optimization)
- 5-10ms latency overhead
- MIT license
- Run on CPU (doesn't compete for GPU resources)

**Architecture Flow**:
```
Browser Audio → WebSocket → [Optional: Silero VAD] → Moshi S2S → WebSocket → Browser Playback
```

---

## 4. Licensing Verification

### Moshi (Apache 2.0)
✅ **Commercial use**: Allowed  
✅ **Modification**: Allowed  
✅ **Distribution**: Allowed  
✅ **Patent grant**: Included  
✅ **Attribution**: Required (in LICENSE file, not runtime)  
✅ **Private use**: Allowed  

### Silero VAD (MIT)
✅ **Commercial use**: Allowed  
✅ **Modification**: Allowed  
✅ **Distribution**: Allowed  
✅ **Patent grant**: Not specified (but permissive)  
✅ **Attribution**: Required (in LICENSE file)  
✅ **Private use**: Allowed  

**Conclusion**: Both licenses are 100% commercially free with no runtime attribution requirements.

---

## 5. Comparison Table

| Model | Latency | License | Telugu Support | Cost | Verdict |
|-------|---------|---------|---------------|------|---------|
| **Moshi** | 200ms | Apache 2.0 | Fine-tune | GPU only | ✅ **SELECTED** |
| SpeechGPT2 | 1000ms | Unclear | Fine-tune | GPU only | ❌ Too slow |
| Seamless | 2000ms | Non-commercial | Native | GPU only | ❌ License |
| OpenAI API | 300ms | Proprietary | Native | $0.06/min | ❌ API cost |
| VALL-E-X | N/A (TTS) | Non-commercial | Fine-tune | GPU only | ❌ License |

---

## Next Steps
1. ✅ Model selected: Moshi
2. ✅ VAD selected: Silero (optional)
3. ⏭️ Design system architecture (next document)
4. ⏭️ Create training plan for Telugu adaptation
5. ⏭️ GPU selection and cost analysis
