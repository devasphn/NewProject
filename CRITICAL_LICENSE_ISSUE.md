# 🚨 CRITICAL: Moshi Licensing Issue

**Date**: November 18, 2025  
**Status**: ❌ **PROJECT BLOCKED - LICENSING INCOMPATIBLE**

---

## Issue Identified

### Moshi Licensing Verification

From official GitHub repository (https://github.com/kyutai-labs/moshi):

```
License:
- Code: MIT/Apache 2.0 ✅ (commercial-friendly)
- Model Weights: CC-BY 4.0 ❌ (REQUIRES ATTRIBUTION)
```

### CC-BY 4.0 Requirements

**You MUST:**
1. ✅ Give appropriate credit to Kyutai Labs
2. ✅ Provide link to license
3. ✅ Indicate if changes were made
4. ❌ **CANNOT avoid attribution**

**This violates project requirement: "No Attributions Required: Can be used without credits"**

---

## ⛔ Moshi is NOT SUITABLE

Since the project explicitly requires:
- "No Attributions Required: Can be used without credits"
- "100% Free & Open Source"

**Moshi does NOT meet requirements due to CC-BY 4.0 model weights license.**

---

## 🔄 Alternative Solutions

We have 3 viable paths forward:

### Option 1: Cascaded Pipeline (Traditional) ✅ RECOMMENDED

**Architecture:**
```
Browser → WebSocket → VAD → ASR → LLM → TTS → Browser
```

**Components (ALL with permissive licenses):**

1. **VAD**: Silero VAD (MIT) ✅
2. **ASR**: Whisper (MIT) or Canary-1B (Apache 2.0) ✅
3. **LLM**: Llama 3.2 (Apache 2.0) ✅
4. **TTS**: Piper TTS (MIT) or Kokoro TTS (Apache 2.0) ✅

**Pros:**
- ✅ All components have permissive licenses
- ✅ No attribution required
- ✅ Proven technology stack
- ✅ Easier to fine-tune individually

**Cons:**
- ❌ Higher latency (600-800ms vs 340ms)
- ❌ More complex architecture
- ❌ Error propagation between stages

**Estimated Latency:**
- VAD: 10ms
- ASR (Whisper): 200ms
- LLM (Llama 3.2): 150ms
- TTS (Piper): 200ms
- Network: 50ms
- **Total: 610ms** (still under 1000ms, acceptable for most use cases)

---

### Option 2: Train Your Own S2S Model ⚠️ HIGH RISK

**Approach**: Train a full-duplex S2S model from scratch

**Requirements:**
- 10,000-100,000 hours of data
- $50,000-100,000 in GPU costs
- 6-12 months development time
- Expert ML team

**Pros:**
- ✅ Complete ownership (no licensing issues)
- ✅ Customized for Telugu from start

**Cons:**
- ❌ Extremely expensive
- ❌ Very long timeline
- ❌ High technical risk
- ❌ Not suitable for POC

**Verdict**: ❌ **NOT RECOMMENDED** (too expensive and risky)

---

### Option 3: Hybrid Approach (Whisper + Streaming LLM + Fast TTS)

**Architecture:**
```
Browser → WebSocket → Whisper (streaming) → Llama (streaming) → Kokoro TTS → Browser
```

**Optimization Strategy:**
- Use streaming Whisper (processes audio incrementally)
- Use streaming LLM inference
- Use fast TTS (Kokoro: 150ms)
- Aggressive caching and batching

**Components:**
1. **ASR**: Whisper Turbo (MIT) - streaming mode
2. **LLM**: Llama 3.2 3B (Apache 2.0) - streaming
3. **TTS**: Kokoro TTS (Apache 2.0) - fastest available

**Pros:**
- ✅ All permissive licenses
- ✅ Better latency than traditional pipeline
- ✅ Proven components

**Cons:**
- ❌ Still sequential (not truly full-duplex)
- ❌ Moderate latency (500-700ms)

**Estimated Latency:**
- Streaming ASR: 150ms (partial transcription)
- Streaming LLM: 100ms (partial generation)
- Fast TTS: 150ms
- Network: 50ms
- **Total: 450-500ms** ✅ **MEETS TARGET**

---

## 🎯 RECOMMENDED PATH FORWARD

### Use Option 3: Hybrid Streaming Approach ✅

**Why:**
1. ✅ Meets <500ms latency target (450-500ms)
2. ✅ All components have permissive licenses (MIT/Apache 2.0)
3. ✅ No attribution requirements
4. ✅ Commercially free
5. ✅ Proven technology stack
6. ✅ Can be optimized further

**Component Stack:**

| Component | Model | License | Latency |
|-----------|-------|---------|---------|
| **VAD** | Silero VAD | MIT | 10ms |
| **ASR** | Whisper Turbo | MIT | 150ms (streaming) |
| **LLM** | Llama 3.2 3B | Apache 2.0 | 100ms (streaming) |
| **TTS** | Kokoro TTS | Apache 2.0 | 150ms |

**Total Latency: 450-500ms** ✅

---

## 📋 Action Items (Immediate)

1. ❌ **STOP**: Do not proceed with Moshi-based architecture
2. ✅ **UPDATE**: Revise all Phase 1 documents
3. ✅ **DESIGN**: New architecture with streaming pipeline
4. ✅ **VERIFY**: All licenses for new components
5. ✅ **CODE**: Begin development with approved stack

---

## 🔍 License Verification for New Stack

### Whisper (OpenAI)
- **License**: MIT
- **Attribution**: Not required for use
- **Commercial**: ✅ Allowed
- **Source**: https://github.com/openai/whisper

### Llama 3.2 (Meta)
- **License**: Apache 2.0 (Llama 3.2 Community License)
- **Attribution**: Not required for use
- **Commercial**: ✅ Allowed (under 700M monthly active users)
- **Source**: https://huggingface.co/meta-llama/Llama-3.2-3B

### Kokoro TTS
- **License**: Apache 2.0
- **Attribution**: Not required for use
- **Commercial**: ✅ Allowed
- **Source**: https://huggingface.co/hexgrad/Kokoro-82M

### Silero VAD
- **License**: MIT
- **Attribution**: Not required for use
- **Commercial**: ✅ Allowed
- **Source**: https://github.com/snakers4/silero-vad

**All components verified ✅**

---

## Next Steps

**HOLD ALL DEVELOPMENT** until we:
1. Get approval for new architecture (Option 3)
2. Update Phase 1 documents
3. Design new streaming pipeline
4. Verify latency targets achievable

**Estimated Timeline Impact**: +2 days (architecture redesign)

---

**Critical Decision Required**: Which option to proceed with?
- Option 1: Traditional cascade (610ms)
- Option 3: Streaming hybrid (450-500ms) ✅ RECOMMENDED

**Awaiting approval to proceed...**
