# Quick Reference Guide
## Telugu S2S Voice Agent - Phase 1

---

## 🎯 One-Page Summary

### The Solution
**Moshi + LoRA fine-tuning on Telugu data** running on **L4 GPU** via **RunPod**

### Key Metrics
- **Latency**: 340ms (target: <500ms) ✅
- **Cost**: $27/user/month at scale
- **Training**: $19 (3-5 days)
- **License**: 100% free (Apache 2.0)

---

## 📋 Quick Facts

### Model Stack
```
Moshi (7B parameters)
├─ Mimi Codec (80ms encode + 80ms decode)
├─ Temporal Transformer (7B)
└─ Depth Transformer

Fine-tuning: LoRA (0.1-1% parameters)
```

### Data Requirements
- **Stage 1**: 150 hours Telugu speech (YouTube)
- **Stage 2**: 12 hours (4 speakers)
- **Stage 3**: 50 hours (emotions)
- **Total**: ~212 hours

### GPU Choice
- **POC/Small Scale**: L4 ($0.39/hour spot)
- **Large Scale (80+ users)**: A40 ($0.69/hour spot)

---

## 💰 Costs at a Glance

| Item | Cost |
|------|------|
| **Training** | $19 |
| **Storage** | $11/month |
| **1 L4 pod** | $281/month |
| **10 users** | $28/user/month |
| **100 users** | $25/user/month (A40) |

---

## ⚡ Latency Breakdown

```
Component               Latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audio Capture           20ms
Network Upload          20ms
Server Queue             5ms
Moshi Processing       200ms ⭐
Network Download        20ms
Audio Playback          15ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                  280ms
(with margin: 340ms)
```

---

## 🔧 Tech Stack

### Frontend
- HTML5 + JavaScript
- Web Audio API (getUserMedia)
- WebSocket (binary frames)

### Backend
- Python 3.10+
- FastAPI (WebSocket server)
- PyTorch 2.0+ (Moshi inference)
- ONNX Runtime (Silero VAD)

### Infrastructure
- RunPod (L4/A40 GPU)
- Ubuntu 22.04
- CUDA 12.1+

---

## 📦 Training Pipeline

```
1. Data Collection
   └─ yt-dlp (YouTube download)
   └─ Whisper (transcription)
   └─ pyannote (diarization)
   
2. Stage 1: Telugu Adaptation (2 days)
   └─ 150 hours dataset
   └─ LoRA fine-tuning
   └─ Cost: $12
   
3. Stage 2: Voice Cloning (0.5 days)
   └─ 12 hours (4 voices)
   └─ Speaker conditioning
   └─ Cost: $3
   
4. Stage 3: Emotions (1 day)
   └─ 50 hours labeled
   └─ Emotion conditioning
   └─ Cost: $4
   
Total: 4 days, $19
```

---

## 🚀 Deployment Flow

```
1. Setup RunPod L4 pod
2. Install dependencies
3. Download Moshi model
4. Load fine-tuned adapters
5. Start FastAPI server
6. Open port 8000 (WebSocket)
7. Deploy index.html (browser client)
```

---

## 📊 Scaling Guide

| Users | GPUs | Type | Cost/Month |
|-------|------|------|------------|
| 10 | 1 | L4 | $281 |
| 50 | 5 | L4 | $1,404 |
| 80 | 8 | L4 | $2,246 |
| 100 | 5 | A40 | $2,484 |
| 200 | 10 | A40 | $4,968 |

**Switch to A40 at 80+ users for better cost-per-user**

---

## 🎯 Decision Tree

### Should I use L4 or A40?

```
Are you in POC/testing phase?
├─ YES → L4 ✅
└─ NO → Do you have 80+ concurrent users?
    ├─ YES → A40 ✅
    └─ NO → L4 ✅
```

### Should I collect YouTube data or hire voice actors?

```
Is budget tight?
├─ YES → YouTube only ✅ ($19 total)
└─ NO → YouTube + Voice actors ✅ ($719 total)
```

### Should I add Silero VAD?

```
Do you need bandwidth optimization?
├─ YES → Add Silero ✅ (5-10ms overhead)
└─ NO → Skip it ✅ (Moshi has implicit VAD)
```

---

## 🔍 File Guide

### Must Read
1. **Phase1_Executive_Summary.md** ← Start here!
2. **README.md** ← Project overview

### Deep Dive
3. **Phase1_Model_Research.md** ← Why Moshi?
4. **Phase1_System_Architecture.md** ← How it works
5. **Phase1_Training_Plan.md** ← How to train
6. **Phase1_GPU_Analysis.md** ← L4 vs A40

### Reference
7. **QUICK_REFERENCE.md** ← This file
8. **telugu-s2s-windsurf.md** ← Original requirements

---

## ⚠️ Common Pitfalls

### ❌ Don't
- Use WebRTC VAD (too inaccurate)
- Use Seamless (non-commercial license)
- Train from scratch (100x cost)
- Deploy on CPU (too slow)
- Use A40 for <80 users (waste of money)

### ✅ Do
- Use Moshi with LoRA
- Start with L4 GPU
- Collect 150+ hours Telugu data
- Test latency with real network
- Monitor cost per user
- Scale horizontally (add pods)

---

## 🆘 Troubleshooting

### Latency >500ms?
1. Check network latency (ping test)
2. Enable FP16 on GPU
3. Reduce audio buffer sizes
4. Consider INT8 quantization
5. Upgrade to A40 if needed

### Out of Memory?
1. Reduce batch size
2. Clear CUDA cache
3. Limit concurrent sessions
4. Upgrade to A40 (48GB)

### Poor Telugu Quality?
1. Increase dataset size (200+ hours)
2. Use professional voice recordings
3. Adjust LoRA rank (try 32)
4. Train longer (10 epochs)

### Low Throughput?
1. Add more pods
2. Implement load balancing
3. Enable connection pooling
4. Optimize WebSocket config

---

## 📞 Quick Commands

### Check GPU
```bash
nvidia-smi
```

### Test Latency
```bash
ping your-runpod-instance.com
```

### Monitor VRAM
```bash
watch -n 1 nvidia-smi
```

### Check Moshi Model
```python
import torch
model = torch.load("moshi-telugu.pt")
print(f"Model loaded: {model}")
```

---

## 🎓 Key Concepts

### Full-Duplex
Both user and agent can speak simultaneously (like humans)

### LoRA
Low-Rank Adaptation - trains only 0.1-1% of parameters

### Streaming
Processes audio frame-by-frame, not entire utterance

### Mimi Codec
Neural audio codec: 24kHz → 1.1kbps → 24kHz (80ms latency)

### Inner Monologue
Moshi predicts text tokens before audio (improves quality)

---

## 📈 Success Metrics

### Technical
- [ ] Latency <500ms (target: 340ms)
- [ ] WER <10% on Telugu test set
- [ ] MOS >4.0/5.0 for voice quality
- [ ] Uptime >99.5%

### Business
- [ ] Cost per user <$30/month
- [ ] Concurrent users: 10+ (POC)
- [ ] Training cost <$100
- [ ] Time to POC <6 weeks

---

## 🔮 Future Enhancements

### Phase 2+
- Multi-language support (Hindi, Tamil)
- Real-time translation
- Background noise reduction
- Voice activity detection improvements
- Mobile app (iOS/Android)
- Edge deployment (on-device)

---

## 💡 Pro Tips

1. **Start small**: Test with 10 users before scaling
2. **Monitor everything**: Latency, VRAM, CPU, network
3. **Use spot instances**: 47% cheaper than on-demand
4. **Test real networks**: Don't rely on localhost benchmarks
5. **Collect diverse data**: Multiple speakers, accents, ages
6. **Version control**: Save checkpoints every epoch
7. **A/B test voices**: Let users choose preferred speaker
8. **Log everything**: Essential for debugging production issues

---

**Last Updated**: November 18, 2025  
**Quick Access**: Keep this page bookmarked! 📌
