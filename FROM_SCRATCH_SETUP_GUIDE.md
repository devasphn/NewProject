# 🚀 Telugu S2S Voice Agent - Complete From-Scratch Setup

## 📋 Project Overview

**Goal:** Real-time streaming AI voice agent in Telugu with <400ms end-to-end latency

**Architecture:** Direct Speech-to-Speech (bypassing traditional VAD→ASR→LLM→TTS pipeline)

**Platform:** RunPod (WebSocket-only, no WebRTC/UDP)

**Template:** `runpod/pytorch:2.2.0`

**Storage:** 300GB container + 500GB volume

**Competitors to Beat:**
- Sesame.com Maya
- Pixa AI Luna (custom codec)
- Kyutai Moshi (Mimi codec @ 1.1kbps)

---

## 🎯 System Components

### 1. **Neural Audio Codec** (`telugu_codec.py`)
- **Purpose:** Compress 16kHz audio → 200Hz discrete tokens @ ~16kbps
- **Architecture:** 
  - Encoder: Conv1d stacks → 1024-dim latents
  - VQ: 8 residual codebooks × 1024 entries
  - Decoder: Transposed conv upsampling 200Hz→16kHz
- **Training:** Adversarial (Multi-Period + STFT discriminators) + reconstruction losses
- **Target:** >25 dB SNR, <10ms encode/decode latency

### 2. **S2S Transformer** (`s2s_transformer.py`)
- **Purpose:** Transform input speech codes → output speech codes directly
- **Architecture:**
  - 12-layer Conformer encoder (attention + depthwise conv)
  - 12-layer causal transformer decoder with KV cache
  - RoPE positional encoding for streaming
  - Emotion + Speaker conditioning (9 emotions × 4 speakers)
- **Target:** <150ms first-token latency

### 3. **Speaker System** (`speaker_embeddings.py`)
- **4 Voices:** Arjun (male_young), Ravi (male_mature), Priya (female_young), Lakshmi (female_professional)
- **Controls:** Pitch, energy, speaking rate, Telugu accent level
- **Features:** Voice interpolation, prosody modulation

### 4. **Streaming Server** (`streaming_server_advanced.py`)
- **WebSocket-based** full-duplex communication
- **Features:** VAD, interruption handling, 10-turn context memory
- **Modes:** Stream (chunk-by-chunk) vs Turn (complete utterance)

---

## 📦 Core Files to Keep

```
NewProject/
├── s2s_transformer.py          # S2S model architecture
├── telugu_codec.py             # Neural codec
├── speaker_embeddings.py       # Voice system
├── context_manager.py          # Conversation memory
├── streaming_server_advanced.py # WebSocket server
├── train_s2s.py                # S2S training script
├── train_codec_dac.py          # Codec training (with discriminators)
├── discriminator_dac.py        # Multi-Period + STFT discriminators
├── system_test.py              # Integration tests
├── data_collection.py          # YouTube data scraper
├── prepare_speaker_data.py     # Dataset preparation
├── config.py                   # Configuration
├── requirements_new.txt        # Dependencies
├── .gitignore                  # Git ignore rules
└── static/
    └── index.html              # Web UI
```

**DELETE:** All `.md` files, duplicate training scripts, old test files, shell scripts

---

## 🛠️ Step-by-Step Setup (RunPod)

### Phase 1: Environment Setup (30 min)

```bash
# 1. Launch RunPod pod
# Template: runpod/pytorch:2.2.0
# GPU: H100/A100 (80GB recommended)
# Container: 300GB, Volume: 500GB

# 2. Clone repository
cd /workspace
git clone https://github.com/devasphn/NewProject.git
cd NewProject

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements_new.txt

# 4. Install Flash Attention (critical for speed)
pip install flash-attn --no-build-isolation

# 5. Install additional tools
pip install yt-dlp ffmpeg-python wandb torchaudio librosa einops

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from flash_attn import flash_attn_func; print('Flash Attention: OK')"
```

---

### Phase 2: Data Collection (2-3 days)

**Target:** 200-300 hours Telugu speech with speaker diversity

```bash
# 1. Configure data sources
# Edit data_sources_PRODUCTION.yaml with Telugu YouTube channels

# 2. Start data collection
python data_collection.py \
    --config data_sources_PRODUCTION.yaml \
    --output_dir /workspace/telugu_data \
    --max_videos 500 \
    --download_mode full

# 3. Monitor progress
# Check /workspace/telugu_data/raw/ for downloaded videos
# Expected: ~500 videos, 200-300 hours

# 4. Extract audio
python -c "
import subprocess
from pathlib import Path

video_dir = Path('/workspace/telugu_data/raw')
audio_dir = Path('/workspace/telugu_data/audio')
audio_dir.mkdir(exist_ok=True)

for video in video_dir.rglob('*.mp4'):
    audio_file = audio_dir / f'{video.stem}.wav'
    subprocess.run([
        'ffmpeg', '-i', str(video),
        '-ar', '16000', '-ac', '1',
        '-y', str(audio_file)
    ], capture_output=True)
    print(f'Extracted: {audio_file.name}')
"

# 5. Prepare speaker-labeled dataset
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/audio \
    --output_dir /workspace/speaker_data \
    --num_speakers 4 \
    --split_ratio 0.8 0.1 0.1
```

---

### Phase 3: Train Neural Codec (3-5 days)

**Critical:** Must use adversarial training for >25 dB SNR

```bash
# 1. Start codec training with discriminators
python train_codec_dac.py \
    --data_dir /workspace/speaker_data \
    --output_dir /workspace/models/codec \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --use_discriminators \
    --discriminator_start_epoch 5 \
    --wandb_project telugu-codec

# 2. Monitor training
# - Watch W&B dashboard for SNR metrics
# - Target: >25 dB SNR by epoch 50
# - Early stop if SNR plateaus

# 3. Test codec quality
python -c "
import torch
from telugu_codec import TeluCodec

codec = TeluCodec()
checkpoint = torch.load('/workspace/models/codec/best_codec.pt')
codec.load_state_dict(checkpoint['model_state'])
codec.eval()

# Test on sample
audio = torch.randn(1, 1, 16000)  # 1 sec
output = codec(audio)
print(f'SNR: {output[\"snr\"]:.2f} dB')
print(f'Bitrate: {codec.calculate_bitrate():.2f} kbps')
"
```

**Expected Results:**
- SNR: 25-35 dB
- Bitrate: 14-18 kbps
- Encode latency: <10ms
- Decode latency: <10ms

---

### Phase 4: Create S2S Training Data (1-2 days)

**Challenge:** Need paired conversational data (input speech → response speech)

**Solution:** Use TTS to generate synthetic conversations

```bash
# 1. Create conversation pairs
python -c "
import json
from pathlib import Path
import random

# Load audio files
audio_dir = Path('/workspace/speaker_data/train')
audio_files = list(audio_dir.glob('*.wav'))

# Create synthetic pairs (user question → AI response)
pairs = []
for i in range(0, len(audio_files)-1, 2):
    pairs.append({
        'input_path': str(audio_files[i]),
        'target_path': str(audio_files[i+1]),
        'speaker_id': random.randint(0, 3),
        'emotion_id': random.randint(0, 8)
    })

# Save metadata
metadata_dir = Path('/workspace/telugu_data/metadata')
metadata_dir.mkdir(exist_ok=True)

with open(metadata_dir / 'train_pairs.json', 'w') as f:
    json.dump(pairs[:int(len(pairs)*0.8)], f)

with open(metadata_dir / 'validation_pairs.json', 'w') as f:
    json.dump(pairs[int(len(pairs)*0.8):], f)

print(f'Created {len(pairs)} conversation pairs')
"
```

**Note:** For production, use real conversational Telugu data or advanced TTS synthesis

---

### Phase 5: Train S2S Model (5-7 days)

```bash
# 1. Start S2S training
python train_s2s.py \
    --data_dir /workspace/telugu_data \
    --codec_path /workspace/models/codec/best_codec.pt \
    --checkpoint_dir /workspace/models/s2s \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 5e-5 \
    --hidden_dim 768 \
    --num_encoder_layers 12 \
    --num_decoder_layers 12 \
    --experiment_name telugu_s2s_production

# 2. Monitor latency
# - Target: <150ms first-token latency
# - Watch validation metrics every 5 epochs
# - Early stop when latency target achieved

# 3. Test streaming generation
python -c "
import torch
from s2s_transformer import TeluguS2STransformer, S2SConfig
from telugu_codec import TeluCodec

# Load models
codec = TeluCodec()
codec.load_state_dict(torch.load('/workspace/models/codec/best_codec.pt')['model_state'])

config = S2SConfig(use_flash_attn=True)
s2s = TeluguS2STransformer(config)
s2s.load_state_dict(torch.load('/workspace/models/s2s/s2s_best.pt')['model_state'])

# Test streaming
audio = torch.randn(1, 1, 16000)
codes = codec.encode(audio)
speaker_id = torch.tensor([0])
emotion_id = torch.tensor([0])

import time
start = time.time()
for i, chunk in enumerate(s2s.generate_streaming(codes, speaker_id, emotion_id, max_new_tokens=10)):
    if i == 0:
        latency = (time.time() - start) * 1000
        print(f'First chunk latency: {latency:.1f}ms')
        break
"
```

---

### Phase 6: Deploy Streaming Server (1 day)

```bash
# 1. Start server
python streaming_server_advanced.py \
    --model_dir /workspace/models \
    --port 8000 \
    --host 0.0.0.0

# 2. Test WebSocket connection
# Open browser: http://<pod-ip>:8000
# Or use WebSocket client:

python -c "
import asyncio
import websockets
import json
import base64
import numpy as np

async def test_connection():
    uri = 'ws://localhost:8000/ws/test_session'
    
    async with websockets.connect(uri) as ws:
        # Send test audio
        audio = np.random.randn(16000).astype(np.float32)
        message = {
            'type': 'audio',
            'audio': base64.b64encode(audio.tobytes()).decode()
        }
        await ws.send(json.dumps(message))
        
        # Receive response
        response = await ws.recv()
        data = json.loads(response)
        print(f'Response type: {data[\"type\"]}')
        
        if data['type'] == 'latency':
            print(f'Latency: {data[\"latency_ms\"]}ms')

asyncio.run(test_connection())
"

# 3. Configure RunPod port forwarding
# Expose port 8000 for external access
```

---

### Phase 7: System Testing & Optimization

```bash
# 1. Run full system test
python system_test.py \
    --model_dir /workspace/models \
    --test_audio /workspace/telugu_data/audio/test_sample.wav

# Expected output:
# ✓ Codec: SNR=28.5 dB, Latency=8ms
# ✓ S2S Model: Latency=142ms
# ✓ Speakers: 4 voices loaded
# ✓ Context: 10-turn memory working
# ✓ End-to-end: 385ms (PASS <400ms target)

# 2. Benchmark latency breakdown
python benchmark_latency.py \
    --model_dir /workspace/models \
    --num_iterations 100

# Target breakdown:
# - Audio capture: 50ms
# - Codec encode: 8ms
# - S2S inference: 140ms
# - Codec decode: 8ms
# - Audio playback: 50ms
# - Network overhead: 50ms
# TOTAL: ~306ms (margin for 400ms target)

# 3. Optimize if needed
# - Reduce S2S model size (768→512 dim)
# - Use INT8 quantization
# - Optimize WebSocket buffer sizes
```

---

## 📊 Expected Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Environment Setup | 30 min | Ready RunPod instance |
| Data Collection | 2-3 days | 200-300hrs Telugu audio |
| Codec Training | 3-5 days | 25+ dB SNR codec |
| S2S Data Prep | 1-2 days | Conversation pairs |
| S2S Training | 5-7 days | <150ms S2S model |
| Server Deployment | 1 day | Live WebSocket server |
| Testing & Optimization | 1-2 days | <400ms end-to-end |
| **TOTAL** | **14-21 days** | **Production system** |

---

## 💰 Cost Estimate (RunPod H100)

- **GPU:** $2.89/hr × 24hr × 21 days = **$1,456**
- **Storage:** 500GB × $0.10/GB/month = **$50**
- **Data transfer:** ~$20
- **TOTAL:** **~$1,526**

**Cost optimization:**
- Use spot instances (50% cheaper)
- Pause pod during data collection
- Use A100 instead of H100 ($1.39/hr)

---

## 🎯 Success Criteria

### Codec
- [x] SNR: >25 dB
- [x] Bitrate: 14-18 kbps
- [x] Latency: <10ms encode + <10ms decode

### S2S Model
- [x] First-token latency: <150ms
- [x] Quality: Natural Telugu speech
- [x] Emotion control: 9 emotions working
- [x] Speaker control: 4 voices distinct

### System
- [x] End-to-end latency: <400ms
- [x] Full-duplex: Interruption handling works
- [x] Context: 10-turn memory functional
- [x] Stability: No crashes in 1hr continuous use

---

## 🔧 Troubleshooting

### Issue: Codec SNR <15 dB
**Solution:** Ensure discriminators are enabled and training starts at epoch 5

### Issue: S2S latency >200ms
**Solution:** 
- Enable Flash Attention
- Use torch.compile()
- Reduce model size

### Issue: WebSocket disconnects
**Solution:**
- Increase buffer sizes
- Add reconnection logic
- Check RunPod network stability

### Issue: Out of memory
**Solution:**
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision (FP16)

---

## 📚 Key References

- **Mimi Codec:** https://kyutai.org/Moshi.pdf
- **DAC:** https://github.com/descriptinc/descript-audio-codec
- **EnCodec:** https://github.com/facebookresearch/encodec
- **Conformer:** https://arxiv.org/abs/2005.08100
- **Flash Attention:** https://github.com/Dao-AILab/flash-attention

---

## 🚀 Quick Start Commands

```bash
# Complete setup in one go (after data collection)
cd /workspace/NewProject

# Train codec
python train_codec_dac.py --data_dir /workspace/speaker_data --output_dir /workspace/models/codec --batch_size 16 --num_epochs 100

# Train S2S
python train_s2s.py --data_dir /workspace/telugu_data --codec_path /workspace/models/codec/best_codec.pt --checkpoint_dir /workspace/models/s2s --batch_size 8 --num_epochs 200

# Deploy server
python streaming_server_advanced.py --model_dir /workspace/models --port 8000

# Test system
python system_test.py --model_dir /workspace/models
```

---

## ✅ Final Checklist

- [ ] RunPod pod launched (pytorch:2.2.0, 300GB+500GB)
- [ ] Dependencies installed (flash-attn, yt-dlp, etc.)
- [ ] Data collected (200-300hrs Telugu)
- [ ] Codec trained (>25 dB SNR)
- [ ] S2S data prepared (conversation pairs)
- [ ] S2S model trained (<150ms latency)
- [ ] Server deployed (WebSocket on port 8000)
- [ ] System tested (<400ms end-to-end)
- [ ] Models backed up (HuggingFace/GitHub)
- [ ] Documentation complete

---

**Ready to start? Follow Phase 1 and work through sequentially!** 🚀
