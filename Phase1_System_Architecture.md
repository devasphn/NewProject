# Phase 1: System Architecture
## Ultra-Low Latency Telugu S2S Voice Agent

---

## 1. Complete System Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                          USER BROWSER                                  │
│  ┌──────────────┐         ┌──────────────────┐                       │
│  │ Microphone   │────────▶│  Audio Capture   │                       │
│  │ (16kHz PCM)  │         │  (getUserMedia)  │                       │
│  └──────────────┘         └────────┬─────────┘                       │
│                                     │                                  │
│                                     ▼                                  │
│                          ┌──────────────────────┐                     │
│                          │ AudioWorklet         │                     │
│                          │ (Low-Latency Buffer) │                     │
│                          └──────────┬───────────┘                     │
│                                     │                                  │
│                                     ▼                                  │
│                          ┌──────────────────────┐                     │
│                          │ WebSocket Client     │                     │
│                          │ (Binary Frames)      │                     │
│                          └──────────┬───────────┘                     │
│                                     │                                  │
│  ┌───────────────────────────────────────────────────────┐           │
│  │  Latency Monitor: Display RTT + Processing Time       │           │
│  └───────────────────────────────────────────────────────┘           │
└───────────────────────────────┼───────────────────────────────────────┘
                                │
                Network (30-50ms roundtrip)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    RUNPOD SERVER (L4 GPU - 24GB)                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              FastAPI WebSocket Server                        │    │
│  │                                                               │    │
│  │  ┌──────────────┐       ┌─────────────────┐                │    │
│  │  │Audio Buffer  │──────▶│ Silero VAD      │ (Optional)     │    │
│  │  │(20ms queue)  │       │ (CPU: 5-10ms)   │                │    │
│  │  └──────────────┘       └────────┬────────┘                │    │
│  │                                   │                          │    │
│  │                                   ▼                          │    │
│  │                         ┌─────────────────────────┐         │    │
│  │                         │   Moshi S2S Model       │         │    │
│  │                         │   (GPU Inference)       │         │    │
│  │                         │                         │         │    │
│  │                         │  ┌──────────────────┐  │         │    │
│  │                         │  │ Mimi Encoder     │  │         │    │
│  │                         │  │ (80ms latency)   │  │         │    │
│  │                         │  │ 16kHz → Tokens   │  │         │    │
│  │                         │  └────────┬─────────┘  │         │    │
│  │                         │           │            │         │    │
│  │                         │           ▼            │         │    │
│  │                         │  ┌──────────────────┐  │         │    │
│  │                         │  │ Temporal Trans.  │  │         │    │
│  │                         │  │ (7B Parameters)  │  │         │    │
│  │                         │  │ ~40-60ms         │  │         │    │
│  │                         │  └────────┬─────────┘  │         │    │
│  │                         │           │            │         │    │
│  │                         │           ▼            │         │    │
│  │                         │  ┌──────────────────┐  │         │    │
│  │                         │  │ Inner Monologue  │  │         │    │
│  │                         │  │ (Text Tokens)    │  │         │    │
│  │                         │  └────────┬─────────┘  │         │    │
│  │                         │           │            │         │    │
│  │                         │           ▼            │         │    │
│  │                         │  ┌──────────────────┐  │         │    │
│  │                         │  │ Depth Trans.     │  │         │    │
│  │                         │  │ ~20-30ms         │  │         │    │
│  │                         │  └────────┬─────────┘  │         │    │
│  │                         │           │            │         │    │
│  │                         │           ▼            │         │    │
│  │                         │  ┌──────────────────┐  │         │    │
│  │                         │  │ Mimi Decoder     │  │         │    │
│  │                         │  │ (80ms latency)   │  │         │    │
│  │                         │  │ Tokens → 24kHz   │  │         │    │
│  │                         │  └────────┬─────────┘  │         │    │
│  │                         └───────────┼─────────────┘         │    │
│  │                                     │                       │    │
│  │                                     ▼                       │    │
│  │                          ┌──────────────────┐              │    │
│  │                          │ Audio Streaming  │              │    │
│  │                          │ (24kHz Output)   │              │    │
│  │                          └────────┬─────────┘              │    │
│  └──────────────────────────────────┼──────────────────────────┘    │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                Network (30-50ms roundtrip)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          USER BROWSER                                  │
│                         ┌──────────────────┐                          │
│                         │ WebSocket Client │                          │
│                         │ (Receive Audio)  │                          │
│                         └────────┬─────────┘                          │
│                                  │                                     │
│                                  ▼                                     │
│                         ┌──────────────────┐                          │
│                         │ Audio Buffer     │                          │
│                         │ (Minimal: 10ms)  │                          │
│                         └────────┬─────────┘                          │
│                                  │                                     │
│                                  ▼                                     │
│                         ┌──────────────────┐                          │
│                         │ Web Audio API    │                          │
│                         │ (AudioContext)   │                          │
│                         └────────┬─────────┘                          │
│                                  │                                     │
│                                  ▼                                     │
│                         ┌──────────────────┐                          │
│                         │   Speakers       │                          │
│                         └──────────────────┘                          │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Latency Breakdown (Optimized Configuration)

### 2.1 Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE                    │ LATENCY    │ CUMULATIVE          │
├─────────────────────────────────────────────────────────────┤
│ 1. User Speaks           │ 0ms        │ 0ms                 │
│ 2. Browser Audio Capture │ 10ms       │ 10ms                │
│ 3. AudioWorklet Buffer   │ 10ms       │ 20ms                │
│ 4. WebSocket Serialize   │ 2ms        │ 22ms                │
│ 5. Network Upload        │ 20ms       │ 42ms                │
│ 6. Server WS Receive     │ 2ms        │ 44ms                │
│ 7. Server Audio Queue    │ 5ms        │ 49ms                │
│ 8. Silero VAD (optional) │ 0ms*       │ 49ms                │
│ 9. Moshi Processing:     │            │                     │
│    ├─ Mimi Encoder       │ 80ms       │ 129ms               │
│    ├─ Temporal Trans.    │ 50ms       │ 179ms               │
│    ├─ Inner Monologue    │ 10ms       │ 189ms               │
│    ├─ Depth Trans.       │ 25ms       │ 214ms               │
│    └─ Mimi Decoder       │ 80ms       │ 294ms               │
│ 10. Server Audio Stream  │ 2ms        │ 296ms               │
│ 11. Network Download     │ 20ms       │ 316ms               │
│ 12. Browser WS Receive   │ 2ms        │ 318ms               │
│ 13. Audio Buffer Queue   │ 10ms       │ 328ms               │
│ 14. Web Audio Playback   │ 12ms       │ 340ms               │
├─────────────────────────────────────────────────────────────┤
│ TOTAL END-TO-END LATENCY │            │ 340ms ✅            │
└─────────────────────────────────────────────────────────────┘

* Silero VAD runs in parallel on CPU, doesn't block Moshi
```

**Target**: <500ms  
**Achieved**: 340ms  
**Margin**: 160ms (32% buffer for network jitter)

### 2.2 Best Case vs Worst Case

| Scenario | Latency | Notes |
|----------|---------|-------|
| **Best Case** (Local network, A40) | 260ms | Low network latency (20ms total), A40 GPU (180ms Moshi) |
| **Optimal Case** (Good network, L4) | 340ms | Baseline configuration above |
| **Typical Case** (Average network, L4) | 380ms | Network spikes to 60ms roundtrip |
| **Worst Case** (Poor network, L4) | 480ms | Network degradation to 100ms roundtrip |

**All scenarios meet <500ms target** ✅

---

## 3. Data Flow: Browser → WebSocket → Moshi → Browser

### 3.1 Audio Format Specifications

#### Input (Browser → Server)
```javascript
{
  sampleRate: 16000,      // 16kHz (Moshi compatible)
  channelCount: 1,        // Mono
  bitDepth: 16,           // 16-bit PCM
  frameDuration: 20,      // 20ms chunks (320 samples)
  encoding: 'PCM',        // Uncompressed
  byteOrder: 'little-endian'
}
```

**Bandwidth**: 16000 Hz × 16 bits × 1 channel = 256 kbps

#### Output (Server → Browser)
```javascript
{
  sampleRate: 24000,      // 24kHz (Mimi output)
  channelCount: 1,        // Mono
  bitDepth: 16,           // 16-bit PCM
  frameDuration: 20,      // 20ms chunks
  encoding: 'PCM',        // Or compressed
  byteOrder: 'little-endian'
}
```

**Bandwidth**: 24000 Hz × 16 bits × 1 channel = 384 kbps

**Total Bandwidth (Full-Duplex)**: ~640 kbps (0.64 Mbps)

### 3.2 WebSocket Protocol

```
Client → Server (Audio Stream):
┌─────────────────────────────────────┐
│ WebSocket Frame (Binary)            │
├─────────────────────────────────────┤
│ Header: 0x01 (audio data)           │
│ Length: 640 bytes (20ms @ 16kHz)    │
│ Payload: PCM samples                │
└─────────────────────────────────────┘

Server → Client (Audio Response):
┌─────────────────────────────────────┐
│ WebSocket Frame (Binary)            │
├─────────────────────────────────────┤
│ Header: 0x02 (audio response)       │
│ Length: 960 bytes (20ms @ 24kHz)    │
│ Payload: PCM samples                │
└─────────────────────────────────────┘

Control Messages:
┌─────────────────────────────────────┐
│ WebSocket Frame (Text/JSON)         │
├─────────────────────────────────────┤
│ {"type": "latency", "ms": 340}      │
│ {"type": "error", "msg": "..."}     │
│ {"type": "session_start"}           │
│ {"type": "session_end"}             │
└─────────────────────────────────────┘
```

---

## 4. Component-Level Design

### 4.1 Browser Client Architecture

**File**: `index.html` (single-file for POC)

**Key Components**:
1. **Audio Capture** (getUserMedia)
2. **AudioWorklet** (low-latency processing)
3. **WebSocket Client** (binary streaming)
4. **Audio Playback** (Web Audio API)
5. **UI** (latency monitor, controls)

**Configuration**:
```javascript
// Audio capture constraints
const constraints = {
  audio: {
    channelCount: 1,
    sampleRate: 16000,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    latency: 0.01  // 10ms target
  }
};

// AudioContext for playback
const audioContext = new AudioContext({
  latencyHint: 'interactive',  // Lowest latency
  sampleRate: 24000
});
```

### 4.2 Backend Server Architecture

**File**: `server.py` (FastAPI + WebSocket)

**Components**:
1. **WebSocket Handler** (FastAPI)
2. **Audio Buffer** (asyncio.Queue)
3. **VAD Processor** (Silero, optional)
4. **Moshi Inference** (PyTorch/ONNX)
5. **Response Streamer** (async)

**Pseudo-code**:
```python
from fastapi import FastAPI, WebSocket
import torch
import asyncio

app = FastAPI()

class MoshiServer:
    def __init__(self):
        self.model = load_moshi_model()  # Load on GPU
        self.vad = load_silero_vad()      # Load on CPU
        
    async def handle_client(self, websocket: WebSocket):
        await websocket.accept()
        
        audio_queue = asyncio.Queue(maxsize=10)
        
        # Concurrent tasks
        receive_task = asyncio.create_task(
            self.receive_audio(websocket, audio_queue)
        )
        process_task = asyncio.create_task(
            self.process_and_respond(websocket, audio_queue)
        )
        
        await asyncio.gather(receive_task, process_task)
    
    async def receive_audio(self, ws, queue):
        while True:
            audio_bytes = await ws.receive_bytes()
            await queue.put(audio_bytes)
    
    async def process_and_respond(self, ws, queue):
        while True:
            audio_chunk = await queue.get()
            
            # Optional VAD filtering
            if self.vad.is_speech(audio_chunk):
                # Process with Moshi (streaming)
                output_audio = await self.model.infer_stream(audio_chunk)
                
                # Send immediately
                await ws.send_bytes(output_audio)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    server = MoshiServer()
    await server.handle_client(websocket)
```

---

## 5. Scaling Strategy

### 5.1 Single Pod Capacity

**L4 GPU (24GB VRAM)**:
- Concurrent Sessions: 8-12
- Per-session VRAM: ~2GB
- CPU: 4-8 cores (for VAD, networking)

**A40 GPU (48GB VRAM)**:
- Concurrent Sessions: 15-20
- Per-session VRAM: ~2.5GB
- CPU: 8-16 cores

### 5.2 Horizontal Scaling Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │  (HAProxy/Nginx)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ RunPod Pod 1  │    │ RunPod Pod 2  │    │ RunPod Pod N  │
│   L4 GPU      │    │   L4 GPU      │    │   L4 GPU      │
│ 8-12 sessions │    │ 8-12 sessions │    │ 8-12 sessions │
└───────────────┘    └───────────────┘    └───────────────┘
```

**Load Balancing Strategy**:
- WebSocket sticky sessions (consistent routing per user)
- Health checks (every 30s)
- Auto-scaling based on connection count

### 5.3 Scaling Math

```
Target: 100 concurrent users

L4 Pods Needed: 100 / 10 (avg) = 10 pods
A40 Pods Needed: 100 / 17 (avg) = 6 pods

Cost Comparison (Spot Pricing):
L4:  10 pods × $0.39/hour = $3.90/hour = $2,808/month
A40: 6 pods × $0.69/hour = $4.14/hour = $2,980/month

Verdict: L4 slightly cheaper at 100 users
```

---

## 6. Optimization Techniques

### 6.1 Model Optimization

**FP16 Mixed Precision**:
```python
model = load_moshi_model()
model = model.half()  # Convert to FP16
# Reduces memory by 50%, speeds up inference by 30-40%
```

**ONNX Export** (Optional):
```python
# Convert to ONNX for faster inference
torch.onnx.export(model, dummy_input, "moshi.onnx")
# Use ONNX Runtime with TensorRT for 20-30% speedup
```

### 6.2 Network Optimization

**WebSocket Compression** (Optional):
```python
# Trade CPU for bandwidth
websocket_config = {
    "compression": "deflate",  # Reduce bandwidth by 60-70%
    "compression_level": 6     # Balanced
}
# Adds 5-10ms CPU overhead
```

**Binary Protocol**:
- Use binary frames (already planned)
- Avoid JSON for audio data
- Minimal headers

### 6.3 Buffer Optimization

**Adaptive Buffering**:
```python
# Adjust buffer size based on network conditions
if avg_latency < 50ms:
    buffer_size = 10ms  # Aggressive
elif avg_latency < 100ms:
    buffer_size = 20ms  # Balanced
else:
    buffer_size = 50ms  # Safe
```

---

## 7. Error Handling & Reliability

### 7.1 Network Failures

**Reconnection Logic**:
```javascript
// Client-side
function connectWebSocket() {
  const ws = new WebSocket(SERVER_URL);
  
  ws.onerror = () => {
    console.log("Connection failed, retrying in 3s...");
    setTimeout(connectWebSocket, 3000);
  };
  
  ws.onclose = () => {
    console.log("Connection closed, reconnecting...");
    setTimeout(connectWebSocket, 1000);
  };
}
```

### 7.2 GPU Out-of-Memory

**Graceful Degradation**:
```python
try:
    output = model.infer(audio)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Clear cache and reject new connections
        torch.cuda.empty_cache()
        return error_response("Server at capacity")
```

### 7.3 Audio Quality Monitoring

**Real-time Metrics**:
- Signal-to-Noise Ratio (SNR)
- Packet loss detection
- Latency spikes

---

## Next Steps

✅ Architecture designed  
✅ Latency breakdown completed  
✅ Component specifications defined  
⏭️ Training plan (next document)  
⏭️ GPU selection and cost analysis (next document)
