# Project: Ultra-Low Latency Telugu Speech-to-Speech AI Voice Agent

## Project Overview
Build a production-ready, real-time Speech-to-Speech AI Voice Agent with **<500ms total latency** featuring fluent Telugu accent with natural emotional speech. This must surpass Luna AI (by Pixa AI) in performance and architecture innovation.

## Core Requirements

### Performance Targets
- **Total Latency**: <500ms (user speech → response audio playback)
- **Architecture Latency**: <150ms (network + processing overhead)
- **Model Inference**: <350ms (S2S model execution time)
- **Audio Quality**: Natural, emotional speech with Telugu accent
- **Voice Options**: 2 male + 2 female speaker voices

### Technical Constraints (RunPod Platform)
- **GPU Options**: A40 or L4 (need recommendation with cost-benefit analysis)
- **Network**: HTTP/TCP only (max 10 ports) - **NO UDP or WebRTC support**
- **Streaming**: Must use WebSocket with duplex streaming
- **Docker**: NOT supported - manual setup only
- **Default Mount**: `/workspace`
- **Deployment**: Single-shot configuration (no changes after pod starts)

### Directory and Git Setup
- **Directory name**: `NewProject`
- **Initialize Git**: Run `git init` inside `NewProject`
- **Pull Project Repo**: Use `git clone [repo_url]` within `NewProject` on Runpod webterminal

### Architecture Requirements
1. **Break Traditional Pipeline**: Avoid VAD→ASR→LLM→TTS cascade
2. **Duplex Streaming**: Support both stream mode and turn mode
3. **End-to-End**: Unified model architecture (speech-in, speech-out)
4. **Innovation**: Use cutting-edge architecture unseen in current market
5. **Longevity**: Design to remain relevant for 5+ years

### Commercial & Licensing Constraints
- **100% Free & Open Source**: All components must be commercially usable
- **No Attributions Required**: Can be used without credits
- **No API Costs**: Zero external API dependencies
- **Only Cost**: RunPod GPU rental + storage

## Deliverables Required

### Phase 1: Research & Architecture Design
1. **Model Architecture Research**
   - Latest end-to-end speech-to-speech models (2024-2025)
   - Alternatives to traditional VAD+ASR+LLM+TTS pipeline
   - Models supporting Telugu language
   - Commercially free models with permissive licenses
   
2. **Architecture Document**
   - Complete system architecture diagram
   - Data flow: browser → WebSocket → VAD → S2S model → audio playback
   - Latency breakdown for each component
   - Scaling strategy

3. **Model Selection & Training Plan**
   - Base model recommendation
   - Training methodology (fine-tuning vs from-scratch)
   - Dataset requirements (YouTube podcasts/videos as source)
   - GPU requirements and training time estimation
   - Cost estimation for training on RunPod

### Phase 2: RunPod Configuration
Provide complete RunPod setup specifications:

1. **Template Selection**
   - Recommended base template
   - Python version and CUDA version
   
2. **Storage Configuration**
   - Container disk space (GB)
   - Volume disk space (GB)
   - Volume mount path: `/workspace`

3. **Environment Variables**
   ```
   [List all required env vars]
   ```

4. **Container Start Command**
   ```bash
   [Complete startup command]
   ```

5. **Manual Setup Script**
   Complete bash script for RunPod web terminal covering:
   ```bash
   # System dependencies
   apt-get update && apt-get install -y [dependencies]
   
   # Python packages
   pip install [packages]
   
   # Model downloads
   [download commands]
   
   # Server startup
   [server startup commands]
   ```

### Phase 3: Application Development

#### 1. Browser Client (`index.html`)
- WebSocket audio streaming (16kHz, 16-bit PCM recommended)
- Real-time audio capture with low-latency settings
- Audio playback with minimal buffering
- Simple, functional UI for POC
- Latency monitoring display

#### 2. Backend Server (`server.py` or equivalent)
- WebSocket server (HTTP/TCP compatible)
- VAD implementation (suggest: Silero VAD or faster)
- S2S model inference pipeline
- Audio streaming protocols
- Error handling and reconnection logic

#### 3. Model Integration
- Model loading and initialization
- Inference optimization (quantization, batching if applicable)
- GPU memory management
- Streaming output generation

#### 4. Supporting Files
- `requirements.txt` with pinned versions
- `README.md` with setup instructions
- Configuration files
- Testing scripts

### Phase 4: Model Training Guide
1. **Data Collection Strategy**
   - How to extract audio from YouTube (tool recommendations)
   - Dataset size recommendations
   - Data preprocessing pipeline
   - Quality filtering criteria

2. **Training Pipeline**
   - Step-by-step training instructions
   - Hyperparameter recommendations
   - Checkpoint saving strategy
   - Validation approach

3. **Telugu Voice Cloning**
   - Speaker voice extraction method
   - Multi-speaker training approach
   - Emotional speech synthesis techniques

## Technical Specifications

### Audio Pipeline
```
User Browser → WebSocket (16kHz PCM) → VAD → S2S Model → WebSocket → Browser Audio Playback
```

### Latency Budget
- Network (roundtrip): 30-50ms
- VAD detection: 20-30ms
- S2S model inference: 300-350ms
- Audio buffering/playback: 20-30ms
- **Total**: <450-460ms

### Model Requirements
- **Input**: Raw audio waveform (16kHz recommended)
- **Output**: Audio waveform with Telugu accent
- **Features**: Emotional expression, natural prosody
- **Optimization**: INT8/FP16 quantization for speed

## Key Challenges to Solve

1. **No WebRTC**: Design WebSocket-based real-time audio streaming with <50ms network latency
2. **End-to-End Model**: Identify/design model that bypasses traditional pipeline
3. **Telugu Language**: Limited resources - creative training approach needed
4. **Emotional Speech**: Natural prosody and emotion in synthetic voice
5. **Ultra-Low Latency**: Every millisecond counts - aggressive optimization needed

## Research Focus Areas

1. **Recent Models to Investigate**
   - SpeechGPT, AudioPaLM, Seamless Communication models
   - Moshi, Parler-TTS, Piper TTS
   - Mimi codec, Encodec for audio compression
   - Any 2024-2025 speech-to-speech models

2. **Architecture Patterns**
   - Streaming transformers
   - Speculative decoding
   - Neural audio codecs
   - Duplex communication patterns

3. **Optimization Techniques**
   - Model quantization (INT8, FP16)
   - Flash Attention
   - Continuous batching
   - KV cache optimization

## Success Criteria (POC)

1. ✅ Working browser interface
2. ✅ Real-time audio streaming via WebSocket
3. ✅ <500ms total latency measurable
4. ✅ Telugu accent detection and response
5. ✅ Natural, emotional voice output
6. ✅ Runs on RunPod A40/L4 GPU
7. ✅ Zero external API dependencies
8. ✅ Deployable with provided scripts

## Development Instructions

1. **Start with Architecture**: Design first, code later
2. **Verify All Licenses**: Must be commercially free
3. **Optimize Relentlessly**: Every millisecond matters
4. **Test on RunPod**: Ensure compatibility with constraints
5. **Document Everything**: I need to understand and modify if needed
6. **Think Innovation**: Don't just copy existing solutions

## Output Format

Please provide:
1. **Architecture document** (markdown with diagrams if possible)
2. **All source code files** with detailed comments
3. **RunPod setup guide** (copy-paste ready)
4. **Training guide** with specific commands
5. **Cost estimation** (training + inference)
6. **Latency breakdown** with optimization suggestions

---

## Additional Context
- **Benchmark**: Luna AI demo by Sparsh Agrawal (Pixa AI) - find and analyze
- **Goal**: Create POC to secure MD approval for investment
- **Timeline**: ASAP for POC, production-ready in next phase
- **Budget**: Only RunPod costs, everything else must be free

**Begin with comprehensive research, then provide the complete architecture and implementation plan. Ask clarifying questions if any requirement is ambiguous.**

---

This optimized prompt provides WindSurf with:
✅ Clear project scope and constraints
✅ Specific technical requirements
✅ RunPod platform limitations
✅ Structured deliverables
✅ Research direction
✅ Success criteria
✅ Cost consciousness

The prompt is now ready for WindSurf to start systematic development!