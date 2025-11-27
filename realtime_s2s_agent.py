#!/usr/bin/env python3
"""
Real-time Speech-to-Speech Agent using YOUR TRAINED MODELS
- Telugu Codec (best_codec.pt) - Audio encode/decode
- S2S Transformer (s2s_best.pt) - Audio code transformation

This is the CORRECT architecture for <400ms latency!
NO ASR, NO LLM, NO TTS - Direct audio-to-audio!
"""

import torch
import numpy as np
import time
import logging
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class S2SConfig:
    codec_path: str = "best_codec.pt"
    s2s_path: str = "s2s_best.pt"
    sample_rate: int = 16000
    chunk_size: int = 4096  # ~256ms chunks
    device: str = "cuda"


class TeluguCodec:
    """Your trained Telugu Codec for audio encoding/decoding"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        logger.info(f"📥 Loading Telugu Codec: {checkpoint_path}")
        
        # Import your codec
        from telugu_codec_fixed import TeluCodec
        
        self.codec = TeluCodec().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'codec_state_dict' in checkpoint:
            self.codec.load_state_dict(checkpoint['codec_state_dict'])
        else:
            self.codec.load_state_dict(checkpoint)
        
        self.codec.eval()
        
        # Warmup
        with torch.no_grad():
            dummy = torch.randn(1, 1, 16000).to(self.device)
            _ = self.codec.encode(dummy)
        
        logger.info("✅ Codec ready!")
    
    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> tuple:
        """Encode audio to codes"""
        start = time.perf_counter()
        
        # Prepare audio tensor
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # [1, 1, T]
        
        # Encode
        codes = self.codec.encode(audio_tensor)
        
        latency = (time.perf_counter() - start) * 1000
        return codes, latency
    
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> tuple:
        """Decode codes to audio"""
        start = time.perf_counter()
        
        audio = self.codec.decode(codes)
        audio_np = audio.squeeze().cpu().numpy()
        
        latency = (time.perf_counter() - start) * 1000
        return audio_np, latency


class S2STransformer:
    """Your trained S2S Transformer for code transformation"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        logger.info(f"📥 Loading S2S Transformer: {checkpoint_path}")
        
        # Import S2S model
        from s2s_transformer import TeluguS2STransformer, S2SConfig as ModelConfig
        
        # Create model with same config as training
        config = ModelConfig(
            hidden_dim=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_quantizers=8,
            vocab_size=1024
        )
        
        self.model = TeluguS2STransformer(config).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info("✅ S2S Transformer ready!")
    
    @torch.no_grad()
    def transform(self, input_codes: torch.Tensor, speaker_id: int = 0, emotion_id: int = 0) -> tuple:
        """Transform input codes to output codes"""
        start = time.perf_counter()
        
        # Prepare inputs
        if input_codes.dim() == 2:
            input_codes = input_codes.unsqueeze(0)  # [1, Q, T]
        
        input_codes = input_codes.to(self.device)
        
        # Create speaker/emotion tensors
        batch_size = input_codes.shape[0]
        speaker = torch.tensor([speaker_id] * batch_size, device=self.device)
        emotion = torch.tensor([emotion_id] * batch_size, device=self.device)
        
        # Generate output codes
        # Use the model's generate method if available, otherwise forward
        try:
            output_codes = self.model.generate_streaming(
                input_codes,
                speaker_id=speaker,
                emotion_id=emotion,
                max_len=input_codes.shape[-1]
            )
        except:
            # Fallback to simple forward pass
            output_codes = self.model(
                input_codes,
                input_codes,  # teacher forcing with same codes
                speaker_id=speaker,
                emotion_id=emotion
            )
            # Get argmax if output is logits
            if output_codes.dim() == 4:  # [B, Q, T, vocab]
                output_codes = output_codes.argmax(dim=-1)
        
        latency = (time.perf_counter() - start) * 1000
        return output_codes, latency


class RealtimeS2SAgent:
    """
    Real-time Speech-to-Speech Agent
    Uses YOUR trained codec and S2S model for <400ms latency!
    """
    
    def __init__(self, config: S2SConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        logger.info("=" * 60)
        logger.info("🎤 REAL-TIME S2S AGENT (Your Models!)")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        
        # Load YOUR trained models
        self.codec = TeluguCodec(config.codec_path, str(self.device))
        
        # Check if S2S model exists
        if Path(config.s2s_path).exists():
            self.s2s = S2STransformer(config.s2s_path, str(self.device))
            self.has_s2s = True
        else:
            logger.warning(f"⚠️ S2S model not found: {config.s2s_path}")
            logger.warning("Running in CODEC-ONLY mode (echo with reconstruction)")
            self.s2s = None
            self.has_s2s = False
        
        logger.info("=" * 60)
        logger.info("✅ Agent Ready!")
        logger.info("=" * 60)
    
    def process(self, audio: np.ndarray) -> dict:
        """
        Process audio through YOUR models:
        Audio → Codec Encode → [S2S Transform] → Codec Decode → Audio
        """
        result = {
            "audio": None,
            "sample_rate": self.config.sample_rate,
            "latencies": {}
        }
        
        total_start = time.perf_counter()
        
        # Step 1: Encode audio to codes using YOUR codec
        codes, encode_latency = self.codec.encode(audio)
        result["latencies"]["encode"] = encode_latency
        logger.info(f"🔢 Encode [{encode_latency:.1f}ms]: codes shape {codes.shape}")
        
        # Step 2: Transform codes using YOUR S2S (if available)
        if self.has_s2s:
            output_codes, s2s_latency = self.s2s.transform(codes)
            result["latencies"]["s2s"] = s2s_latency
            logger.info(f"🔄 S2S [{s2s_latency:.1f}ms]: output shape {output_codes.shape}")
        else:
            # Codec-only mode: just reconstruct
            output_codes = codes
            result["latencies"]["s2s"] = 0
        
        # Step 3: Decode codes to audio using YOUR codec
        output_audio, decode_latency = self.codec.decode(output_codes)
        result["latencies"]["decode"] = decode_latency
        result["audio"] = output_audio
        logger.info(f"🔊 Decode [{decode_latency:.1f}ms]: audio length {len(output_audio)}")
        
        # Total latency
        result["latencies"]["total"] = (time.perf_counter() - total_start) * 1000
        logger.info(f"⏱️ TOTAL: {result['latencies']['total']:.1f}ms")
        
        return result


# ============================================================
# Web Interface
# ============================================================

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Real-time S2S Agent (Your Models!)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #4CAF50; margin-bottom: 25px; font-size: 0.95em; }
        .highlight { color: #ff9800; }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
        }
        .mic-area { text-align: center; }
        .mic-btn {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 4px solid #4CAF50;
            background: rgba(76, 175, 80, 0.3);
            color: white;
            font-size: 50px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .mic-btn.active {
            border-color: #ff5722;
            background: rgba(255, 87, 34, 0.5);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,87,34,0.5); }
            50% { transform: scale(1.05); box-shadow: 0 0 20px 10px rgba(255,87,34,0.2); }
        }
        .status { text-align: center; margin: 20px 0; font-size: 1.2em; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .stat {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 15px;
            text-align: center;
        }
        .stat-val { font-size: 1.8em; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.8em; color: #aaa; margin-top: 5px; }
        .architecture {
            background: rgba(0,0,0,0.4);
            padding: 20px;
            border-radius: 15px;
            font-family: monospace;
            text-align: center;
            margin: 20px 0;
        }
        .arch-arrow { color: #4CAF50; font-size: 1.5em; }
        .arch-box { 
            display: inline-block;
            background: rgba(76, 175, 80, 0.3);
            padding: 8px 15px;
            border-radius: 10px;
            margin: 5px;
        }
        .info-box {
            background: rgba(33, 150, 243, 0.2);
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 0 10px 10px 0;
        }
        .success { color: #4CAF50; }
        .warning { color: #ff9800; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Real-time S2S Agent</h1>
        <p class="subtitle">Using <span class="highlight">YOUR Trained Models</span> | Target: <span class="highlight">&lt;100ms</span></p>
        
        <div class="card">
            <div class="architecture">
                <span class="arch-box">🎤 Audio In</span>
                <span class="arch-arrow">→</span>
                <span class="arch-box">📦 YOUR Codec<br>(Encode)</span>
                <span class="arch-arrow">→</span>
                <span class="arch-box">🔄 YOUR S2S<br>(Transform)</span>
                <span class="arch-arrow">→</span>
                <span class="arch-box">📦 YOUR Codec<br>(Decode)</span>
                <span class="arch-arrow">→</span>
                <span class="arch-box">🔊 Audio Out</span>
            </div>
        </div>
        
        <div class="card">
            <div class="mic-area">
                <button class="mic-btn" id="micBtn" onclick="toggleMic()">🎙️</button>
                <div class="status" id="status">Click to start streaming</div>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-val" id="encodeMs">--</div>
                    <div class="stat-label">Encode (ms)</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="s2sMs">--</div>
                    <div class="stat-label">S2S (ms)</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="decodeMs">--</div>
                    <div class="stat-label">Decode (ms)</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="totalMs">--</div>
                    <div class="stat-label">TOTAL (ms)</div>
                </div>
            </div>
        </div>
        
        <div class="card info-box">
            <h3>🎯 This is YOUR Architecture!</h3>
            <ul style="margin: 10px 0 0 20px;">
                <li><b>best_codec.pt</b> - Your trained audio codec (~9ms encode/decode)</li>
                <li><b>s2s_best.pt</b> - Your trained S2S transformer</li>
                <li class="success"><b>No ASR, No LLM, No TTS</b> - Direct audio-to-audio!</li>
                <li class="warning"><b>Target latency:</b> &lt;100ms total</li>
            </ul>
            <p style="margin-top: 15px;">🎧 <b>Use headphones!</b> Speak Telugu and hear the transformed audio.</p>
        </div>
    </div>
    
    <script>
        let ws, audioCtx, mediaStream, processor, isActive = false;
        
        function setStatus(t) { document.getElementById('status').textContent = t; }
        
        function connect() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';
            ws.onopen = () => setStatus('✅ Connected - Click to start');
            ws.onclose = () => setTimeout(connect, 1000);
            ws.onmessage = (e) => {
                if (typeof e.data === 'string') {
                    const d = JSON.parse(e.data);
                    if (d.latencies) {
                        document.getElementById('encodeMs').textContent = d.latencies.encode?.toFixed(1) || '--';
                        document.getElementById('s2sMs').textContent = d.latencies.s2s?.toFixed(1) || '--';
                        document.getElementById('decodeMs').textContent = d.latencies.decode?.toFixed(1) || '--';
                        document.getElementById('totalMs').textContent = d.latencies.total?.toFixed(1) || '--';
                        
                        // Color code total latency
                        const total = d.latencies.total || 0;
                        const el = document.getElementById('totalMs');
                        if (total < 100) el.style.color = '#4CAF50';
                        else if (total < 400) el.style.color = '#ff9800';
                        else el.style.color = '#f44336';
                    }
                } else {
                    playAudio(e.data);
                }
            };
        }
        
        async function toggleMic() {
            const btn = document.getElementById('micBtn');
            if (isActive) {
                isActive = false;
                processor?.disconnect();
                mediaStream?.getTracks().forEach(t => t.stop());
                audioCtx?.close();
                btn.classList.remove('active');
                setStatus('Click to start streaming');
                return;
            }
            
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
                });
                audioCtx = new AudioContext({ sampleRate: 16000 });
                const src = audioCtx.createMediaStreamSource(mediaStream);
                processor = audioCtx.createScriptProcessor(4096, 1, 1);
                src.connect(processor);
                processor.connect(audioCtx.destination);
                
                processor.onaudioprocess = (e) => {
                    if (!isActive) return;
                    const input = e.inputBuffer.getChannelData(0);
                    
                    // Check if there's actual audio (not just silence)
                    const rms = Math.sqrt(input.reduce((a, b) => a + b * b, 0) / input.length);
                    if (rms < 0.01) return;  // Skip silence
                    
                    // Send as float32
                    if (ws?.readyState === WebSocket.OPEN) {
                        ws.send(new Float32Array(input).buffer);
                    }
                };
                
                isActive = true;
                btn.classList.add('active');
                setStatus('🎤 Streaming... Speak now!');
            } catch (e) {
                setStatus('❌ ' + e.message);
            }
        }
        
        function playAudio(buf) {
            const pCtx = new AudioContext({ sampleRate: 16000 });
            const f32 = new Float32Array(buf);
            const ab = pCtx.createBuffer(1, f32.length, 16000);
            ab.getChannelData(0).set(f32);
            const s = pCtx.createBufferSource();
            s.buffer = ab;
            s.connect(pCtx.destination);
            s.start();
        }
        
        connect();
    </script>
</body>
</html>
'''


def create_app(agent: RealtimeS2SAgent):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    
    app = FastAPI()
    
    @app.get("/")
    async def index():
        return HTMLResponse(HTML_PAGE)
    
    @app.websocket("/ws")
    async def ws_handler(ws: WebSocket):
        await ws.accept()
        logger.info("🔗 Client connected")
        
        try:
            while True:
                data = await ws.receive_bytes()
                
                # Convert to numpy
                audio = np.frombuffer(data, dtype=np.float32)
                
                # Process through YOUR models
                result = agent.process(audio)
                
                # Send latency stats
                await ws.send_json({"latencies": result["latencies"]})
                
                # Send processed audio
                if result["audio"] is not None:
                    await ws.send_bytes(result["audio"].astype(np.float32).tobytes())
                    
        except WebSocketDisconnect:
            logger.info("🔌 Disconnected")
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Real-time S2S Agent using YOUR trained models")
    parser.add_argument("--codec", default="best_codec.pt", help="Path to your trained codec")
    parser.add_argument("--s2s", default="s2s_best.pt", help="Path to your trained S2S model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    
    config = S2SConfig(codec_path=args.codec, s2s_path=args.s2s)
    agent = RealtimeS2SAgent(config)
    app = create_app(agent)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
