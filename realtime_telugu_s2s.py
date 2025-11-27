#!/usr/bin/env python3
"""
Real-time Telugu S2S Voice Assistant
====================================

Uses YOUR trained models:
- best_codec.pt - Audio encode/decode
- best_conversation_s2s.pt - Question → Answer generation

Architecture:
Audio In → YOUR Codec → YOUR S2S → YOUR Codec → Audio Out

Target latency: <200ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TrainConfig from training script for checkpoint compatibility
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_s2s_conversation import TrainConfig
except ImportError:
    # Define it here if import fails
    @dataclass
    class TrainConfig:
        data_dir: str = "data/telugu_conversations"
        codec_path: str = "best_codec.pt"
        hidden_dim: int = 512
        num_heads: int = 8
        num_encoder_layers: int = 6
        num_decoder_layers: int = 6
        num_quantizers: int = 8
        vocab_size: int = 1024
        max_seq_len: int = 2048
        batch_size: int = 4
        learning_rate: float = 1e-4
        epochs: int = 50
        warmup_steps: int = 500
        gradient_clip: float = 1.0
        save_every: int = 5
        output_dir: str = "checkpoints/s2s_conversation"
        device: str = "cuda"


@dataclass
class Config:
    codec_path: str = "best_codec.pt"
    s2s_path: str = "checkpoints/s2s_conversation/best_conversation_s2s.pt"
    sample_rate: int = 16000
    device: str = "cuda"
    
    # S2S model config (must match training)
    hidden_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_quantizers: int = 8
    vocab_size: int = 1024
    max_seq_len: int = 2048


class TeluguCodec:
    """Your trained Telugu Codec"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        logger.info(f"📥 Loading Codec: {checkpoint_path}")
        
        from telugu_codec_fixed import TeluCodec
        
        self.codec = TeluCodec().to(self.device)
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
        start = time.perf_counter()
        audio_tensor = torch.from_numpy(audio.copy()).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        codes = self.codec.encode(audio_tensor)
        return codes, (time.perf_counter() - start) * 1000
    
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> tuple:
        start = time.perf_counter()
        audio = self.codec.decode(codes)
        return audio.squeeze().cpu().numpy(), (time.perf_counter() - start) * 1000


class ConversationS2S(nn.Module):
    """S2S Transformer for Conversation (same as training)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        embed_dim = config.hidden_dim // config.num_quantizers
        self.token_embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, embed_dim)
            for _ in range(config.num_quantizers)
        ])
        
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        
        self.output_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim // config.num_quantizers, config.vocab_size)
            for _ in range(config.num_quantizers)
        ])
    
    def embed_codes(self, codes):
        B, Q, T = codes.shape
        embeddings = []
        for q in range(Q):
            emb = self.token_embed[q](codes[:, q])
            embeddings.append(emb)
        return torch.cat(embeddings, dim=-1)
    
    @torch.no_grad()
    def generate(self, input_codes, max_len: int = None, temperature: float = 0.8):
        """Generate response codes given input codes"""
        self.eval()
        B, Q, T1 = input_codes.shape
        device = input_codes.device
        
        if max_len is None:
            max_len = min(T1 * 2, 500)  # Response roughly same length as input
        
        # Encode input
        enc_input = self.embed_codes(input_codes) + self.pos_embed[:, :T1]
        memory = self.encoder(enc_input)
        
        # Start generation
        output_codes = torch.zeros(B, Q, 1, dtype=torch.long, device=device)
        
        for t in range(max_len):
            dec_input = self.embed_codes(output_codes) + self.pos_embed[:, :t+1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=causal_mask)
            
            next_tokens = []
            chunk_size = self.config.hidden_dim // Q
            
            for q in range(Q):
                chunk = dec_output[:, -1, q*chunk_size:(q+1)*chunk_size]
                logit = self.output_heads[q](chunk)
                
                if temperature > 0:
                    probs = F.softmax(logit / temperature, dim=-1)
                    token = torch.multinomial(probs, 1)
                else:
                    token = logit.argmax(dim=-1, keepdim=True)
                
                next_tokens.append(token)
            
            next_tokens = torch.stack(next_tokens, dim=1)
            output_codes = torch.cat([output_codes, next_tokens], dim=-1)
        
        return output_codes[:, :, 1:]  # Remove initial zero


class TeluguS2SAgent:
    """Real-time Telugu S2S Voice Agent using YOUR models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        logger.info("=" * 60)
        logger.info("🎤 TELUGU S2S AGENT (Your Trained Models!)")
        logger.info("=" * 60)
        
        # Load YOUR codec
        self.codec = TeluguCodec(config.codec_path, str(self.device))
        
        # Load YOUR trained S2S
        logger.info(f"📥 Loading S2S: {config.s2s_path}")
        self.s2s = ConversationS2S(config).to(self.device)
        
        checkpoint = torch.load(config.s2s_path, map_location=self.device, weights_only=False)
        if 'model_state' in checkpoint:
            self.s2s.load_state_dict(checkpoint['model_state'])
        else:
            self.s2s.load_state_dict(checkpoint)
        self.s2s.eval()
        
        logger.info("✅ S2S ready!")
        logger.info("=" * 60)
        logger.info("✅ Agent Ready! Using YOUR trained models!")
        logger.info("=" * 60)
    
    def process(self, audio: np.ndarray) -> dict:
        """
        Full S2S pipeline using YOUR models:
        Audio → Codec Encode → S2S Generate → Codec Decode → Audio
        """
        result = {
            "audio": None,
            "latencies": {}
        }
        
        total_start = time.perf_counter()
        
        # Step 1: Encode with YOUR codec
        input_codes, enc_lat = self.codec.encode(audio)
        result["latencies"]["encode"] = enc_lat
        logger.info(f"📦 Encode [{enc_lat:.1f}ms]: shape {input_codes.shape}")
        
        # Step 2: Generate response with YOUR S2S
        s2s_start = time.perf_counter()
        output_codes = self.s2s.generate(input_codes, temperature=0.8)
        s2s_lat = (time.perf_counter() - s2s_start) * 1000
        result["latencies"]["s2s"] = s2s_lat
        logger.info(f"🔄 S2S [{s2s_lat:.1f}ms]: shape {output_codes.shape}")
        
        # Step 3: Decode with YOUR codec
        output_audio, dec_lat = self.codec.decode(output_codes)
        result["latencies"]["decode"] = dec_lat
        result["audio"] = output_audio
        logger.info(f"🔊 Decode [{dec_lat:.1f}ms]: length {len(output_audio)}")
        
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
    <title>🎤 Telugu S2S - Your Trained Models!</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        .container { max-width: 750px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 8px; font-size: 2em; }
        .subtitle { text-align: center; margin-bottom: 25px; }
        .highlight { color: #00ff88; font-weight: bold; }
        .card {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .architecture {
            background: rgba(0,255,136,0.1);
            border: 2px solid #00ff88;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
        }
        .arch-flow {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .arch-box {
            background: rgba(0,255,136,0.2);
            padding: 12px 18px;
            border-radius: 12px;
            font-weight: bold;
        }
        .arch-arrow { font-size: 1.5em; color: #00ff88; }
        .your-model { background: rgba(255,152,0,0.3); border: 2px solid #ff9800; }
        .mic-area { text-align: center; padding: 20px 0; }
        .mic-btn {
            width: 130px;
            height: 130px;
            border-radius: 50%;
            border: 4px solid #00ff88;
            background: rgba(0, 255, 136, 0.2);
            color: white;
            font-size: 55px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 0 30px rgba(0,255,136,0.3);
        }
        .mic-btn:hover { transform: scale(1.05); }
        .mic-btn.active {
            border-color: #ff5722;
            background: rgba(255, 87, 34, 0.4);
            animation: pulse 1s infinite;
            box-shadow: 0 0 40px rgba(255,87,34,0.5);
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.08); }
        }
        .status { text-align: center; margin: 20px 0; font-size: 1.3em; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 25px;
        }
        .stat {
            background: rgba(0,0,0,0.4);
            padding: 18px;
            border-radius: 15px;
            text-align: center;
        }
        .stat-val { font-size: 2em; font-weight: bold; color: #00ff88; }
        .stat-label { font-size: 0.8em; color: #aaa; margin-top: 8px; }
        .info-box {
            background: rgba(0,200,255,0.15);
            border-left: 4px solid #00c8ff;
            padding: 18px;
            border-radius: 0 15px 15px 0;
            margin-top: 20px;
        }
        .success { color: #00ff88; }
        .latency-good { color: #00ff88; }
        .latency-ok { color: #ff9800; }
        .latency-bad { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu S2S Voice Agent</h1>
        <p class="subtitle">Using <span class="highlight">YOUR Trained Models</span> | Direct Audio-to-Audio</p>
        
        <div class="card">
            <div class="architecture">
                <div style="margin-bottom: 10px; font-size: 0.9em; color: #aaa;">YOUR ARCHITECTURE:</div>
                <div class="arch-flow">
                    <span class="arch-box">🎤 Audio In</span>
                    <span class="arch-arrow">→</span>
                    <span class="arch-box your-model">📦 YOUR Codec<br><small>Encode</small></span>
                    <span class="arch-arrow">→</span>
                    <span class="arch-box your-model">🧠 YOUR S2S<br><small>Generate</small></span>
                    <span class="arch-arrow">→</span>
                    <span class="arch-box your-model">📦 YOUR Codec<br><small>Decode</small></span>
                    <span class="arch-arrow">→</span>
                    <span class="arch-box">🔊 Audio Out</span>
                </div>
                <div style="margin-top: 15px; font-size: 0.85em; color: #00ff88;">
                    ✅ No ASR | ✅ No LLM | ✅ No TTS | ✅ Pure Audio-to-Audio
                </div>
            </div>
            
            <div class="mic-area">
                <button class="mic-btn" id="micBtn" onclick="toggleMic()">🎙️</button>
                <div class="status" id="status">Click to start listening</div>
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
            <h3>🎯 What This Does</h3>
            <ul style="margin: 12px 0 0 20px; line-height: 1.8;">
                <li>Speak Telugu into the microphone</li>
                <li>Your <b>trained codec</b> encodes the audio to codes</li>
                <li>Your <b>trained S2S</b> generates response codes</li>
                <li>Your <b>trained codec</b> decodes to audio response</li>
                <li class="success"><b>Target:</b> &lt;200ms end-to-end latency!</li>
            </ul>
            <p style="margin-top: 15px; color: #ff9800;">
                ⚠️ <b>Note:</b> This is trained on 100 pairs. More data = better responses!
            </p>
        </div>
    </div>
    
    <script>
        let ws, audioCtx, mediaStream, processor, isActive = false, buffer = [];
        
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
                        document.getElementById('encodeMs').textContent = d.latencies.encode?.toFixed(0) || '--';
                        document.getElementById('s2sMs').textContent = d.latencies.s2s?.toFixed(0) || '--';
                        document.getElementById('decodeMs').textContent = d.latencies.decode?.toFixed(0) || '--';
                        
                        const total = d.latencies.total || 0;
                        const totalEl = document.getElementById('totalMs');
                        totalEl.textContent = total.toFixed(0);
                        totalEl.className = 'stat-val ' + (total < 200 ? 'latency-good' : total < 500 ? 'latency-ok' : 'latency-bad');
                    }
                    if (d.status === 'done') setStatus('🎤 Listening...');
                } else {
                    playAudio(e.data);
                }
            };
        }
        
        async function toggleMic() {
            const btn = document.getElementById('micBtn');
            if (isActive) {
                isActive = false;
                if (buffer.length > 0) sendBuffer();
                processor?.disconnect();
                mediaStream?.getTracks().forEach(t => t.stop());
                btn.classList.remove('active');
                setStatus('Click to start listening');
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
                
                let silenceCount = 0;
                processor.onaudioprocess = (e) => {
                    if (!isActive) return;
                    const input = e.inputBuffer.getChannelData(0);
                    const rms = Math.sqrt(input.reduce((a,b) => a+b*b, 0) / input.length);
                    
                    if (rms > 0.015) {
                        buffer.push(...input);
                        silenceCount = 0;
                        setStatus('🗣️ Speaking...');
                    } else if (buffer.length > 0) {
                        silenceCount++;
                        if (silenceCount > 3) {
                            sendBuffer();
                            silenceCount = 0;
                        }
                    }
                };
                
                isActive = true;
                buffer = [];
                btn.classList.add('active');
                setStatus('🎤 Listening...');
            } catch (e) {
                setStatus('❌ ' + e.message);
            }
        }
        
        function sendBuffer() {
            if (buffer.length < 8000) { buffer = []; return; }
            setStatus('⏳ Processing with YOUR models...');
            const f32 = new Float32Array(buffer);
            if (ws?.readyState === WebSocket.OPEN) ws.send(f32.buffer);
            buffer = [];
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
            setStatus('🔊 Playing response...');
        }
        
        connect();
    </script>
</body>
</html>
'''


def create_app(agent: TeluguS2SAgent):
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
                audio = np.frombuffer(data, dtype=np.float32)
                
                if len(audio) < 8000:
                    continue
                
                # Process with YOUR trained models!
                result = agent.process(audio)
                
                # Send latencies
                await ws.send_json({
                    "latencies": result["latencies"],
                    "status": "done"
                })
                
                # Send audio response
                if result["audio"] is not None:
                    await ws.send_bytes(result["audio"].astype(np.float32).tobytes())
                    
        except WebSocketDisconnect:
            logger.info("🔌 Disconnected")
    
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Telugu S2S using YOUR trained models")
    parser.add_argument("--codec", default="best_codec.pt")
    parser.add_argument("--s2s", default="checkpoints/s2s_conversation/best_conversation_s2s.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    
    config = Config(codec_path=args.codec, s2s_path=args.s2s)
    agent = TeluguS2SAgent(config)
    app = create_app(agent)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
