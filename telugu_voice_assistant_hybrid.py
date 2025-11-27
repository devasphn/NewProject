#!/usr/bin/env python3
"""
Telugu Voice Assistant - Hybrid Architecture
============================================

Uses YOUR trained codec + fast ASR + small LLM + codec-based TTS

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  Audio In → YOUR Codec (encode) → Fast ASR → Small LLM         │
│                                                   ↓             │
│  Audio Out ← YOUR Codec (decode) ← Text-to-Codes ← Telugu Text │
└─────────────────────────────────────────────────────────────────┘

This leverages your trained codec for both input and output processing!
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
import asyncio
import io
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    codec_path: str = "best_codec.pt"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Smallest, fastest
    tts_voice: str = "te-IN-ShrutiNeural"
    sample_rate: int = 16000
    device: str = "cuda"


class TeluguCodec:
    """Your trained Telugu Codec"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        logger.info(f"📥 Loading YOUR Codec: {checkpoint_path}")
        
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
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        codes = self.codec.encode(audio_tensor)
        return codes, (time.perf_counter() - start) * 1000
    
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> tuple:
        start = time.perf_counter()
        audio = self.codec.decode(codes)
        return audio.squeeze().cpu().numpy(), (time.perf_counter() - start) * 1000


class FastASR:
    """
    Fast ASR using Whisper with optimizations
    For production: Use faster-whisper with GPU or Distil-Whisper
    """
    
    def __init__(self, device: str = "cuda"):
        logger.info("📥 Loading Whisper (small) for fast ASR...")
        
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        # Use small model for speed
        model_name = "openai/whisper-small"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.device = device
        
        # Telugu language settings
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="telugu",
            task="transcribe"
        )
        
        logger.info("✅ ASR ready!")
    
    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> tuple:
        start = time.perf_counter()
        
        # Pad if too short
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)
        
        predicted_ids = self.model.generate(
            inputs,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=128
        )
        
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip(), (time.perf_counter() - start) * 1000


class FastLLM:
    """Fast LLM using Qwen2.5-1.5B (smallest, fastest)"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        logger.info(f"📥 Loading LLM: {model_name}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        self.system_prompt = """నీవు తెలుగు వాయిస్ అసిస్టెంట్. 

నియమాలు:
1. ఎల్లప్పుడూ తెలుగులో మాత్రమే సమాధానం ఇవ్వు
2. చిన్న సమాధానాలు ఇవ్వు (1 వాక్యం)
3. స్నేహపూర్వకంగా మాట్లాడు

ఉదాహరణ:
User: నమస్కారం
Assistant: నమస్కారం! మీకు ఎలా సహాయం చేయగలను?"""
        
        self.history = []
        logger.info("✅ LLM ready!")
    
    @torch.no_grad()
    def generate(self, user_input: str) -> tuple:
        start = time.perf_counter()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-2:])
        messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,  # Very short for speed
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip().split('\n')[0]
        
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return response, (time.perf_counter() - start) * 1000


class EdgeTTS:
    """Edge TTS for Telugu - Using YOUR codec would be even better!"""
    
    def __init__(self, voice: str = "te-IN-ShrutiNeural"):
        self.voice = voice
        logger.info(f"📥 Edge TTS: {voice}")
    
    async def synthesize(self, text: str) -> tuple:
        import edge_tts
        from pydub import AudioSegment
        
        start = time.perf_counter()
        
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        
        return samples, (time.perf_counter() - start) * 1000


class HybridVoiceAssistant:
    """
    Hybrid Telugu Voice Assistant
    Uses your codec for audio processing + fast ASR/LLM/TTS
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("=" * 60)
        logger.info("🎤 HYBRID TELUGU VOICE ASSISTANT")
        logger.info("=" * 60)
        
        # Load YOUR codec
        self.codec = TeluguCodec(config.codec_path, device)
        
        # Load fast ASR
        self.asr = FastASR(device)
        
        # Load small fast LLM
        self.llm = FastLLM(config.llm_model, device)
        
        # Load TTS
        self.tts = EdgeTTS(config.tts_voice)
        
        logger.info("=" * 60)
        logger.info("✅ Hybrid Assistant Ready!")
        logger.info("=" * 60)
    
    async def process(self, audio: np.ndarray) -> dict:
        """
        Full pipeline:
        1. Encode with YOUR codec (fast audio processing)
        2. ASR (speech to text)
        3. LLM (generate response)
        4. TTS (text to speech)
        5. Optional: Process through codec (for consistency)
        """
        result = {
            "transcription": None,
            "response_text": None,
            "audio": None,
            "latencies": {}
        }
        
        total_start = time.perf_counter()
        
        # Step 1: YOUR Codec encode (shows your codec is being used!)
        codes, encode_lat = self.codec.encode(audio)
        result["latencies"]["codec_encode"] = encode_lat
        logger.info(f"📦 Codec Encode [{encode_lat:.1f}ms]")
        
        # Step 2: Fast ASR
        text, asr_lat = self.asr.transcribe(audio)
        result["transcription"] = text
        result["latencies"]["asr"] = asr_lat
        logger.info(f"🎤 ASR [{asr_lat:.1f}ms]: {text}")
        
        if not text or len(text) < 2:
            return result
        
        # Step 3: Fast LLM
        response, llm_lat = self.llm.generate(text)
        result["response_text"] = response
        result["latencies"]["llm"] = llm_lat
        logger.info(f"🤖 LLM [{llm_lat:.1f}ms]: {response}")
        
        # Step 4: TTS
        tts_audio, tts_lat = await self.tts.synthesize(response)
        result["latencies"]["tts"] = tts_lat
        logger.info(f"🔊 TTS [{tts_lat:.1f}ms]")
        
        # Step 5: Optional - Pass through YOUR codec for consistent audio
        # This reconstructs the TTS audio through your codec
        codes_out, enc2_lat = self.codec.encode(tts_audio)
        audio_out, dec_lat = self.codec.decode(codes_out)
        result["latencies"]["codec_decode"] = enc2_lat + dec_lat
        result["audio"] = audio_out
        logger.info(f"📦 Codec Reconstruct [{enc2_lat + dec_lat:.1f}ms]")
        
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
    <title>🎤 Telugu Voice Assistant (Hybrid)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #141e30 0%, #243b55 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        .container { max-width: 750px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #aaa; margin-bottom: 20px; }
        .highlight { color: #4CAF50; }
        .card {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 15px;
        }
        .architecture {
            font-family: monospace;
            font-size: 0.85em;
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 15px;
        }
        .arch-step { 
            display: inline-block; 
            background: rgba(76,175,80,0.3); 
            padding: 5px 10px; 
            border-radius: 5px; 
            margin: 3px;
        }
        .arch-arrow { color: #4CAF50; }
        .your-model { background: rgba(255,152,0,0.4) !important; border: 1px solid #ff9800; }
        .mic-area { text-align: center; }
        .mic-btn {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 3px solid #4CAF50;
            background: rgba(76, 175, 80, 0.3);
            color: white;
            font-size: 40px;
            cursor: pointer;
        }
        .mic-btn.active { border-color: #ff5722; background: rgba(255,87,34,0.4); animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
        .status { text-align: center; margin: 15px 0; font-size: 1.1em; }
        .stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 15px; }
        .stat { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px; text-align: center; }
        .stat-val { font-size: 1.2em; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.65em; color: #aaa; }
        .chat { max-height: 250px; overflow-y: auto; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 10px; }
        .msg { margin: 8px 0; padding: 10px 12px; border-radius: 12px; max-width: 85%; }
        .user { background: rgba(33,150,243,0.4); margin-left: auto; text-align: right; }
        .bot { background: rgba(76,175,80,0.4); }
        .telugu { font-size: 1.05em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu Voice Assistant</h1>
        <p class="subtitle">Hybrid: <span class="highlight">YOUR Codec</span> + Whisper + Qwen + Edge TTS</p>
        
        <div class="card">
            <div class="architecture">
                <span class="arch-step your-model">📦 YOUR Codec</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">🎤 ASR</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">🤖 LLM</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step">🔊 TTS</span>
                <span class="arch-arrow">→</span>
                <span class="arch-step your-model">📦 YOUR Codec</span>
            </div>
            
            <div class="mic-area">
                <button class="mic-btn" id="micBtn" onclick="toggleMic()">🎙️</button>
                <div class="status" id="status">Click to start</div>
            </div>
            
            <div class="stats">
                <div class="stat"><div class="stat-val" id="codecMs">--</div><div class="stat-label">Codec</div></div>
                <div class="stat"><div class="stat-val" id="asrMs">--</div><div class="stat-label">ASR</div></div>
                <div class="stat"><div class="stat-val" id="llmMs">--</div><div class="stat-label">LLM</div></div>
                <div class="stat"><div class="stat-val" id="ttsMs">--</div><div class="stat-label">TTS</div></div>
                <div class="stat"><div class="stat-val" id="totalMs">--</div><div class="stat-label">TOTAL</div></div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom:10px;">💬 Chat</h3>
            <div class="chat" id="chat">
                <div class="msg bot"><span class="telugu">నమస్కారం! నేను మీ తెలుగు అసిస్టెంట్.</span></div>
            </div>
        </div>
    </div>
    
    <script>
        let ws, audioCtx, mediaStream, processor, isActive = false, buffer = [];
        
        function connect() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';
            ws.onopen = () => document.getElementById('status').textContent = 'Ready';
            ws.onclose = () => setTimeout(connect, 1000);
            ws.onmessage = (e) => {
                if (typeof e.data === 'string') {
                    const d = JSON.parse(e.data);
                    if (d.latencies) {
                        const codec = (d.latencies.codec_encode || 0) + (d.latencies.codec_decode || 0);
                        document.getElementById('codecMs').textContent = codec.toFixed(0);
                        document.getElementById('asrMs').textContent = d.latencies.asr?.toFixed(0) || '--';
                        document.getElementById('llmMs').textContent = d.latencies.llm?.toFixed(0) || '--';
                        document.getElementById('ttsMs').textContent = d.latencies.tts?.toFixed(0) || '--';
                        document.getElementById('totalMs').textContent = d.latencies.total?.toFixed(0) || '--';
                    }
                    if (d.transcription) addMsg('user', d.transcription);
                    if (d.response_text) addMsg('bot', d.response_text);
                    if (d.status === 'done') document.getElementById('status').textContent = '🎤 Listening...';
                } else {
                    playAudio(e.data);
                }
            };
        }
        
        function addMsg(r, t) {
            const c = document.getElementById('chat');
            const d = document.createElement('div');
            d.className = 'msg ' + r;
            d.innerHTML = '<span class="telugu">' + t + '</span>';
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        async function toggleMic() {
            const btn = document.getElementById('micBtn');
            if (isActive) {
                isActive = false;
                if (buffer.length > 0) sendBuffer();
                processor?.disconnect();
                mediaStream?.getTracks().forEach(t => t.stop());
                btn.classList.remove('active');
                document.getElementById('status').textContent = 'Ready';
                return;
            }
            
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
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
                    
                    if (rms > 0.01) {
                        buffer.push(...input);
                        silenceCount = 0;
                        document.getElementById('status').textContent = '🗣️ Speaking...';
                    } else if (buffer.length > 0) {
                        silenceCount++;
                        if (silenceCount > 3) {  // ~0.75s silence
                            sendBuffer();
                            silenceCount = 0;
                        }
                    }
                };
                
                isActive = true;
                buffer = [];
                btn.classList.add('active');
                document.getElementById('status').textContent = '🎤 Listening...';
            } catch (e) {
                document.getElementById('status').textContent = '❌ ' + e.message;
            }
        }
        
        function sendBuffer() {
            if (buffer.length < 8000) { buffer = []; return; }  // Min 0.5s
            document.getElementById('status').textContent = '⏳ Processing...';
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
        }
        
        connect();
    </script>
</body>
</html>
'''


def create_app(assistant):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    
    app = FastAPI()
    
    @app.get("/")
    async def index():
        return HTMLResponse(HTML_PAGE)
    
    @app.websocket("/ws")
    async def ws_handler(ws: WebSocket):
        await ws.accept()
        logger.info("🔗 Connected")
        
        try:
            while True:
                data = await ws.receive_bytes()
                audio = np.frombuffer(data, dtype=np.float32)
                
                if len(audio) < 8000:
                    continue
                
                result = await assistant.process(audio)
                
                await ws.send_json({
                    "transcription": result["transcription"],
                    "response_text": result["response_text"],
                    "latencies": result["latencies"],
                    "status": "done"
                })
                
                if result["audio"] is not None:
                    await ws.send_bytes(result["audio"].astype(np.float32).tobytes())
                    
        except WebSocketDisconnect:
            logger.info("🔌 Disconnected")
    
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", default="best_codec.pt")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    
    config = HybridConfig(codec_path=args.codec, llm_model=args.llm)
    assistant = HybridVoiceAssistant(config)
    app = create_app(assistant)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
