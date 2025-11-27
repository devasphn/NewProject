#!/usr/bin/env python3
"""
Telugu Voice Agent - FAST Version
Using faster-whisper + Qwen2.5-3B for low latency
"""

import torch
import numpy as np
import time
import logging
import asyncio
import io
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    whisper_model: str = "medium"  # medium is good balance of speed/quality
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Smaller = faster
    tts_voice: str = "te-IN-ShrutiNeural"
    sample_rate: int = 16000
    silence_threshold: float = 0.015
    min_speech_duration: float = 0.8
    silence_duration: float = 0.7


class FasterWhisperASR:
    """Faster-Whisper ASR - Using CPU to avoid cuDNN issues"""
    
    def __init__(self, model_size: str = "large-v3", device: str = "cpu"):
        logger.info(f"📥 Loading faster-whisper: {model_size} (CPU mode)")
        
        from faster_whisper import WhisperModel
        
        # Use CPU with int8 for speed (avoids cuDNN issues)
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"  # Fast on CPU
        )
        logger.info("✅ Faster-Whisper ready (CPU)")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe with Telugu language forced"""
        start = time.perf_counter()
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Transcribe with Telugu language
        segments, info = self.model.transcribe(
            audio,
            language="te",  # Force Telugu
            task="transcribe",
            beam_size=3,  # Faster with smaller beam
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            )
        )
        
        # Collect text
        text = " ".join([seg.text for seg in segments]).strip()
        latency = (time.perf_counter() - start) * 1000
        
        logger.info(f"🎤 Detected language: {info.language} ({info.language_probability:.1%})")
        
        return {"text": text, "latency_ms": latency, "language": info.language}


class QwenLLM:
    """Qwen2.5 LLM - Using 3B for speed"""
    
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
        self.device = device
        
        self.system_prompt = """నీవు తెలుగు అసిస్టెంట్. You are a Telugu assistant.

Rules:
1. ALWAYS reply in Telugu script (తెలుగు లిపి) only
2. Keep responses SHORT - 1 sentence only
3. Be helpful and friendly
4. If you don't understand, ask in Telugu

Example:
User: నమస్కారం
Assistant: నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?"""
        
        self.history = []
        logger.info("✅ LLM ready")
    
    @torch.no_grad()
    def generate(self, user_input: str) -> dict:
        start = time.perf_counter()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-2:])  # Keep minimal history
        messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,  # Short responses
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Clean response
        response = response.split('\n')[0].strip()  # Take first line only
        
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return {"text": response, "latency_ms": (time.perf_counter() - start) * 1000}


class EdgeTTS:
    """Edge TTS for Telugu"""
    
    def __init__(self, voice: str = "te-IN-ShrutiNeural"):
        self.voice = voice
        logger.info(f"📥 Edge TTS: {voice}")
    
    async def synthesize_async(self, text: str) -> dict:
        import edge_tts
        from pydub import AudioSegment
        
        start = time.perf_counter()
        
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
        
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        
        return {
            "audio": samples,
            "sample_rate": 24000,
            "latency_ms": (time.perf_counter() - start) * 1000
        }


class AudioBuffer:
    """Audio buffer with VAD"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        self.chunks_per_second = config.sample_rate / 4096
        self.silence_chunks_needed = int(config.silence_duration * self.chunks_per_second)
        
    def add_chunk(self, audio: np.ndarray):
        """Add chunk, return complete audio if speech ended"""
        rms = np.sqrt(np.mean(audio ** 2))
        is_speech = rms > self.config.silence_threshold
        
        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.silence_chunks = 0
                logger.info("🎤 Speech started...")
            self.buffer.append(audio)
            self.silence_chunks = 0
        else:
            if self.is_speaking:
                self.buffer.append(audio)
                self.silence_chunks += 1
                
                if self.silence_chunks >= self.silence_chunks_needed:
                    return self._flush()
        
        return None
    
    def _flush(self):
        if not self.buffer:
            return None
        
        audio = np.concatenate(self.buffer)
        duration = len(audio) / self.config.sample_rate
        
        self.buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        
        if duration >= self.config.min_speech_duration:
            logger.info(f"📝 Got {duration:.1f}s audio")
            return audio
        return None


class TeluguVoiceAgent:
    """Fast Telugu Voice Agent"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        logger.info("=" * 60)
        logger.info("🎤 TELUGU VOICE AGENT - FAST")
        logger.info("=" * 60)
        
        self.asr = FasterWhisperASR(self.config.whisper_model)
        self.llm = QwenLLM(self.config.llm_model)
        self.tts = EdgeTTS(self.config.tts_voice)
        
        logger.info("=" * 60)
        logger.info("✅ Agent Ready!")
        logger.info("=" * 60)
    
    async def process(self, audio: np.ndarray) -> dict:
        result = {
            "transcription": None,
            "response_text": None,
            "audio": None,
            "latencies": {}
        }
        
        # ASR
        asr_result = self.asr.transcribe(audio, self.config.sample_rate)
        result["transcription"] = asr_result["text"]
        result["latencies"]["asr"] = asr_result["latency_ms"]
        logger.info(f"🎤 ASR [{asr_result['latency_ms']:.0f}ms]: {asr_result['text']}")
        
        # Skip empty
        if not result["transcription"] or len(result["transcription"]) < 2:
            return result
        
        # LLM
        llm_result = self.llm.generate(result["transcription"])
        result["response_text"] = llm_result["text"]
        result["latencies"]["llm"] = llm_result["latency_ms"]
        logger.info(f"🤖 LLM [{llm_result['latency_ms']:.0f}ms]: {llm_result['text']}")
        
        # TTS
        tts_result = await self.tts.synthesize_async(result["response_text"])
        result["audio"] = tts_result["audio"]
        result["latencies"]["tts"] = tts_result["latency_ms"]
        result["latencies"]["total"] = sum(result["latencies"].values())
        logger.info(f"🔊 TTS [{tts_result['latency_ms']:.0f}ms]")
        logger.info(f"⏱️ Total: {result['latencies']['total']:.0f}ms")
        
        return result


# ============================================================
# Web Interface
# ============================================================

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Telugu Voice Agent - Fast</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #aaa; margin-bottom: 25px; font-size: 0.9em; }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
        }
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
            transition: all 0.3s;
        }
        .mic-btn.listening {
            border-color: #ff5722;
            background: rgba(255, 87, 34, 0.4);
            animation: pulse 1s infinite;
        }
        .mic-btn.speaking { border-color: #ff9800; background: rgba(255, 152, 0, 0.5); }
        .mic-btn.processing { border-color: #2196f3; background: rgba(33, 150, 243, 0.4); }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .status { text-align: center; margin: 15px 0; font-size: 1.1em; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin-top: 15px;
        }
        .stat {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-val { font-size: 1.4em; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.7em; color: #aaa; }
        .chat {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
        }
        .msg {
            margin: 8px 0;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 85%;
        }
        .user { background: rgba(33, 150, 243, 0.4); margin-left: auto; text-align: right; }
        .bot { background: rgba(76, 175, 80, 0.4); }
        .telugu { font-size: 1.1em; line-height: 1.4; }
        .visualizer {
            height: 50px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            margin: 15px 0;
        }
        .visualizer canvas { width: 100%; height: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu Voice Agent</h1>
        <p class="subtitle">faster-whisper (medium) + Qwen2.5-3B + Edge TTS | Target: &lt;2000ms</p>
        
        <div class="card">
            <div class="mic-area">
                <button class="mic-btn" id="micBtn" onclick="toggleMic()">🎙️</button>
                <div class="status" id="status">Click to start</div>
            </div>
            
            <div class="visualizer"><canvas id="viz"></canvas></div>
            
            <div class="stats">
                <div class="stat"><div class="stat-val" id="asrMs">--</div><div class="stat-label">ASR</div></div>
                <div class="stat"><div class="stat-val" id="llmMs">--</div><div class="stat-label">LLM</div></div>
                <div class="stat"><div class="stat-val" id="ttsMs">--</div><div class="stat-label">TTS</div></div>
                <div class="stat"><div class="stat-val" id="totalMs">--</div><div class="stat-label">TOTAL</div></div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 10px;">💬 Chat</h3>
            <div class="chat" id="chat">
                <div class="msg bot"><span class="telugu">నమస్కారం! నేను మీ తెలుగు అసిస్టెంట్.</span></div>
            </div>
        </div>
        
        <div class="card" style="font-size: 0.9em;">
            <b>📋 Tips:</b> Click mic → Speak Telugu → Pause 0.7s → Get response<br>
            <b>🎧 Use headphones!</b> Say: "నమస్కారం, నీ పేరు ఏమిటి?"
        </div>
    </div>
    
    <script>
        let ws, audioCtx, mediaStream, processor, analyser, isListening = false;
        const canvas = document.getElementById('viz');
        const ctx = canvas.getContext('2d');
        
        function setStatus(t, c = '') {
            document.getElementById('status').textContent = t;
            document.getElementById('micBtn').className = 'mic-btn ' + c;
        }
        
        function addMsg(role, text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'msg ' + role;
            div.innerHTML = '<span class="telugu">' + text + '</span>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function connect() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';
            ws.onopen = () => setStatus('Ready - Click to start');
            ws.onclose = () => setTimeout(connect, 1000);
            ws.onmessage = (e) => {
                if (typeof e.data === 'string') {
                    const d = JSON.parse(e.data);
                    if (d.status === 'listening') setStatus('🎤 Listening...', 'listening');
                    else if (d.status === 'speaking') setStatus('🗣️ Speaking...', 'speaking');
                    else if (d.status === 'processing') setStatus('⏳ Processing...', 'processing');
                    else if (d.status === 'done') setStatus('🎤 Listening...', 'listening');
                    
                    if (d.latencies) {
                        document.getElementById('asrMs').textContent = d.latencies.asr?.toFixed(0) || '--';
                        document.getElementById('llmMs').textContent = d.latencies.llm?.toFixed(0) || '--';
                        document.getElementById('ttsMs').textContent = d.latencies.tts?.toFixed(0) || '--';
                        document.getElementById('totalMs').textContent = d.latencies.total?.toFixed(0) || '--';
                    }
                    if (d.transcription) addMsg('user', d.transcription);
                    if (d.response_text) addMsg('bot', d.response_text);
                } else {
                    playAudio(e.data);
                }
            };
        }
        
        async function toggleMic() {
            if (isListening) { stopMic(); return; }
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
                audioCtx = new AudioContext({ sampleRate: 16000 });
                const src = audioCtx.createMediaStreamSource(mediaStream);
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                src.connect(analyser);
                processor = audioCtx.createScriptProcessor(4096, 1, 1);
                src.connect(processor);
                processor.connect(audioCtx.destination);
                processor.onaudioprocess = (e) => {
                    if (!isListening) return;
                    const input = e.inputBuffer.getChannelData(0);
                    const int16 = new Int16Array(input.length);
                    for (let i = 0; i < input.length; i++) int16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
                    if (ws?.readyState === WebSocket.OPEN) ws.send(int16.buffer);
                };
                isListening = true;
                setStatus('🎤 Listening...', 'listening');
                drawViz();
            } catch (e) { setStatus('❌ ' + e.message); }
        }
        
        function stopMic() {
            isListening = false;
            processor?.disconnect();
            mediaStream?.getTracks().forEach(t => t.stop());
            audioCtx?.close();
            setStatus('Click to start');
        }
        
        function playAudio(buf) {
            const pCtx = new AudioContext({ sampleRate: 24000 });
            const f32 = new Float32Array(buf);
            const ab = pCtx.createBuffer(1, f32.length, 24000);
            ab.getChannelData(0).set(f32);
            const s = pCtx.createBufferSource();
            s.buffer = ab;
            s.connect(pCtx.destination);
            s.start();
        }
        
        function drawViz() {
            if (!analyser || !isListening) return;
            requestAnimationFrame(drawViz);
            const data = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(data);
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width / data.length * 2;
            let x = 0;
            for (let i = 0; i < data.length; i++) {
                const h = (data[i] / 255) * canvas.height;
                ctx.fillStyle = `hsl(${120 + i}, 70%, 50%)`;
                ctx.fillRect(x, canvas.height - h, w, h);
                x += w + 1;
            }
        }
        
        connect();
    </script>
</body>
</html>
'''


def create_app(agent, config):
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
        buffer = AudioBuffer(config)
        
        try:
            while True:
                data = await ws.receive_bytes()
                int16 = np.frombuffer(data, dtype=np.int16)
                float32 = int16.astype(np.float32) / 32768.0
                
                complete = buffer.add_chunk(float32)
                
                if buffer.is_speaking:
                    await ws.send_json({"status": "speaking"})
                else:
                    await ws.send_json({"status": "listening"})
                
                if complete is not None:
                    await ws.send_json({"status": "processing"})
                    
                    try:
                        result = await agent.process(complete)
                        await ws.send_json({
                            "status": "done",
                            "transcription": result["transcription"],
                            "response_text": result["response_text"],
                            "latencies": result["latencies"]
                        })
                        if result["audio"] is not None:
                            await ws.send_bytes(result["audio"].astype(np.float32).tobytes())
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        await ws.send_json({"status": "done", "error": str(e)})
                        
        except WebSocketDisconnect:
            logger.info("🔌 Disconnected")
    
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--whisper", default="medium")  # medium for CPU speed
    parser.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()
    
    config = AgentConfig(whisper_model=args.whisper, llm_model=args.llm)
    agent = TeluguVoiceAgent(config)
    app = create_app(agent, config)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
