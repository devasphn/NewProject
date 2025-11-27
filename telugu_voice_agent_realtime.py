#!/usr/bin/env python3
"""
Telugu Voice Agent - Real-time Streaming
ASR (Whisper) → LLM (Qwen2.5) → TTS (Edge TTS)

Low latency real-time Telugu conversation
"""

import torch
import numpy as np
import time
import logging
import asyncio
import io
import tempfile
import os
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    whisper_model: str = "openai/whisper-large-v3"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    tts_voice: str = "te-IN-ShrutiNeural"  # Telugu female
    max_new_tokens: int = 150
    temperature: float = 0.7


class WhisperASR:
    """Whisper ASR for Telugu"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        logger.info(f"📥 Loading Whisper: {model_name}")
        
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="telugu", 
            task="transcribe"
        )
        logger.info("✅ Whisper ready")
    
    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe audio to Telugu text"""
        import torchaudio
        
        start = time.perf_counter()
        
        # Resample if needed
        if sample_rate != 16000:
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
            audio = audio_tensor.squeeze().numpy()
        
        # Process
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)
        
        predicted_ids = self.model.generate(
            inputs,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=256
        )
        
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        latency = (time.perf_counter() - start) * 1000
        
        return {"text": text.strip(), "latency_ms": latency}


class QwenLLM:
    """Qwen2.5 LLM for Telugu responses"""
    
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
        
        self.system_prompt = """You are a helpful Telugu voice assistant.
Always respond in Telugu (తెలుగు) language only.
Keep responses short and conversational (1-2 sentences).
Be friendly and natural."""
        
        self.history = []
        logger.info("✅ LLM ready")
    
    @torch.no_grad()
    def generate(self, user_input: str) -> dict:
        start = time.perf_counter()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-4:])  # Keep last 2 exchanges
        messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return {"text": response, "latency_ms": (time.perf_counter() - start) * 1000}


class EdgeTTS:
    """Edge TTS for Telugu"""
    
    def __init__(self, voice: str = "te-IN-ShrutiNeural"):
        self.voice = voice
        logger.info(f"📥 Edge TTS voice: {voice}")
        logger.info("✅ TTS ready")
    
    async def synthesize_async(self, text: str) -> dict:
        import edge_tts
        
        start = time.perf_counter()
        
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        # Convert MP3 to numpy
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
        
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        
        return {
            "audio": samples,
            "sample_rate": 24000,
            "latency_ms": (time.perf_counter() - start) * 1000
        }
    
    def synthesize(self, text: str) -> dict:
        return asyncio.get_event_loop().run_until_complete(self.synthesize_async(text))


class TeluguVoiceAgent:
    """Complete Telugu Voice Agent"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        logger.info("=" * 60)
        logger.info("🎤 TELUGU VOICE AGENT - REAL-TIME")
        logger.info("=" * 60)
        
        self.asr = WhisperASR(self.config.whisper_model)
        self.llm = QwenLLM(self.config.llm_model)
        self.tts = EdgeTTS(self.config.tts_voice)
        
        logger.info("=" * 60)
        logger.info("✅ Agent Ready!")
        logger.info("=" * 60)
    
    async def process_async(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Process audio and return response"""
        result = {
            "transcription": None,
            "response_text": None,
            "audio": None,
            "sample_rate": 24000,
            "latencies": {}
        }
        
        # ASR
        asr_result = self.asr.transcribe(audio, sample_rate)
        result["transcription"] = asr_result["text"]
        result["latencies"]["asr"] = asr_result["latency_ms"]
        logger.info(f"🎤 ASR [{asr_result['latency_ms']:.0f}ms]: {asr_result['text']}")
        
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
        logger.info(f"🔊 TTS [{tts_result['latency_ms']:.0f}ms]: {len(tts_result['audio'])} samples")
        logger.info(f"⏱️ Total: {result['latencies']['total']:.0f}ms")
        
        return result


# ============================================================
# FastAPI Server
# ============================================================

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Telugu Voice Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; font-size: 2em; }
        .card {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
        }
        .record-btn {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 4px solid white;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            font-size: 40px;
            cursor: pointer;
            margin: 20px auto;
            display: block;
            transition: all 0.3s;
        }
        .record-btn:hover { transform: scale(1.1); }
        .record-btn.recording {
            animation: pulse 1s infinite;
            background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,255,255,0.5); }
            50% { transform: scale(1.05); box-shadow: 0 0 20px 10px rgba(255,255,255,0.3); }
        }
        .status {
            text-align: center;
            font-size: 1.2em;
            margin: 15px 0;
            min-height: 30px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .stat {
            background: rgba(0,0,0,0.2);
            padding: 15px 10px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-val { font-size: 1.8em; font-weight: bold; }
        .stat-label { font-size: 0.8em; opacity: 0.8; }
        .chat {
            max-height: 300px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
        }
        .msg {
            margin: 10px 0;
            padding: 12px 15px;
            border-radius: 15px;
            max-width: 85%;
        }
        .user {
            background: rgba(52, 152, 219, 0.5);
            margin-left: auto;
            text-align: right;
        }
        .bot {
            background: rgba(46, 204, 113, 0.5);
        }
        .telugu { font-size: 1.1em; }
        .instructions {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .instructions li { margin: 8px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu Voice Agent</h1>
        
        <div class="card">
            <button class="record-btn" id="recordBtn" onmousedown="startRec()" onmouseup="stopRec()" ontouchstart="startRec()" ontouchend="stopRec()">🎙️</button>
            <div class="status" id="status">Press and hold to speak</div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-val" id="asrMs">--</div>
                    <div class="stat-label">ASR ms</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="llmMs">--</div>
                    <div class="stat-label">LLM ms</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="ttsMs">--</div>
                    <div class="stat-label">TTS ms</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="totalMs">--</div>
                    <div class="stat-label">Total ms</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 15px;">💬 Conversation</h3>
            <div class="chat" id="chat">
                <div class="msg bot">
                    <span class="telugu">నమస్కారం! నేను మీ తెలుగు అసిస్టెంట్. ఎలా సహాయం చేయగలను?</span>
                </div>
            </div>
        </div>
        
        <div class="card instructions">
            <h3>📋 How to use:</h3>
            <ul>
                <li><b>Press and hold</b> the microphone button</li>
                <li>Speak in Telugu: <span class="telugu">"నమస్కారం, మీ పేరు ఏమిటి?"</span></li>
                <li><b>Release</b> to get response</li>
                <li>🎧 Use headphones for best experience!</li>
            </ul>
        </div>
    </div>
    
    <script>
        let mediaRecorder, audioChunks = [], ws;
        
        // Connect WebSocket
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws`);
        ws.binaryType = 'arraybuffer';
        
        ws.onopen = () => setStatus('✅ Connected - Hold button to speak');
        ws.onclose = () => { setStatus('❌ Disconnected'); setTimeout(() => location.reload(), 2000); };
        
        ws.onmessage = (e) => {
            if (typeof e.data === 'string') {
                const d = JSON.parse(e.data);
                if (d.latencies) {
                    document.getElementById('asrMs').textContent = d.latencies.asr?.toFixed(0) || '--';
                    document.getElementById('llmMs').textContent = d.latencies.llm?.toFixed(0) || '--';
                    document.getElementById('ttsMs').textContent = d.latencies.tts?.toFixed(0) || '--';
                    document.getElementById('totalMs').textContent = d.latencies.total?.toFixed(0) || '--';
                }
                if (d.transcription) addMsg('user', d.transcription);
                if (d.response_text) addMsg('bot', d.response_text);
                if (d.error) setStatus('❌ ' + d.error);
                else setStatus('✅ Ready - Hold button to speak');
            } else {
                playAudio(e.data);
            }
        };
        
        function setStatus(s) { document.getElementById('status').textContent = s; }
        
        function addMsg(type, text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'msg ' + type;
            div.innerHTML = '<span class="telugu">' + text + '</span>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        async function startRec() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { sampleRate: 16000, channelCount: 1 } 
                });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                audioChunks = [];
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.start();
                document.getElementById('recordBtn').classList.add('recording');
                setStatus('🔴 Recording...');
            } catch (e) {
                setStatus('❌ Mic error: ' + e.message);
            }
        }
        
        function stopRec() {
            if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
            
            mediaRecorder.onstop = async () => {
                document.getElementById('recordBtn').classList.remove('recording');
                setStatus('⏳ Processing...');
                
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                const buffer = await blob.arrayBuffer();
                
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(buffer);
                }
                
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
            };
            
            mediaRecorder.stop();
        }
        
        function playAudio(buffer) {
            const ctx = new AudioContext({ sampleRate: 24000 });
            const float32 = new Float32Array(buffer);
            const audioBuffer = ctx.createBuffer(1, float32.length, 24000);
            audioBuffer.getChannelData(0).set(float32);
            const src = ctx.createBufferSource();
            src.buffer = audioBuffer;
            src.connect(ctx.destination);
            src.start();
            setStatus('🔊 Playing response...');
            src.onended = () => setStatus('✅ Ready - Hold button to speak');
        }
    </script>
</body>
</html>
'''


def create_app(agent: TeluguVoiceAgent):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    
    app = FastAPI()
    
    @app.get("/")
    async def index():
        return HTMLResponse(HTML_PAGE)
    
    @app.websocket("/ws")
    async def websocket_handler(ws: WebSocket):
        await ws.accept()
        logger.info("🔗 Client connected")
        
        try:
            while True:
                data = await ws.receive_bytes()
                
                try:
                    # Decode WebM audio using pydub
                    from pydub import AudioSegment
                    
                    # Save to temp file (pydub needs file for webm)
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                        f.write(data)
                        temp_path = f.name
                    
                    # Load and convert
                    audio_seg = AudioSegment.from_file(temp_path, format='webm')
                    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
                    
                    # Convert to numpy
                    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
                    samples = samples / 32768.0
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    logger.info(f"📥 Received {len(samples)} samples ({len(samples)/16000:.1f}s)")
                    
                    # Process
                    result = await agent.process_async(samples, 16000)
                    
                    # Send JSON response
                    await ws.send_json({
                        "transcription": result["transcription"],
                        "response_text": result["response_text"],
                        "latencies": result["latencies"]
                    })
                    
                    # Send audio
                    if result["audio"] is not None:
                        audio_bytes = result["audio"].astype(np.float32).tobytes()
                        await ws.send_bytes(audio_bytes)
                        
                except Exception as e:
                    logger.error(f"❌ Error: {e}")
                    await ws.send_json({"error": str(e)})
                    
        except WebSocketDisconnect:
            logger.info("🔌 Client disconnected")
    
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--whisper", default="openai/whisper-large-v3")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    
    config = AgentConfig(whisper_model=args.whisper, llm_model=args.llm)
    agent = TeluguVoiceAgent(config)
    
    app = create_app(agent)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
