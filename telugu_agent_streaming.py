#!/usr/bin/env python3
"""
Telugu Voice Agent - Continuous Real-time Streaming
With Voice Activity Detection (VAD) - No buttons needed!

Audio streams continuously → VAD detects speech → Process → Respond
"""

import torch
import numpy as np
import time
import logging
import asyncio
import io
import tempfile
import os
import collections
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    whisper_model: str = "openai/whisper-large-v3"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    tts_voice: str = "te-IN-ShrutiNeural"
    sample_rate: int = 16000
    chunk_size: int = 4096  # Must match browser (power of 2)
    chunk_duration_ms: int = 256  # 4096/16000*1000 = 256ms
    silence_threshold: float = 0.01  # VAD threshold
    min_speech_duration: float = 0.5  # Min 0.5s speech to process
    max_speech_duration: float = 10.0  # Max 10s per utterance
    silence_duration: float = 0.8  # 0.8s silence = end of utterance


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
        
        # Ensure audio is long enough (pad if needed)
        min_samples = sample_rate  # At least 1 second
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
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
        
        # Create attention mask
        attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
        
        predicted_ids = self.model.generate(
            inputs,
            attention_mask=attention_mask,
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
        
        self.system_prompt = """You are a helpful Telugu voice assistant. నీవు తెలుగు వాయిస్ అసిస్టెంట్.
Rules:
1. ALWAYS respond in Telugu (తెలుగు) only
2. Keep responses SHORT (1-2 sentences max)
3. Be conversational and friendly
4. If user speaks Telugu, respond in Telugu"""
        
        self.history = []
        logger.info("✅ LLM ready")
    
    @torch.no_grad()
    def generate(self, user_input: str) -> dict:
        start = time.perf_counter()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-4:])
        messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
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
        logger.info(f"📥 Edge TTS: {voice}")
        logger.info("✅ TTS ready")
    
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
    """Manages audio buffering with VAD"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_start_time = None
        
    def add_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk and return complete utterance if speech ended.
        Returns None if still listening.
        """
        # Calculate RMS energy for VAD
        rms = np.sqrt(np.mean(audio ** 2))
        is_speech = rms > self.config.silence_threshold
        
        if is_speech:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.silence_frames = 0
                logger.info("🎤 Speech detected...")
            
            self.buffer.append(audio)
            self.silence_frames = 0
            
            # Check max duration
            duration = len(np.concatenate(self.buffer)) / self.config.sample_rate
            if duration > self.config.max_speech_duration:
                return self._flush_buffer()
                
        else:
            if self.is_speaking:
                self.buffer.append(audio)
                self.silence_frames += 1
                
                # Check if silence duration exceeded
                silence_duration = (self.silence_frames * self.config.chunk_duration_ms) / 1000
                if silence_duration >= self.config.silence_duration:
                    return self._flush_buffer()
        
        return None
    
    def _flush_buffer(self) -> Optional[np.ndarray]:
        """Flush buffer and return complete audio"""
        if not self.buffer:
            return None
            
        audio = np.concatenate(self.buffer)
        duration = len(audio) / self.config.sample_rate
        
        self.buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        
        if duration >= self.config.min_speech_duration:
            logger.info(f"📝 Captured {duration:.1f}s of speech")
            return audio
        else:
            logger.info(f"⏭️ Too short ({duration:.1f}s), skipping")
            return None


class TeluguVoiceAgent:
    """Complete Telugu Voice Agent with streaming"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        logger.info("=" * 60)
        logger.info("🎤 TELUGU VOICE AGENT - STREAMING")
        logger.info("=" * 60)
        
        self.asr = WhisperASR(self.config.whisper_model)
        self.llm = QwenLLM(self.config.llm_model)
        self.tts = EdgeTTS(self.config.tts_voice)
        
        logger.info("=" * 60)
        logger.info("✅ Agent Ready!")
        logger.info("=" * 60)
    
    async def process(self, audio: np.ndarray) -> dict:
        """Process complete utterance"""
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
        
        # Skip empty or too short
        text = result["transcription"]
        if not text or len(text) < 2 or text in [".", ",", "...", " "]:
            logger.info("⏭️ Empty transcription, skipping")
            return result
        
        # LLM
        llm_result = self.llm.generate(text)
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


# HTML with real-time streaming
HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Telugu Voice Agent - Streaming</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }
        .container { max-width: 750px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 25px; }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
        }
        .mic-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
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
        .mic-btn.active {
            border-color: #f44336;
            background: rgba(244, 67, 54, 0.3);
            animation: pulse 1.5s infinite;
        }
        .mic-btn.speaking {
            border-color: #FF9800;
            background: rgba(255, 152, 0, 0.5);
        }
        .mic-btn.processing {
            border-color: #2196F3;
            background: rgba(33, 150, 243, 0.3);
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244,67,54,0.5); }
            50% { transform: scale(1.05); box-shadow: 0 0 20px 10px rgba(244,67,54,0.2); }
        }
        .status {
            font-size: 1.1em;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            background: rgba(0,0,0,0.2);
        }
        .visualizer {
            height: 60px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            margin: 15px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .visualizer canvas { width: 100%; height: 100%; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .stat {
            background: rgba(0,0,0,0.3);
            padding: 12px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-val { font-size: 1.5em; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.75em; color: #aaa; }
        .chat {
            max-height: 350px;
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
        .user { background: rgba(33, 150, 243, 0.4); margin-left: auto; text-align: right; }
        .bot { background: rgba(76, 175, 80, 0.4); }
        .telugu { font-size: 1.15em; line-height: 1.5; }
        .info-box {
            background: rgba(33, 150, 243, 0.2);
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu Voice Agent</h1>
        <p class="subtitle">Real-time Streaming • Whisper + Qwen2.5 + Edge TTS</p>
        
        <div class="card">
            <div class="mic-container">
                <button class="mic-btn" id="micBtn" onclick="toggleMic()">🎙️</button>
                <div class="status" id="status">Click to start listening</div>
            </div>
            
            <div class="visualizer">
                <canvas id="visualizer"></canvas>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-val" id="asrMs">--</div>
                    <div class="stat-label">ASR (ms)</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="llmMs">--</div>
                    <div class="stat-label">LLM (ms)</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="ttsMs">--</div>
                    <div class="stat-label">TTS (ms)</div>
                </div>
                <div class="stat">
                    <div class="stat-val" id="totalMs">--</div>
                    <div class="stat-label">Total (ms)</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 15px;">💬 Conversation</h3>
            <div class="chat" id="chat">
                <div class="msg bot">
                    <span class="telugu">నమస్కారం! నేను మీ తెలుగు అసిస్టెంట్. మీతో మాట్లాడటానికి సిద్ధంగా ఉన్నాను!</span>
                </div>
            </div>
        </div>
        
        <div class="card info-box">
            <h3>📋 How it works:</h3>
            <ul style="margin: 10px 0 0 20px;">
                <li><b>Click microphone</b> to start continuous listening</li>
                <li><b>Just speak naturally</b> - it detects when you talk</li>
                <li><b>Pause for 1 second</b> to trigger processing</li>
                <li><b>Use headphones</b> to avoid echo! 🎧</li>
            </ul>
        </div>
    </div>
    
    <script>
        const SAMPLE_RATE = 16000;
        const CHUNK_SIZE = 4096;  // Must be power of 2: 256, 512, 1024, 2048, 4096, 8192, 16384
        
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;
        let analyser = null;
        let isListening = false;
        let isProcessing = false;
        
        const canvas = document.getElementById('visualizer');
        const ctx = canvas.getContext('2d');
        
        function setStatus(text, type = '') {
            document.getElementById('status').textContent = text;
            const btn = document.getElementById('micBtn');
            btn.className = 'mic-btn ' + type;
        }
        
        function addMsg(role, text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'msg ' + role;
            div.innerHTML = '<span class="telugu">' + text + '</span>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function connectWS() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';
            
            ws.onopen = () => console.log('WS connected');
            ws.onclose = () => {
                console.log('WS closed, reconnecting...');
                setTimeout(connectWS, 1000);
            };
            
            ws.onmessage = (e) => {
                if (typeof e.data === 'string') {
                    const d = JSON.parse(e.data);
                    
                    if (d.status === 'listening') {
                        setStatus('🎤 Listening... speak now!', 'active');
                    } else if (d.status === 'speaking') {
                        setStatus('🗣️ Speech detected...', 'speaking');
                    } else if (d.status === 'processing') {
                        setStatus('⏳ Processing...', 'processing');
                        isProcessing = true;
                    } else if (d.status === 'done') {
                        isProcessing = false;
                        setStatus('🎤 Listening... speak now!', 'active');
                    }
                    
                    if (d.latencies) {
                        document.getElementById('asrMs').textContent = d.latencies.asr?.toFixed(0) || '--';
                        document.getElementById('llmMs').textContent = d.latencies.llm?.toFixed(0) || '--';
                        document.getElementById('ttsMs').textContent = d.latencies.tts?.toFixed(0) || '--';
                        document.getElementById('totalMs').textContent = d.latencies.total?.toFixed(0) || '--';
                    }
                    
                    if (d.transcription) addMsg('user', d.transcription);
                    if (d.response_text) addMsg('bot', d.response_text);
                } else {
                    // Audio response
                    playAudio(e.data);
                }
            };
        }
        
        async function toggleMic() {
            if (isListening) {
                stopListening();
            } else {
                await startListening();
            }
        }
        
        async function startListening() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: SAMPLE_RATE,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                const source = audioContext.createMediaStreamSource(mediaStream);
                
                // Analyser for visualization
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                source.connect(analyser);
                
                // Script processor for streaming
                processor = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                processor.onaudioprocess = (e) => {
                    if (!isListening || isProcessing) return;
                    
                    const input = e.inputBuffer.getChannelData(0);
                    const int16 = new Int16Array(input.length);
                    for (let i = 0; i < input.length; i++) {
                        int16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
                    }
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(int16.buffer);
                    }
                };
                
                isListening = true;
                setStatus('🎤 Listening... speak now!', 'active');
                drawVisualizer();
                
            } catch (err) {
                setStatus('❌ Microphone error: ' + err.message);
            }
        }
        
        function stopListening() {
            isListening = false;
            
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(t => t.stop());
                mediaStream = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            setStatus('Click to start listening');
            document.getElementById('micBtn').className = 'mic-btn';
        }
        
        function playAudio(buffer) {
            const playCtx = new AudioContext({ sampleRate: 24000 });
            const float32 = new Float32Array(buffer);
            const audioBuffer = playCtx.createBuffer(1, float32.length, 24000);
            audioBuffer.getChannelData(0).set(float32);
            const src = playCtx.createBufferSource();
            src.buffer = audioBuffer;
            src.connect(playCtx.destination);
            src.start();
        }
        
        function drawVisualizer() {
            if (!analyser || !isListening) return;
            
            requestAnimationFrame(drawVisualizer);
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteFrequencyData(dataArray);
            
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * canvas.height;
                const hue = (i / bufferLength) * 120 + 120; // Green to cyan
                ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        }
        
        // Initialize
        connectWS();
        
        // Resize canvas
        function resizeCanvas() {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    </script>
</body>
</html>
'''


def create_app(agent: TeluguVoiceAgent, config: AgentConfig):
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
        
        buffer = AudioBuffer(config)
        
        try:
            while True:
                data = await ws.receive_bytes()
                
                # Convert Int16 to Float32
                int16_data = np.frombuffer(data, dtype=np.int16)
                float_data = int16_data.astype(np.float32) / 32768.0
                
                # Add to buffer with VAD
                complete_audio = buffer.add_chunk(float_data)
                
                # Send status updates
                if buffer.is_speaking:
                    await ws.send_json({"status": "speaking"})
                else:
                    await ws.send_json({"status": "listening"})
                
                # Process if we have complete utterance
                if complete_audio is not None:
                    await ws.send_json({"status": "processing"})
                    
                    try:
                        result = await agent.process(complete_audio)
                        
                        # Send response
                        await ws.send_json({
                            "status": "done",
                            "transcription": result["transcription"],
                            "response_text": result["response_text"],
                            "latencies": result["latencies"]
                        })
                        
                        # Send audio
                        if result["audio"] is not None:
                            audio_bytes = result["audio"].astype(np.float32).tobytes()
                            await ws.send_bytes(audio_bytes)
                            
                    except Exception as e:
                        logger.error(f"❌ Processing error: {e}")
                        await ws.send_json({"status": "done", "error": str(e)})
                        
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
    app = create_app(agent, config)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
