#!/usr/bin/env python3
"""
Telugu Voice Agent - Complete Pipeline
ASR (Whisper) → LLM (Qwen2.5) → TTS (Indic Parler-TTS)

Real-time Telugu conversation with ultra-low latency
"""

import torch
import numpy as np
import time
import logging
import asyncio
import json
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass

# Audio
import torchaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the voice agent"""
    # ASR
    whisper_model: str = "openai/whisper-large-v3"
    whisper_device: str = "cuda"
    
    # LLM
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.7
    
    # TTS
    tts_model: str = "ai4bharat/indic-parler-tts"
    tts_device: str = "cuda"
    tts_speaker: str = "Meera"  # Telugu female voice
    
    # Audio
    sample_rate: int = 16000
    
    # System prompt
    system_prompt: str = """You are a helpful Telugu voice assistant. 
Always respond in Telugu language (తెలుగు).
Keep responses concise and natural for speech.
Be friendly and conversational."""


class WhisperASR:
    """Whisper-based Automatic Speech Recognition for Telugu"""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.device = device
        logger.info(f"📥 Loading Whisper ASR: {model_name}")
        
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        
        # Force Telugu language
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="telugu", 
            task="transcribe"
        )
        
        logger.info("✅ Whisper ASR ready")
    
    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe audio to Telugu text"""
        start_time = time.perf_counter()
        
        # Resample if needed
        if sample_rate != 16000:
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
            audio = audio_tensor.numpy()
        
        # Process audio
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)
        
        # Generate transcription
        predicted_ids = self.model.generate(
            inputs,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=256
        )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "text": transcription.strip(),
            "language": "te",
            "latency_ms": latency
        }


class QwenLLM:
    """Qwen2.5 LLM for Telugu response generation"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        system_prompt: str = None
    ):
        self.device = device
        logger.info(f"📥 Loading LLM: {model_name}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.system_prompt = system_prompt or """You are a helpful Telugu voice assistant.
Always respond in Telugu language (తెలుగు).
Keep responses concise and natural for speech.
Be friendly and conversational."""
        
        self.conversation_history = []
        
        logger.info("✅ LLM ready")
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    @torch.no_grad()
    def generate(
        self, 
        user_input: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ) -> dict:
        """Generate Telugu response"""
        start_time = time.perf_counter()
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode response only (not the prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep history limited
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return {
            "text": response.strip(),
            "latency_ms": latency
        }


class IndicTTS:
    """Indic Parler-TTS for Telugu speech synthesis"""
    
    def __init__(
        self,
        model_name: str = "ai4bharat/indic-parler-tts",
        device: str = "cuda"
    ):
        self.device = device
        logger.info(f"📥 Loading TTS: {model_name}")
        
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        self.sample_rate = 22050  # Indic Parler-TTS output rate
        
        logger.info("✅ TTS ready")
    
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker: str = "Meera",
        emotion: str = "neutral"
    ) -> dict:
        """Synthesize Telugu speech from text"""
        start_time = time.perf_counter()
        
        # Create description for the voice
        description = f"{speaker} speaks in a clear, {emotion} tone with natural Telugu pronunciation. The recording is high quality with no background noise."
        
        # Tokenize
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        # Generate audio
        generation = self.model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )
        
        audio = generation.cpu().numpy().squeeze()
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "latency_ms": latency
        }


class TeluguVoiceAgent:
    """Complete Telugu Voice Agent Pipeline"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("="*60)
        logger.info("🎤 TELUGU VOICE AGENT")
        logger.info("="*60)
        logger.info(f"Device: {self.device}")
        
        # Initialize components
        self._init_asr()
        self._init_llm()
        self._init_tts()
        
        logger.info("="*60)
        logger.info("✅ Voice Agent Ready!")
        logger.info("="*60)
    
    def _init_asr(self):
        """Initialize ASR"""
        try:
            self.asr = WhisperASR(
                model_name=self.config.whisper_model,
                device=self.config.whisper_device
            )
        except Exception as e:
            logger.error(f"Failed to load ASR: {e}")
            self.asr = None
    
    def _init_llm(self):
        """Initialize LLM"""
        try:
            self.llm = QwenLLM(
                model_name=self.config.llm_model,
                device=self.config.llm_device,
                system_prompt=self.config.system_prompt
            )
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.llm = None
    
    def _init_tts(self):
        """Initialize TTS"""
        try:
            self.tts = IndicTTS(
                model_name=self.config.tts_model,
                device=self.config.tts_device
            )
        except Exception as e:
            logger.error(f"Failed to load TTS: {e}")
            self.tts = None
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> dict:
        """
        Full pipeline: Audio → Text → Response → Audio
        
        Returns dict with:
        - transcription: What user said
        - response_text: LLM response in Telugu
        - response_audio: Synthesized audio
        - latencies: Breakdown of timing
        """
        total_start = time.perf_counter()
        result = {
            "transcription": None,
            "response_text": None,
            "response_audio": None,
            "sample_rate": self.config.sample_rate,
            "latencies": {}
        }
        
        # Step 1: ASR - Transcribe Telugu speech
        if self.asr:
            asr_result = self.asr.transcribe(audio, sample_rate)
            result["transcription"] = asr_result["text"]
            result["latencies"]["asr_ms"] = asr_result["latency_ms"]
            logger.info(f"🎤 ASR [{asr_result['latency_ms']:.0f}ms]: {asr_result['text']}")
        else:
            logger.warning("ASR not available")
            return result
        
        # Skip if empty transcription
        if not result["transcription"] or len(result["transcription"].strip()) < 2:
            logger.warning("Empty transcription, skipping")
            return result
        
        # Step 2: LLM - Generate Telugu response
        if self.llm:
            llm_result = self.llm.generate(
                result["transcription"],
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            result["response_text"] = llm_result["text"]
            result["latencies"]["llm_ms"] = llm_result["latency_ms"]
            logger.info(f"🤖 LLM [{llm_result['latency_ms']:.0f}ms]: {llm_result['text'][:100]}...")
        else:
            logger.warning("LLM not available")
            return result
        
        # Step 3: TTS - Synthesize Telugu speech
        if self.tts:
            tts_result = self.tts.synthesize(
                result["response_text"],
                speaker=self.config.tts_speaker
            )
            result["response_audio"] = tts_result["audio"]
            result["sample_rate"] = tts_result["sample_rate"]
            result["latencies"]["tts_ms"] = tts_result["latency_ms"]
            logger.info(f"🔊 TTS [{tts_result['latency_ms']:.0f}ms]: Generated {len(tts_result['audio'])} samples")
        else:
            logger.warning("TTS not available")
        
        # Total latency
        result["latencies"]["total_ms"] = (time.perf_counter() - total_start) * 1000
        logger.info(f"⏱️ Total latency: {result['latencies']['total_ms']:.0f}ms")
        
        return result
    
    def reset(self):
        """Reset conversation history"""
        if self.llm:
            self.llm.reset_conversation()


# ============================================================
# FastAPI Server for Real-time Streaming
# ============================================================

def create_server(agent: TeluguVoiceAgent):
    """Create FastAPI server for the voice agent"""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    
    app = FastAPI(title="Telugu Voice Agent")
    
    HTML_CONTENT = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Telugu Voice Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .btn {
            padding: 15px 40px;
            font-size: 18px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin: 5px;
        }
        .btn-record {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
        }
        .btn-record.recording {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .btn-stop {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
        }
        .conversation {
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }
        .message {
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
        }
        .user { background: rgba(52, 152, 219, 0.3); text-align: right; }
        .assistant { background: rgba(46, 204, 113, 0.3); }
        .telugu { font-size: 1.2em; }
        .latency { font-size: 0.8em; color: #888; margin-top: 5px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .stat-box {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #00cec9; }
        .stat-label { font-size: 0.9em; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu Voice Agent</h1>
        <p class="subtitle">Whisper ASR → Qwen2.5 LLM → Indic Parler TTS</p>
        
        <div class="card">
            <div style="text-align: center;">
                <button class="btn btn-record" id="recordBtn" onclick="toggleRecording()">
                    🎙️ Hold to Speak
                </button>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value" id="asrLatency">--</div>
                    <div class="stat-label">ASR (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="llmLatency">--</div>
                    <div class="stat-label">LLM (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="ttsLatency">--</div>
                    <div class="stat-label">TTS (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="totalLatency">--</div>
                    <div class="stat-label">Total (ms)</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 15px;">💬 Conversation</h3>
            <div class="conversation" id="conversation">
                <div class="message assistant">
                    <div class="telugu">నమస్కారం! నేను మీ తెలుగు వాయిస్ అసిస్టెంట్. మీకు ఎలా సహాయం చేయగలను?</div>
                    <div class="latency">Welcome message</div>
                </div>
            </div>
        </div>
        
        <div class="card" style="background: rgba(46, 204, 113, 0.2); border-left: 4px solid #2ecc71;">
            <h3>📋 Instructions</h3>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li>Click and hold <b>"Hold to Speak"</b> to record</li>
                <li>Speak in Telugu: <span class="telugu">"నమస్కారం, ఎలా ఉన్నారు?"</span></li>
                <li>Release to send - you'll hear the Telugu response!</li>
                <li>Use headphones for best experience 🎧</li>
            </ul>
        </div>
    </div>
    
    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let ws = null;
        let isRecording = false;
        
        // Connect WebSocket
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/agent`);
            ws.binaryType = 'arraybuffer';
            
            ws.onopen = () => console.log('Connected');
            ws.onmessage = handleMessage;
            ws.onclose = () => setTimeout(connect, 1000);
        }
        
        function handleMessage(event) {
            if (typeof event.data === 'string') {
                const data = JSON.parse(event.data);
                
                // Update stats
                if (data.latencies) {
                    document.getElementById('asrLatency').textContent = 
                        data.latencies.asr_ms?.toFixed(0) || '--';
                    document.getElementById('llmLatency').textContent = 
                        data.latencies.llm_ms?.toFixed(0) || '--';
                    document.getElementById('ttsLatency').textContent = 
                        data.latencies.tts_ms?.toFixed(0) || '--';
                    document.getElementById('totalLatency').textContent = 
                        data.latencies.total_ms?.toFixed(0) || '--';
                }
                
                // Add messages to conversation
                if (data.transcription) {
                    addMessage('user', data.transcription);
                }
                if (data.response_text) {
                    addMessage('assistant', data.response_text);
                }
            } else {
                // Audio data - play it
                playAudio(event.data);
            }
        }
        
        function addMessage(role, text) {
            const conv = document.getElementById('conversation');
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            msg.innerHTML = `<div class="telugu">${text}</div>`;
            conv.appendChild(msg);
            conv.scrollTop = conv.scrollHeight;
        }
        
        async function playAudio(arrayBuffer) {
            const audioContext = new AudioContext({ sampleRate: 22050 });
            const float32Data = new Float32Array(arrayBuffer);
            const audioBuffer = audioContext.createBuffer(1, float32Data.length, 22050);
            audioBuffer.getChannelData(0).set(float32Data);
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start();
        }
        
        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            
            if (!isRecording) {
                // Start recording
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = sendAudio;
                
                mediaRecorder.start();
                isRecording = true;
                btn.textContent = '🔴 Recording...';
                btn.classList.add('recording');
            } else {
                // Stop recording
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                isRecording = false;
                btn.textContent = '🎙️ Hold to Speak';
                btn.classList.remove('recording');
            }
        }
        
        async function sendAudio() {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const arrayBuffer = await blob.arrayBuffer();
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(arrayBuffer);
            }
        }
        
        connect();
    </script>
</body>
</html>
'''
    
    @app.get("/")
    async def get_index():
        return HTMLResponse(content=HTML_CONTENT)
    
    @app.websocket("/ws/agent")
    async def websocket_agent(websocket: WebSocket):
        await websocket.accept()
        logger.info("🔗 Client connected")
        
        try:
            while True:
                # Receive audio data
                data = await websocket.receive_bytes()
                
                # Convert webm to numpy array
                import io
                import soundfile as sf
                
                # Decode audio
                audio_io = io.BytesIO(data)
                try:
                    audio, sr = sf.read(audio_io)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                except Exception as e:
                    logger.error(f"Audio decode error: {e}")
                    continue
                
                # Process through agent
                result = agent.process(audio, sr)
                
                # Send response
                await websocket.send_json({
                    "transcription": result["transcription"],
                    "response_text": result["response_text"],
                    "latencies": result["latencies"]
                })
                
                # Send audio if available
                if result["response_audio"] is not None:
                    audio_bytes = result["response_audio"].astype(np.float32).tobytes()
                    await websocket.send_bytes(audio_bytes)
                    
        except WebSocketDisconnect:
            logger.info("🔌 Client disconnected")
    
    return app


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Telugu Voice Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--whisper", default="openai/whisper-large-v3")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    
    # Create config
    config = AgentConfig(
        whisper_model=args.whisper,
        llm_model=args.llm
    )
    
    # Initialize agent
    agent = TeluguVoiceAgent(config)
    
    # Create and run server
    app = create_server(agent)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
