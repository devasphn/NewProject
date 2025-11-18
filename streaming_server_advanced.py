#!/usr/bin/env python3
"""
Advanced Telugu S2S Streaming Server
Full-duplex, interruption handling, context management
Ultra-low latency with <150ms response time
"""

import asyncio
import json
import time
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Import our models
from telugu_codec import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig, EMOTION_IDS, SPEAKER_IDS
from speaker_embeddings import SpeakerEmbeddingSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Stores conversation context for continuity"""
    turns: deque  # Last 10 conversation turns
    current_emotion: str = "neutral"
    current_speaker: str = "female_young"
    user_preferences: Dict = None
    session_start: float = 0
    total_interactions: int = 0
    
    def __init__(self, max_turns: int = 10):
        self.turns = deque(maxlen=max_turns)
        self.user_preferences = {}
        self.session_start = time.time()
    
    def add_turn(self, user_input: str, bot_response: str, emotion: str = None):
        """Add a conversation turn to history"""
        self.turns.append({
            "user": user_input,
            "bot": bot_response,
            "emotion": emotion or self.current_emotion,
            "timestamp": time.time()
        })
        self.total_interactions += 1
    
    def get_context_summary(self) -> str:
        """Get summarized context for model"""
        if not self.turns:
            return ""
        
        # Get last 3 turns for immediate context
        recent_turns = list(self.turns)[-3:]
        context = []
        for turn in recent_turns:
            context.append(f"User: {turn['user'][:50]}...")
            context.append(f"Bot: {turn['bot'][:50]}...")
        return " ".join(context)

@dataclass
class StreamingConfig:
    """Configuration for streaming modes"""
    mode: str = "stream"  # "stream" or "turn"
    enable_interruption: bool = True
    vad_threshold: float = 0.3
    silence_duration: float = 0.5  # seconds
    chunk_size_ms: int = 100
    lookahead_chunks: int = 2
    
class FullDuplexStreamingServer:
    """
    Advanced streaming server with:
    - Full-duplex communication
    - Interruption handling
    - Context management
    - Stream and turn modes
    """
    
    def __init__(self, model_dir: str = "models/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models(model_dir)
        
        # Session management
        self.sessions: Dict[str, ConversationContext] = {}
        self.active_streams: Dict[str, bool] = {}
        
        # Audio processing queues
        self.input_queues: Dict[str, Queue] = {}
        self.output_queues: Dict[str, Queue] = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_requests": 0,
            "average_latency": 0,
            "interruptions": 0
        }
    
    def _load_models(self, model_dir: str):
        """Load all required models"""
        try:
            # Load codec
            self.codec = TeluCodec().to(self.device)
            codec_path = f"{model_dir}/best_codec.pt"
            checkpoint = torch.load(codec_path, map_location=self.device)
            self.codec.load_state_dict(checkpoint["model_state"])
            self.codec.eval()
            logger.info("✓ Codec loaded")
            
            # Load S2S model
            config = S2SConfig(use_flash_attn=True)
            self.s2s_model = TeluguS2STransformer(config).to(self.device)
            s2s_path = f"{model_dir}/s2s_best.pt"
            checkpoint = torch.load(s2s_path, map_location=self.device)
            self.s2s_model.load_state_dict(checkpoint["model_state"])
            self.s2s_model.eval()
            logger.info("✓ S2S model loaded")
            
            # Load speaker embedding system
            self.speaker_system = SpeakerEmbeddingSystem().to(self.device)
            speaker_path = f"{model_dir}/speaker_embeddings.json"
            if os.path.exists(speaker_path):
                self.speaker_system.load_embeddings(speaker_path)
            logger.info("✓ Speaker system loaded")
            
            # Compile models for speed
            if hasattr(torch, 'compile'):
                self.codec = torch.compile(self.codec, mode='reduce-overhead')
                self.s2s_model = torch.compile(self.s2s_model, mode='reduce-overhead')
                logger.info("✓ Models compiled with torch.compile()")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def handle_websocket(
        self,
        websocket: WebSocket,
        session_id: str,
        config: StreamingConfig
    ):
        """Main WebSocket handler with full-duplex support"""
        
        # Initialize session
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext()
            self.input_queues[session_id] = Queue()
            self.output_queues[session_id] = Queue()
            self.stats["total_sessions"] += 1
        
        self.active_streams[session_id] = True
        self.stats["active_sessions"] += 1
        
        # Start processing threads
        input_task = asyncio.create_task(
            self._handle_input_stream(websocket, session_id, config)
        )
        output_task = asyncio.create_task(
            self._handle_output_stream(websocket, session_id, config)
        )
        processing_task = asyncio.create_task(
            self._process_audio_pipeline(session_id, config)
        )
        
        try:
            # Wait for any task to complete (or fail)
            done, pending = await asyncio.wait(
                [input_task, output_task, processing_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
        finally:
            self.active_streams[session_id] = False
            self.stats["active_sessions"] -= 1
    
    async def _handle_input_stream(
        self,
        websocket: WebSocket,
        session_id: str,
        config: StreamingConfig
    ):
        """Handle incoming audio from user"""
        vad_buffer = []
        silence_start = None
        
        while self.active_streams.get(session_id, False):
            try:
                # Receive audio chunk
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                
                if data["type"] == "audio":
                    audio_data = np.frombuffer(
                        base64.b64decode(data["audio"]),
                        dtype=np.float32
                    )
                    
                    # Voice Activity Detection
                    energy = np.sqrt(np.mean(audio_data**2))
                    
                    if energy > config.vad_threshold:
                        # Voice detected
                        if config.enable_interruption and self._is_bot_speaking(session_id):
                            # Interrupt bot speech
                            await self._handle_interruption(session_id)
                            self.stats["interruptions"] += 1
                        
                        vad_buffer.append(audio_data)
                        silence_start = None
                        
                    else:
                        # Silence detected
                        if silence_start is None:
                            silence_start = time.time()
                        
                        if vad_buffer and (time.time() - silence_start) > config.silence_duration:
                            # End of utterance detected
                            if config.mode == "turn":
                                # Process complete utterance in turn mode
                                full_audio = np.concatenate(vad_buffer)
                                self.input_queues[session_id].put(("audio", full_audio))
                                vad_buffer = []
                            
                    if config.mode == "stream" and vad_buffer:
                        # Stream mode: process chunks immediately
                        if len(vad_buffer) >= config.lookahead_chunks:
                            chunk = np.concatenate(vad_buffer[:config.lookahead_chunks])
                            self.input_queues[session_id].put(("audio_chunk", chunk))
                            vad_buffer = vad_buffer[1:]  # Sliding window
                
                elif data["type"] == "config":
                    # Update configuration
                    await self._update_config(session_id, data["config"])
                
                elif data["type"] == "interrupt":
                    # Manual interruption
                    await self._handle_interruption(session_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Input stream error: {e}")
                break
    
    async def _handle_output_stream(
        self,
        websocket: WebSocket,
        session_id: str,
        config: StreamingConfig
    ):
        """Handle outgoing audio to user"""
        while self.active_streams.get(session_id, False):
            try:
                # Check output queue
                if not self.output_queues[session_id].empty():
                    output_type, data = self.output_queues[session_id].get(timeout=0.1)
                    
                    if output_type == "audio":
                        # Send audio chunk
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64.b64encode(data.tobytes()).decode(),
                            "timestamp": time.time()
                        })
                    
                    elif output_type == "metadata":
                        # Send metadata (emotion, speaker info)
                        await websocket.send_json({
                            "type": "metadata",
                            "data": data,
                            "timestamp": time.time()
                        })
                    
                    elif output_type == "latency":
                        # Send latency information
                        await websocket.send_json({
                            "type": "latency",
                            "latency_ms": data,
                            "timestamp": time.time()
                        })
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Output stream error: {e}")
                break
    
    async def _process_audio_pipeline(
        self,
        session_id: str,
        config: StreamingConfig
    ):
        """Main audio processing pipeline"""
        context = self.sessions[session_id]
        
        while self.active_streams.get(session_id, False):
            try:
                if not self.input_queues[session_id].empty():
                    input_type, audio_data = self.input_queues[session_id].get(timeout=0.1)
                    
                    # Track request
                    self.stats["total_requests"] += 1
                    start_time = time.time()
                    
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
                    
                    # Process based on mode
                    if config.mode == "stream":
                        await self._process_streaming(
                            session_id, audio_tensor, context, start_time
                        )
                    else:  # turn mode
                        await self._process_turn_based(
                            session_id, audio_tensor, context, start_time
                        )
                    
                await asyncio.sleep(0.01)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
    
    async def _process_streaming(
        self,
        session_id: str,
        audio: torch.Tensor,
        context: ConversationContext,
        start_time: float
    ):
        """Process audio in streaming mode"""
        try:
            # Encode audio to codes
            with torch.no_grad():
                codes = self.codec.encode(audio)
            
            # Get speaker and emotion embeddings
            speaker_id = SPEAKER_IDS.get(context.current_speaker, 0)
            emotion_id = EMOTION_IDS.get(context.current_emotion, 0)
            
            speaker_tensor = torch.tensor([speaker_id], device=self.device)
            emotion_tensor = torch.tensor([emotion_id], device=self.device)
            
            # Generate speaker embedding with emotion
            speaker_embedding = self.speaker_system(
                speaker_tensor,
                accent_level=1  # Moderate Telugu accent
            )
            
            # Stream generate response
            first_chunk = True
            for chunk_codes in self.s2s_model.generate_streaming(
                codes, speaker_tensor, emotion_tensor,
                max_new_tokens=100,
                temperature=0.7
            ):
                # Check for interruption
                if not self.active_streams.get(session_id, False):
                    break
                
                # Decode to audio
                with torch.no_grad():
                    audio_chunk = self.codec.decode(chunk_codes)
                
                # Send first chunk latency
                if first_chunk:
                    latency = (time.time() - start_time) * 1000
                    self.output_queues[session_id].put(("latency", latency))
                    self.stats["average_latency"] = \
                        0.9 * self.stats["average_latency"] + 0.1 * latency
                    first_chunk = False
                
                # Send audio chunk
                audio_np = audio_chunk.squeeze().cpu().numpy()
                self.output_queues[session_id].put(("audio", audio_np))
                
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
    
    async def _process_turn_based(
        self,
        session_id: str,
        audio: torch.Tensor,
        context: ConversationContext,
        start_time: float
    ):
        """Process complete utterance in turn-based mode"""
        try:
            # Process complete audio
            with torch.no_grad():
                # Encode
                codes = self.codec.encode(audio)
                
                # Get embeddings
                speaker_id = SPEAKER_IDS.get(context.current_speaker, 0)
                emotion_id = EMOTION_IDS.get(context.current_emotion, 0)
                
                speaker_tensor = torch.tensor([speaker_id], device=self.device)
                emotion_tensor = torch.tensor([emotion_id], device=self.device)
                
                # Generate complete response
                response_codes = self.s2s_model.generate(
                    codes, speaker_tensor, emotion_tensor,
                    max_new_tokens=200,
                    temperature=0.7
                )
                
                # Decode
                response_audio = self.codec.decode(response_codes)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self.output_queues[session_id].put(("latency", latency))
            
            # Send complete audio
            audio_np = response_audio.squeeze().cpu().numpy()
            self.output_queues[session_id].put(("audio", audio_np))
            
            # Update context (simplified - would use ASR in production)
            context.add_turn(
                user_input="[Audio input]",
                bot_response="[Audio response]",
                emotion=context.current_emotion
            )
            
        except Exception as e:
            logger.error(f"Turn-based processing error: {e}")
    
    async def _handle_interruption(self, session_id: str):
        """Handle user interruption of bot speech"""
        logger.info(f"Interruption detected for session {session_id}")
        
        # Clear output queue
        while not self.output_queues[session_id].empty():
            try:
                self.output_queues[session_id].get_nowait()
            except Empty:
                break
        
        # Send interruption signal
        self.output_queues[session_id].put(("metadata", {"interrupted": True}))
    
    def _is_bot_speaking(self, session_id: str) -> bool:
        """Check if bot is currently speaking"""
        return not self.output_queues[session_id].empty()
    
    async def _update_config(self, session_id: str, config: Dict):
        """Update session configuration"""
        context = self.sessions[session_id]
        
        if "speaker" in config:
            context.current_speaker = config["speaker"]
        
        if "emotion" in config:
            context.current_emotion = config["emotion"]
        
        if "preferences" in config:
            context.user_preferences.update(config["preferences"])
        
        logger.info(f"Updated config for session {session_id}: {config}")

# FastAPI app
app = FastAPI(title="Telugu S2S Advanced Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global server instance
server = None

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    global server
    server = FullDuplexStreamingServer(model_dir="/workspace/models")
    logger.info("✓ Server initialized")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming"""
    await websocket.accept()
    
    # Get initial configuration
    init_data = await websocket.receive_json()
    session_id = init_data.get("session_id", str(time.time()))
    
    config = StreamingConfig(
        mode=init_data.get("mode", "stream"),
        enable_interruption=init_data.get("interruption", True),
        vad_threshold=init_data.get("vad_threshold", 0.3),
        chunk_size_ms=init_data.get("chunk_size", 100)
    )
    
    logger.info(f"New connection: {session_id} in {config.mode} mode")
    
    try:
        await server.handle_websocket(websocket, session_id, config)
    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_id in server.active_streams:
            server.active_streams[session_id] = False

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return JSONResponse(server.stats)

@app.get("/speakers")
async def get_speakers():
    """Get available speakers"""
    return JSONResponse(server.speaker_system.speakers)

@app.get("/")
async def root():
    """Serve demo page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telugu S2S Advanced Demo</title>
        <style>
            body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .controls { display: flex; gap: 20px; margin: 20px 0; }
            .control-group { flex: 1; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            .active { background: #4CAF50; color: white; }
            .stats { background: #f0f0f0; padding: 10px; border-radius: 5px; }
            .mode-switch { display: flex; gap: 10px; margin: 20px 0; }
            select { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>🎤 Telugu S2S Advanced System</h1>
        <p>Full-duplex streaming with interruption handling and context management</p>
        
        <div class="mode-switch">
            <button onclick="setMode('stream')" class="mode-btn active">Stream Mode</button>
            <button onclick="setMode('turn')" class="mode-btn">Turn Mode</button>
            <label>
                <input type="checkbox" id="interruption" checked> Enable Interruption
            </label>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <h3>Speaker</h3>
                <select id="speaker" onchange="updateConfig()">
                    <option value="male_young">Arjun (Young Male)</option>
                    <option value="male_mature">Ravi (Mature Male)</option>
                    <option value="female_young" selected>Priya (Young Female)</option>
                    <option value="female_professional">Lakshmi (Professional)</option>
                </select>
            </div>
            
            <div class="control-group">
                <h3>Emotion</h3>
                <select id="emotion" onchange="updateConfig()">
                    <option value="neutral">Neutral</option>
                    <option value="happy">Happy</option>
                    <option value="laugh">Laughing</option>
                    <option value="excited">Excited</option>
                    <option value="empathy">Empathetic</option>
                    <option value="surprise">Surprised</option>
                    <option value="thinking">Thinking</option>
                    <option value="telugu_heavy">Heavy Telugu Accent</option>
                    <option value="telugu_mild">Mild Telugu Accent</option>
                </select>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="startRecording()" id="startBtn">🎤 Start Talking</button>
            <button onclick="stopRecording()" id="stopBtn" disabled>⏹️ Stop</button>
            <button onclick="interrupt()" id="interruptBtn">🖐️ Interrupt Bot</button>
        </div>
        
        <div class="stats" id="stats">
            <p>Mode: <span id="currentMode">stream</span></p>
            <p>Latency: <span id="latency">--</span> ms</p>
            <p>Status: <span id="status">Ready</span></p>
            <p>Interactions: <span id="interactions">0</span></p>
        </div>
        
        <script>
            let ws;
            let audioContext;
            let mediaStream;
            let processor;
            let mode = 'stream';
            let sessionId = Date.now().toString();
            let interactionCount = 0;
            
            function initWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = () => {
                    ws.send(JSON.stringify({
                        session_id: sessionId,
                        mode: mode,
                        interruption: document.getElementById('interruption').checked,
                        chunk_size: 100
                    }));
                    document.getElementById('status').textContent = 'Connected';
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'audio') {
                        playAudio(data.audio);
                    } else if (data.type === 'latency') {
                        document.getElementById('latency').textContent = data.latency_ms.toFixed(1);
                    } else if (data.type === 'metadata' && data.data.interrupted) {
                        console.log('Bot was interrupted');
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    document.getElementById('status').textContent = 'Error';
                };
                
                ws.onclose = () => {
                    document.getElementById('status').textContent = 'Disconnected';
                };
            }
            
            function setMode(newMode) {
                mode = newMode;
                document.querySelectorAll('.mode-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                event.target.classList.add('active');
                document.getElementById('currentMode').textContent = mode;
                
                // Reconnect with new mode
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
                initWebSocket();
            }
            
            function updateConfig() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'config',
                        config: {
                            speaker: document.getElementById('speaker').value,
                            emotion: document.getElementById('emotion').value
                        }
                    }));
                }
            }
            
            function interrupt() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'interrupt'}));
                }
            }
            
            async function startRecording() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
                mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(1024, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const float32Array = new Float32Array(inputData);
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'audio',
                            audio: btoa(String.fromCharCode(...new Uint8Array(float32Array.buffer)))
                        }));
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').textContent = 'Recording';
            }
            
            function stopRecording() {
                if (processor) {
                    processor.disconnect();
                    processor = null;
                }
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('status').textContent = 'Stopped';
                
                interactionCount++;
                document.getElementById('interactions').textContent = interactionCount;
            }
            
            function playAudio(base64Audio) {
                const audioData = atob(base64Audio);
                const arrayBuffer = new ArrayBuffer(audioData.length);
                const view = new Uint8Array(arrayBuffer);
                
                for (let i = 0; i < audioData.length; i++) {
                    view[i] = audioData.charCodeAt(i);
                }
                
                audioContext.decodeAudioData(arrayBuffer, (buffer) => {
                    const source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start(0);
                });
            }
            
            // Initialize on load
            window.onload = () => {
                initWebSocket();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_dir", default="/workspace/models")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Run server
    uvicorn.run(
        "streaming_server_advanced:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False
    )