#!/usr/bin/env python3
"""
Real-time Telugu Codec Streaming Server
WebSocket server for browser-based real-time audio streaming

User speaks → WebSocket → Encode → Decode → WebSocket → User hears
"""

import torch
import numpy as np
import asyncio
import json
import time
import logging
import argparse
from pathlib import Path

# WebSocket and HTTP server
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("❌ FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

from telugu_codec_fixed import TeluCodec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Global codec instance
codec = None
device = None
latency_stats = []


def load_codec(codec_path: str):
    """Load the Telugu codec"""
    global codec, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")
    
    logger.info("📥 Loading Telugu Codec...")
    codec = TeluCodec().to(device)
    checkpoint = torch.load(codec_path, map_location=device, weights_only=False)
    if 'codec_state_dict' in checkpoint:
        codec.load_state_dict(checkpoint['codec_state_dict'])
    else:
        codec.load_state_dict(checkpoint)
    codec.eval()
    
    # Warmup
    logger.info("🔥 Warming up codec...")
    with torch.no_grad():
        dummy = torch.randn(1, 1, 8000).to(device)
        for _ in range(5):
            codes = codec.encode(dummy)
            _ = codec.decode(codes)
    
    logger.info("✅ Codec ready!")


@torch.no_grad()
def process_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> tuple:
    """
    Process audio through codec
    Returns: (reconstructed_audio, latency_ms)
    """
    global codec, device
    
    start_time = time.perf_counter()
    
    # Convert to tensor [1, 1, T]
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)
    
    # Normalize
    max_val = audio_tensor.abs().max()
    if max_val > 0.01:
        audio_tensor = audio_tensor / max_val * 0.9
    
    # Encode
    codes = codec.encode(audio_tensor)
    
    # Decode
    reconstructed = codec.decode(codes)
    
    # Back to numpy
    output = reconstructed.squeeze().cpu().numpy()
    
    # Match length
    target_len = len(audio_data)
    if len(output) > target_len:
        output = output[:target_len]
    elif len(output) < target_len:
        output = np.pad(output, (0, target_len - len(output)))
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return output, latency_ms


# Create FastAPI app
app = FastAPI(title="Telugu Codec Real-time Test")


# HTML for the web interface
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 Telugu Codec Real-time Test</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
            animation: pulse 2s infinite;
        }
        
        .status-dot.connected {
            background: #44ff44;
        }
        
        .status-dot.streaming {
            background: #ffaa00;
            animation: pulse 0.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 20px 0;
        }
        
        button {
            padding: 15px 40px;
            font-size: 18px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-start {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
        }
        
        .btn-start:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0, 184, 148, 0.4);
        }
        
        .btn-stop {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
        }
        
        .btn-stop:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(231, 76, 60, 0.4);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-box {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #00cec9;
        }
        
        .stat-label {
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .visualizer {
            height: 100px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .visualizer canvas {
            width: 100%;
            height: 100%;
        }
        
        .instructions {
            background: rgba(0, 184, 148, 0.2);
            border-left: 4px solid #00b894;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin-top: 20px;
        }
        
        .instructions h3 {
            margin-bottom: 10px;
        }
        
        .instructions ul {
            margin-left: 20px;
        }
        
        .instructions li {
            margin: 5px 0;
        }
        
        .telugu-text {
            font-size: 1.2em;
            color: #ffaa00;
        }
        
        .log {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        
        .log-entry {
            margin: 3px 0;
            color: #aaa;
        }
        
        .log-entry.info { color: #00cec9; }
        .log-entry.success { color: #00b894; }
        .log-entry.error { color: #e74c3c; }
        .log-entry.latency { color: #fdcb6e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu Codec Test</h1>
        <p class="subtitle">Real-time Speech Encoding & Decoding</p>
        
        <div class="card">
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Disconnected</span>
            </div>
            
            <div class="visualizer">
                <canvas id="visualizer"></canvas>
            </div>
            
            <div class="controls">
                <button class="btn-start" id="startBtn" onclick="startStreaming()">
                    🎙️ Start Streaming
                </button>
                <button class="btn-stop" id="stopBtn" onclick="stopStreaming()" disabled>
                    ⏹️ Stop
                </button>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value" id="latencyValue">--</div>
                    <div class="stat-label">Latency (ms)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="chunksValue">0</div>
                    <div class="stat-label">Chunks Processed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="avgLatencyValue">--</div>
                    <div class="stat-label">Avg Latency (ms)</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="instructions">
                <h3>📋 Instructions</h3>
                <ul>
                    <li>Click <strong>"Start Streaming"</strong> to enable your microphone</li>
                    <li>Speak in Telugu: <span class="telugu-text">"నమస్కారం, ఎలా ఉన్నారు?"</span></li>
                    <li>You'll hear your voice reconstructed through the codec</li>
                    <li>Watch the latency stats to see real-time performance</li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 10px;">📜 Log</h3>
            <div class="log" id="log"></div>
        </div>
    </div>
    
    <script>
        // Configuration
        const SAMPLE_RATE = 16000;
        const CHUNK_SIZE = 4096;  // ~256ms chunks
        
        // State
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;
        let isStreaming = false;
        let chunks = 0;
        let latencies = [];
        
        // Audio playback with proper buffering
        let playbackQueue = [];
        let isPlaying = false;
        let nextPlayTime = 0;
        let playbackContext = null;
        
        // Visualizer
        let analyser = null;
        let visualizerCanvas = document.getElementById('visualizer');
        let visualizerCtx = visualizerCanvas.getContext('2d');
        
        function log(message, type = 'info') {
            const logDiv = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function updateStatus(status, isConnected, isActive) {
            document.getElementById('statusText').textContent = status;
            const dot = document.getElementById('statusDot');
            dot.className = 'status-dot';
            if (isActive) dot.classList.add('streaming');
            else if (isConnected) dot.classList.add('connected');
        }
        
        function updateStats(latency) {
            latencies.push(latency);
            chunks++;
            
            document.getElementById('latencyValue').textContent = latency.toFixed(1);
            document.getElementById('chunksValue').textContent = chunks;
            
            const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            document.getElementById('avgLatencyValue').textContent = avg.toFixed(1);
        }
        
        async function startStreaming() {
            try {
                log('Requesting microphone access...', 'info');
                
                // Get microphone
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: SAMPLE_RATE,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                log('Microphone access granted', 'success');
                
                // Create audio context
                audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                
                // Setup analyser for visualization
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                source.connect(analyser);
                
                // Create script processor for sending audio
                processor = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                // Connect WebSocket (use wss:// for HTTPS, ws:// for HTTP)
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws/audio`;
                log(`Connecting to ${wsUrl}...`, 'info');
                
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';
                
                ws.onopen = () => {
                    log('WebSocket connected!', 'success');
                    updateStatus('Connected - Streaming', true, true);
                    isStreaming = true;
                    
                    // Reset playback timing
                    nextPlayTime = audioContext.currentTime;
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                    log('🔊 Audio playback enabled - speak now!', 'success');
                    
                    // Start visualization
                    drawVisualizer();
                };
                
                ws.onmessage = (event) => {
                    if (typeof event.data === 'string') {
                        // JSON message with stats
                        const data = JSON.parse(event.data);
                        if (data.latency) {
                            updateStats(data.latency);
                            // Only log every 10th chunk to reduce spam
                            if (chunks % 10 === 0) {
                                log(`Processed ${chunks} chunks - Avg Latency: ${data.avg_latency.toFixed(1)}ms`, 'latency');
                            }
                        }
                    } else {
                        // Binary audio data - play it!
                        playAudio(event.data);
                    }
                };
                
                ws.onerror = (error) => {
                    log(`WebSocket error: ${error}`, 'error');
                };
                
                ws.onclose = () => {
                    log('WebSocket closed', 'info');
                    stopStreaming();
                };
                
                // Send audio chunks
                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN && isStreaming) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const int16Data = float32ToInt16(inputData);
                        ws.send(int16Data.buffer);
                    }
                };
                
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
                console.error(error);
            }
        }
        
        function stopStreaming() {
            isStreaming = false;
            
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            updateStatus('Disconnected', false, false);
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            log('Streaming stopped', 'info');
        }
        
        function float32ToInt16(float32Array) {
            const int16Array = new Int16Array(float32Array.length);
            for (let i = 0; i < float32Array.length; i++) {
                const s = Math.max(-1, Math.min(1, float32Array[i]));
                int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            return int16Array;
        }
        
        function int16ToFloat32(int16Array) {
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 0x7FFF;
            }
            return float32Array;
        }
        
        async function playAudio(arrayBuffer) {
            if (!audioContext) return;
            
            try {
                const int16Data = new Int16Array(arrayBuffer);
                const float32Data = int16ToFloat32(int16Data);
                
                // Create audio buffer
                const audioBuffer = audioContext.createBuffer(1, float32Data.length, SAMPLE_RATE);
                audioBuffer.getChannelData(0).set(float32Data);
                
                // Create source
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                
                // Create gain node for volume control
                const gainNode = audioContext.createGain();
                gainNode.gain.value = 1.5;  // Boost volume slightly
                
                source.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                // Schedule playback with proper timing
                const currentTime = audioContext.currentTime;
                if (nextPlayTime < currentTime) {
                    nextPlayTime = currentTime;
                }
                
                source.start(nextPlayTime);
                nextPlayTime += audioBuffer.duration;
                
                // Log first few plays
                if (chunks <= 5) {
                    log(`Playing audio chunk (${float32Data.length} samples)`, 'success');
                }
            } catch (error) {
                console.error('Playback error:', error);
                log(`Playback error: ${error.message}`, 'error');
            }
        }
        
        function drawVisualizer() {
            if (!analyser || !isStreaming) return;
            
            requestAnimationFrame(drawVisualizer);
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteFrequencyData(dataArray);
            
            visualizerCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            visualizerCtx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
            
            const barWidth = (visualizerCanvas.width / bufferLength) * 2.5;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * visualizerCanvas.height;
                
                const gradient = visualizerCtx.createLinearGradient(0, visualizerCanvas.height, 0, 0);
                gradient.addColorStop(0, '#00cec9');
                gradient.addColorStop(1, '#00b894');
                
                visualizerCtx.fillStyle = gradient;
                visualizerCtx.fillRect(x, visualizerCanvas.height - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        }
        
        // Resize canvas
        function resizeCanvas() {
            visualizerCanvas.width = visualizerCanvas.offsetWidth;
            visualizerCanvas.height = visualizerCanvas.offsetHeight;
        }
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        log('Ready! Click "Start Streaming" to begin.', 'success');
    </script>
</body>
</html>
"""


@app.get("/")
async def get_index():
    """Serve the main page"""
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()
    logger.info("🔗 Client connected")
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Convert from int16 to float32
            int16_data = np.frombuffer(data, dtype=np.int16)
            float32_data = int16_data.astype(np.float32) / 32767.0
            
            # Process through codec
            reconstructed, latency_ms = process_audio(float32_data)
            
            # Track stats
            latency_stats.append(latency_ms)
            if len(latency_stats) > 100:
                latency_stats.pop(0)
            
            # Send latency stats
            await websocket.send_json({
                "latency": latency_ms,
                "avg_latency": np.mean(latency_stats) if latency_stats else 0
            })
            
            # Convert back to int16 and send
            int16_output = (reconstructed * 32767).astype(np.int16)
            await websocket.send_bytes(int16_output.tobytes())
            
    except WebSocketDisconnect:
        logger.info("🔌 Client disconnected")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


@app.get("/stats")
async def get_stats():
    """Get current latency statistics"""
    if not latency_stats:
        return {"status": "no data"}
    
    return {
        "current": latency_stats[-1] if latency_stats else 0,
        "average": np.mean(latency_stats),
        "min": np.min(latency_stats),
        "max": np.max(latency_stats),
        "samples": len(latency_stats)
    }


def main():
    parser = argparse.ArgumentParser(description="Real-time Telugu Codec Server")
    parser.add_argument("--codec_path", default="/workspace/models/codec/best_codec.pt",
                       help="Path to codec checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8010, help="Server port")
    args = parser.parse_args()
    
    # Load codec
    load_codec(args.codec_path)
    
    # Start server
    logger.info(f"\n{'='*60}")
    logger.info("🎤 TELUGU CODEC REAL-TIME SERVER")
    logger.info(f"{'='*60}")
    logger.info(f"Open in browser: http://{args.host}:{args.port}")
    logger.info(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
