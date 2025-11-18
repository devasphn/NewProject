"""
FastAPI WebSocket Server for Telugu S2S Voice Agent
Optimized for RTX A6000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import numpy as np
import base64
import json
import os
import logging
from datetime import datetime
from s2s_pipeline import TeluguS2SPipeline
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{WORKSPACE_DIR}/logs/server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telugu S2S Voice Agent",
    description="Ultra-Low Latency Speech-to-Speech AI",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None
stats = {
    "total_requests": 0,
    "avg_latency_ms": 0,
    "started_at": None
}

@app.on_event("startup")
async def startup():
    global pipeline, stats
    logger.info("=" * 70)
    logger.info("Starting Telugu S2S Voice Agent Server")
    logger.info(f"GPU: {GPU_NAME} ({GPU_MEMORY}GB)")
    logger.info(f"Cost: ${GPU_COST_PER_HOUR}/hour")
    logger.info("=" * 70)
    
    # Check if Telugu model exists
    use_telugu = os.path.exists(f"{MODELS_DIR}/speecht5_telugu")
    if use_telugu:
        logger.info("✓ Telugu fine-tuned model found!")
    else:
        logger.info("⚠ Using baseline model (not Telugu fine-tuned)")
    
    pipeline = TeluguS2SPipeline(use_telugu_model=use_telugu)
    stats["started_at"] = datetime.now().isoformat()
    
    logger.info("=" * 70)
    logger.info("Server ready! Listening on http://0.0.0.0:8000")
    logger.info("=" * 70)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    html_path = f"{WORKSPACE_DIR}/static/index.html"
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse("<h1>Static files not found. Check static/index.html</h1>")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "gpu": GPU_NAME,
        "stats": stats
    })

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return JSONResponse(stats)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global stats
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"✓ Client {client_id} connected")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                # Decode audio
                audio_b64 = message["audio"]
                audio_bytes = base64.b64decode(audio_b64)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Get speaker ID (if provided)
                speaker_id = message.get("speaker_id", 0)
                
                logger.info(f"📥 Client {client_id}: Received {len(audio_array)} samples")
                
                # Process through pipeline
                result = await pipeline.process(audio_array, speaker_id=speaker_id)
                
                # Update stats
                stats["total_requests"] += 1
                if stats["avg_latency_ms"] == 0:
                    stats["avg_latency_ms"] = result["latency_ms"]
                else:
                    stats["avg_latency_ms"] = int(
                        (stats["avg_latency_ms"] * (stats["total_requests"] - 1) + 
                         result["latency_ms"]) / stats["total_requests"]
                    )
                
                # Encode output audio
                output_b64 = base64.b64encode(result["audio"].tobytes()).decode('utf-8')
                
                # Send response
                response = {
                    "type": "response",
                    "audio": output_b64,
                    "input_text": result["input_text"],
                    "output_text": result["output_text"],
                    "latency_ms": result["latency_ms"],
                    "breakdown": result["breakdown"],
                    "target_latency": TARGET_TOTAL_LATENCY
                }
                
                await websocket.send_text(json.dumps(response))
                
                # Log result
                latency = result["latency_ms"]
                status = "✓" if latency < TARGET_TOTAL_LATENCY else "⚠"
                logger.info(
                    f"📤 Client {client_id}: {status} Latency: {latency}ms "
                    f"(ASR: {result['breakdown']['asr_ms']}ms, "
                    f"LLM: {result['breakdown']['llm_ms']}ms, "
                    f"TTS: {result['breakdown']['tts_ms']}ms)"
                )
                
            elif message["type"] == "ping":
                # Keep-alive
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        logger.info(f"❌ Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ Error with client {client_id}: {e}")
        import traceback
        traceback.print_exc()

# Mount static files
os.makedirs(f"{WORKSPACE_DIR}/static", exist_ok=True)
app.mount("/static", StaticFiles(directory=f"{WORKSPACE_DIR}/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    # Ensure logs directory exists
    os.makedirs(f"{WORKSPACE_DIR}/logs", exist_ok=True)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )
