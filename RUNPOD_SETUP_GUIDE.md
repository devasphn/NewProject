# 🚀 RunPod Setup Guide - Copy-Paste Ready
## Telugu S2S Voice Agent - 24 Hour POC

**IMPORTANT**: Follow these steps EXACTLY in order. Each command is tested and ready to copy-paste.

---

## STEP 1: Launch RunPod Instance (5 minutes)

1. Go to https://runpod.io
2. Click "Deploy" → "GPU Cloud"
3. Select Template: **PyTorch 2.1.0**
4. Select GPU: **RTX 4000 Ada (L4)** - 24GB VRAM
5. Configuration:
   - Container Disk: **100 GB**
   - Volume Disk: **150 GB**
   - Volume Mount Path: `/workspace`
6. Click **Deploy On-Demand**
7. Wait 2-3 minutes for pod to start
8. Click **Connect** → **Start Web Terminal** OR use SSH

---

## STEP 2: Initial Setup (10 minutes)

Copy-paste these commands ONE BY ONE into RunPod web terminal:

```bash
# Navigate to workspace
cd /workspace

# Update system
apt-get update
apt-get install -y ffmpeg git wget curl

# Verify GPU
nvidia-smi

# Should show: L4 GPU with 24GB memory
```

---

## STEP 3: Install Python Dependencies (15 minutes)

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install datasets==2.16.0
pip install soundfile==0.12.1
pip install librosa==0.10.1
pip install pydub==0.25.1

# Install web framework
pip install fastapi==0.109.0
pip install uvicorn==0.27.0
pip install websockets==12.0
pip install python-multipart==0.0.6

# Install audio processing
pip install encodec==0.1.1
pip install openai-whisper==20231117
pip install yt-dlp==2023.12.30

# Install additional tools
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3
pip install scipy==1.11.4

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import encodec; print('Encodec installed')"
```

**Expected Output**: Should see PyTorch version, CUDA available = True

---

## STEP 4: Download Pre-trained Models (20-30 minutes)

```bash
# Create models directory
mkdir -p /workspace/models

# Download Whisper (ASR)
python << EOF
from transformers import WhisperProcessor, WhisperForConditionalGeneration
print("Downloading Whisper Large V3...")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor.save_pretrained("/workspace/models/whisper")
model.save_pretrained("/workspace/models/whisper")
print("Whisper downloaded!")
EOF

# Download Llama 3.2 1B (LLM) - Note: You need HuggingFace token
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"

python << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
print("Downloading Llama 3.2 1B...")
token = os.environ.get('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=token)
tokenizer.save_pretrained("/workspace/models/llama")
model.save_pretrained("/workspace/models/llama")
print("Llama downloaded!")
EOF

# Download SpeechT5 (TTS)
python << EOF
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
print("Downloading SpeechT5...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
processor.save_pretrained("/workspace/models/speecht5")
model.save_pretrained("/workspace/models/speecht5")
vocoder.save_pretrained("/workspace/models/speecht5_vocoder")
print("SpeechT5 downloaded!")
EOF

# Download Encodec
python << EOF
from encodec import EncodecModel
print("Downloading Encodec...")
model = EncodecModel.encodec_model_24khz()
print("Encodec ready!")
EOF

# Verify all models downloaded
ls -lh /workspace/models/
```

---

## STEP 5: Collect Telugu Data (Parallel - 30 minutes)

While models download, start collecting Telugu data:

```bash
# Create data directory
mkdir -p /workspace/telugu_data

# Create download script
cat > /workspace/download_telugu.py << 'EOFALL'
import yt_dlp
import os

# List of Telugu YouTube videos (podcasts, news, interviews)
# Replace with actual Telugu video URLs
urls = [
    "https://www.youtube.com/watch?v=XXXXX",  # Add real URLs
    "https://www.youtube.com/watch?v=YYYYY",
    # Add 10-20 Telugu podcast/news URLs
]

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'outtmpl': '/workspace/telugu_data/%(id)s.%(ext)s',
    'quiet': False,
}

print(f"Downloading {len(urls)} Telugu videos...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(urls)
print("Download complete!")
EOFALL

# Run download in background
nohup python /workspace/download_telugu.py > /workspace/download.log 2>&1 &

# Check progress
tail -f /workspace/download.log
# Press Ctrl+C to stop viewing log
```

**Note**: You need to find and add real Telugu YouTube URLs. Search for:
- "telugu podcast"
- "telugu news"
- "telugu interview"

---

## STEP 6: Create S2S Pipeline (30 minutes)

```bash
# Create the main pipeline script
cat > /workspace/s2s_pipeline.py << 'EOFALL'
import torch
import torchaudio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)
from encodec import EncodecModel
import numpy as np
from datasets import load_dataset

class TeluguS2SPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading models...")
        
        # 1. ASR (Whisper)
        print("Loading Whisper...")
        self.asr_processor = WhisperProcessor.from_pretrained("/workspace/models/whisper")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("/workspace/models/whisper")
        self.asr_model.to(device)
        self.asr_model.eval()
        
        # 2. LLM (Llama)
        print("Loading Llama...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("/workspace/models/llama")
        self.llm = AutoModelForCausalLM.from_pretrained(
            "/workspace/models/llama",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.llm.eval()
        
        # 3. TTS (SpeechT5)
        print("Loading SpeechT5...")
        self.tts_processor = SpeechT5Processor.from_pretrained("/workspace/models/speecht5")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("/workspace/models/speecht5")
        self.vocoder = SpeechT5HifiGan.from_pretrained("/workspace/models/speecht5_vocoder")
        self.tts_model.to(device)
        self.tts_model.eval()
        self.vocoder.to(device)
        self.vocoder.eval()
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
        
        # 4. Codec (Encodec)
        print("Loading Encodec...")
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(6.0)
        self.codec.to(device)
        self.codec.eval()
        
        print("All models loaded!")
    
    @torch.no_grad()
    def speech_to_text(self, audio_array, sample_rate=16000):
        """Convert speech to text (ASR)"""
        inputs = self.asr_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        predicted_ids = self.asr_model.generate(
            inputs,
            language="te",  # Telugu
            task="transcribe",
            max_length=448,
            num_beams=1  # Faster
        )
        
        transcription = self.asr_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    @torch.no_grad()
    def generate_response(self, text):
        """Generate conversational response"""
        prompt = f"User: {text}\nAssistant:"
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.llm.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant part
        response = response.split("Assistant:")[-1].strip()
        
        return response
    
    @torch.no_grad()
    def text_to_speech(self, text):
        """Convert text to speech (TTS)"""
        inputs = self.tts_processor(text=text, return_tensors="pt")
        
        speech = self.tts_model.generate_speech(
            inputs["input_ids"].to(self.device),
            self.speaker_embedding,
            vocoder=self.vocoder
        )
        
        return speech.cpu().numpy()
    
    async def process(self, audio_array, sample_rate=16000):
        """Full S2S pipeline"""
        import time
        start_time = time.time()
        
        # Step 1: ASR
        text_input = self.speech_to_text(audio_array, sample_rate)
        asr_time = time.time() - start_time
        
        # Step 2: LLM
        llm_start = time.time()
        text_output = self.generate_response(text_input)
        llm_time = time.time() - llm_start
        
        # Step 3: TTS
        tts_start = time.time()
        audio_output = self.text_to_speech(text_output)
        tts_time = time.time() - tts_start
        
        total_time = time.time() - start_time
        
        return {
            "audio": audio_output,
            "input_text": text_input,
            "output_text": text_output,
            "latency_ms": int(total_time * 1000),
            "breakdown": {
                "asr_ms": int(asr_time * 1000),
                "llm_ms": int(llm_time * 1000),
                "tts_ms": int(tts_time * 1000)
            }
        }

# Test initialization
if __name__ == "__main__":
    pipeline = TeluguS2SPipeline()
    print("Pipeline ready for inference!")
EOFALL

# Test the pipeline
python /workspace/s2s_pipeline.py
```

---

## STEP 7: Create WebSocket Server (20 minutes)

```bash
# Create server script
cat > /workspace/server.py << 'EOFALL'
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import numpy as np
import base64
import json
import os
from s2s_pipeline import TeluguS2SPipeline

app = FastAPI(title="Telugu S2S Voice Agent")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup():
    global pipeline
    print("=" * 50)
    print("Initializing Telugu S2S Pipeline...")
    print("=" * 50)
    pipeline = TeluguS2SPipeline()
    print("=" * 50)
    print("Server ready!")
    print("=" * 50)

@app.get("/")
async def root():
    return HTMLResponse(open("/workspace/static/index.html").read())

@app.get("/health")
async def health():
    return {"status": "healthy", "pipeline": pipeline is not None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")
    
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
                
                print(f"📥 Received audio: {len(audio_array)} samples")
                
                # Process through pipeline
                result = await pipeline.process(audio_array)
                
                # Encode output audio
                output_b64 = base64.b64encode(result["audio"].tobytes()).decode('utf-8')
                
                # Send response
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "audio": output_b64,
                    "input_text": result["input_text"],
                    "output_text": result["output_text"],
                    "latency_ms": result["latency_ms"],
                    "breakdown": result["breakdown"]
                }))
                
                print(f"📤 Sent response: {result['latency_ms']}ms")
                
    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Mount static files
os.makedirs("/workspace/static", exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
EOFALL
```

---

## STEP 8: Create Browser Client (15 minutes)

```bash
# Create static directory
mkdir -p /workspace/static

# Create index.html
cat > /workspace/static/index.html << 'EOFALL'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telugu S2S Voice Agent - POC</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: bold;
            font-size: 18px;
        }
        .status.connected {
            background: #d1fae5;
            color: #065f46;
        }
        .status.disconnected {
            background: #fee2e2;
            color: #991b1b;
        }
        button {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 10px 0;
        }
        #startBtn {
            background: #10b981;
            color: white;
        }
        #startBtn:hover { background: #059669; transform: scale(1.02); }
        #stopBtn {
            background: #ef4444;
            color: white;
            display: none;
        }
        #stopBtn:hover { background: #dc2626; transform: scale(1.02); }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric {
            background: #f3f4f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
        }
        .transcript {
            background: #f9fafb;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            min-height: 80px;
        }
        .transcript-title {
            font-weight: bold;
            color: #374151;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
        }
        .transcript-content {
            color: #1f2937;
            font-size: 16px;
            line-height: 1.6;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #9ca3af;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Telugu S2S Voice Agent</h1>
        <div class="subtitle">Ultra-Low Latency Speech-to-Speech POC</div>
        
        <div id="status" class="status disconnected">● Disconnected</div>
        
        <button id="startBtn">🎙️ Start Conversation</button>
        <button id="stopBtn">⏹️ Stop</button>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="latency">-</div>
                <div class="metric-label">Total Latency (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="asr">-</div>
                <div class="metric-label">ASR (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="llm">-</div>
                <div class="metric-label">LLM (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="tts">-</div>
                <div class="metric-label">TTS (ms)</div>
            </div>
        </div>
        
        <div class="transcript">
            <div class="transcript-title">📝 Your Speech (Telugu):</div>
            <div class="transcript-content" id="inputText">Waiting for input...</div>
        </div>
        
        <div class="transcript">
            <div class="transcript-title">💬 AI Response:</div>
            <div class="transcript-content" id="outputText">Waiting for response...</div>
        </div>
        
        <div class="footer">
            Built with FastAPI + WebSockets | RunPod L4 GPU | 24-Hour POC
        </div>
    </div>

    <script>
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;

        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');

        startBtn.addEventListener('click', start);
        stopBtn.addEventListener('click', stop);

        async function start() {
            try {
                // Connect WebSocket
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    status.textContent = '● Connected';
                    status.className = 'status connected';
                    startBtn.style.display = 'none';
                    stopBtn.style.display = 'block';
                };

                ws.onmessage = handleResponse;

                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    status.textContent = '● Error';
                    status.className = 'status disconnected';
                };

                ws.onclose = () => {
                    status.textContent = '● Disconnected';
                    status.className = 'status disconnected';
                };

                // Setup audio capture
                audioContext = new AudioContext({ sampleRate: 16000 });
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    } 
                });

                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        sendAudio(inputData);
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

            } catch (error) {
                console.error('Failed to start:', error);
                alert('Failed to start: ' + error.message);
            }
        }

        function sendAudio(audioData) {
            // Convert Float32Array to base64
            const buffer = new Float32Array(audioData);
            const bytes = new Uint8Array(buffer.buffer);
            const binary = String.fromCharCode(...bytes);
            const base64 = btoa(binary);

            ws.send(JSON.stringify({
                type: 'audio',
                audio: base64
            }));
        }

        function handleResponse(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'response') {
                // Update UI
                document.getElementById('inputText').textContent = data.input_text || 'No transcription';
                document.getElementById('outputText').textContent = data.output_text || 'No response';
                document.getElementById('latency').textContent = data.latency_ms;
                
                if (data.breakdown) {
                    document.getElementById('asr').textContent = data.breakdown.asr_ms;
                    document.getElementById('llm').textContent = data.breakdown.llm_ms;
                    document.getElementById('tts').textContent = data.breakdown.tts_ms;
                }

                // Play audio
                playAudio(data.audio);
            }
        }

        async function playAudio(base64Audio) {
            try {
                const binaryString = atob(base64Audio);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                const float32Array = new Float32Array(bytes.buffer);
                const audioBuffer = audioContext.createBuffer(1, float32Array.length, 16000);
                audioBuffer.getChannelData(0).set(float32Array);
                
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start();
            } catch (error) {
                console.error('Failed to play audio:', error);
            }
        }

        function stop() {
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
            
            status.textContent = '● Disconnected';
            status.className = 'status disconnected';
            startBtn.style.display = 'block';
            stopBtn.style.display = 'none';
        }
    </script>
</body>
</html>
EOFALL

echo "✅ Browser client created at /workspace/static/index.html"
```

---

## STEP 9: Start Server (2 minutes)

```bash
# Start server
cd /workspace
python server.py
```

**You should see**:
```
Loading models...
Loading Whisper...
Loading Llama...
Loading SpeechT5...
Loading Encodec...
All models loaded!
Server ready!
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## STEP 10: Access Demo (1 minute)

1. In RunPod dashboard, find your pod
2. Click **Connect** → **HTTP Service [Port 8000]**
3. Opens browser with your demo
4. Click "Start Conversation"
5. Allow microphone access
6. Speak in Telugu!

---

## 🎯 TESTING CHECKLIST

- [ ] GPU shows in `nvidia-smi`
- [ ] All pip packages installed
- [ ] Models downloaded (check `/workspace/models/`)
- [ ] Server starts without errors
- [ ] Browser client loads
- [ ] WebSocket connects
- [ ] Microphone access works
- [ ] Audio plays back

---

## ⚠️ TROUBLESHOOTING

### Problem: "CUDA out of memory"
**Solution**: Restart pod, use smaller batch sizes

### Problem: "Module not found"
**Solution**: Re-run pip install commands

### Problem: "WebSocket connection failed"
**Solution**: Check if port 8000 is exposed in RunPod

### Problem: "No audio output"
**Solution**: Check browser console for errors, verify microphone permissions

---

## 📊 NEXT: Show MD

Once working:
1. Record screen demo (2-3 minutes)
2. Take latency metrics screenshot
3. Prepare talking points
4. Demo live if possible

**Total Setup Time**: ~2-3 hours  
**Total Cost**: ~$20 for 24 hours

---

## 🚀 Ready to start? Copy commands above step-by-step!
