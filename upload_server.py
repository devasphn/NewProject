#!/usr/bin/env python3
"""
File Upload Server for RunPod
With progress logging and verification
"""

from flask import Flask, request, jsonify
import os
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = '/workspace'

@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>📤 RunPod File Upload</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            color: white;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            background: rgba(255,255,255,0.1);
            padding: 40px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        h1 { text-align: center; margin-bottom: 30px; }
        .upload-area {
            border: 3px dashed #00cec9;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: rgba(0, 206, 201, 0.1);
        }
        .upload-area.dragover {
            background: rgba(0, 206, 201, 0.2);
            border-color: #00b894;
        }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
        .btn:hover { transform: scale(1.02); }
        .btn:disabled { 
            opacity: 0.5; 
            cursor: not-allowed;
            transform: none;
        }
        .progress-container {
            display: none;
            margin-top: 20px;
        }
        .progress-bar {
            height: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00b894, #00cec9);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .status {
            margin-top: 15px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .status.success { background: rgba(0, 184, 148, 0.3); }
        .status.error { background: rgba(231, 76, 60, 0.3); }
        .status.info { background: rgba(52, 152, 219, 0.3); }
        .file-info {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            display: none;
        }
        .log {
            background: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .log-entry { margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📤 Upload File to RunPod</h1>
        
        <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
            <p style="font-size: 48px; margin: 0;">📁</p>
            <p>Click or drag file here</p>
            <p style="color: #888; font-size: 14px;">Supports any file type</p>
        </div>
        
        <input type="file" id="fileInput" onchange="handleFileSelect(event)">
        
        <div class="file-info" id="fileInfo"></div>
        
        <button class="btn" id="uploadBtn" onclick="uploadFile()" disabled>
            ⬆️ Upload File
        </button>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>
        </div>
        
        <div class="status" id="status" style="display:none;"></div>
        
        <div class="log" id="log">
            <div class="log-entry">📋 Ready for upload...</div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // Drag and drop
        const dropArea = document.getElementById('dropArea');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
            dropArea.addEventListener(evt, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        dropArea.addEventListener('dragover', () => dropArea.classList.add('dragover'));
        dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
        dropArea.addEventListener('drop', e => {
            dropArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                selectedFile = e.dataTransfer.files[0];
                showFileInfo();
            }
        });
        
        function log(msg) {
            const logDiv = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            logDiv.innerHTML += `<div class="log-entry">[${time}] ${msg}</div>`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function handleFileSelect(event) {
            selectedFile = event.target.files[0];
            showFileInfo();
        }
        
        function showFileInfo() {
            if (!selectedFile) return;
            const size = (selectedFile.size / (1024*1024)).toFixed(2);
            document.getElementById('fileInfo').innerHTML = 
                `📄 <strong>${selectedFile.name}</strong><br>Size: ${size} MB`;
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('uploadBtn').disabled = false;
            log(`Selected: ${selectedFile.name} (${size} MB)`);
        }
        
        function uploadFile() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const xhr = new XMLHttpRequest();
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const status = document.getElementById('status');
            const btn = document.getElementById('uploadBtn');
            
            progressContainer.style.display = 'block';
            btn.disabled = true;
            btn.textContent = '⏳ Uploading...';
            
            log('Starting upload...');
            
            xhr.upload.addEventListener('progress', e => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    progressFill.style.width = percent + '%';
                    progressFill.textContent = percent + '%';
                    
                    const loadedMB = (e.loaded / (1024*1024)).toFixed(1);
                    const totalMB = (e.total / (1024*1024)).toFixed(1);
                    log(`Progress: ${loadedMB}MB / ${totalMB}MB (${percent}%)`);
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const result = JSON.parse(xhr.responseText);
                    status.className = 'status success';
                    status.innerHTML = `✅ <strong>Upload Complete!</strong><br>
                        File: ${result.filename}<br>
                        Size: ${result.size_mb} MB<br>
                        Path: ${result.path}`;
                    status.style.display = 'block';
                    log(`✅ SUCCESS: File saved to ${result.path}`);
                    btn.textContent = '✅ Upload Complete!';
                } else {
                    status.className = 'status error';
                    status.innerHTML = '❌ Upload failed: ' + xhr.statusText;
                    status.style.display = 'block';
                    log('❌ ERROR: ' + xhr.statusText);
                    btn.disabled = false;
                    btn.textContent = '⬆️ Retry Upload';
                }
            });
            
            xhr.addEventListener('error', () => {
                status.className = 'status error';
                status.innerHTML = '❌ Upload failed - network error';
                status.style.display = 'block';
                log('❌ Network error');
                btn.disabled = false;
                btn.textContent = '⬆️ Retry Upload';
            });
            
            xhr.open('POST', '/upload');
            xhr.send(formData);
        }
    </script>
</body>
</html>
'''

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({"error": "No file uploaded"}), 400
        
        f = request.files['file']
        if f.filename == '':
            logger.error("Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        filename = f.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        logger.info(f"📥 Receiving file: {filename}")
        start_time = time.time()
        
        # Save file
        f.save(filepath)
        
        elapsed = time.time() - start_time
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        
        logger.info(f"✅ File saved: {filepath}")
        logger.info(f"📊 Size: {size_mb:.2f} MB")
        logger.info(f"⏱️ Time: {elapsed:.1f} seconds")
        logger.info(f"🚀 Speed: {size_mb/elapsed:.2f} MB/s")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "path": filepath,
            "size_bytes": size_bytes,
            "size_mb": f"{size_mb:.2f}",
            "time_seconds": f"{elapsed:.1f}"
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/files', methods=['GET'])
def list_files():
    """List files in /workspace"""
    files = []
    for f in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            files.append({
                "name": f,
                "size_mb": f"{size/(1024*1024):.2f}"
            })
    return jsonify(files)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("📤 FILE UPLOAD SERVER")
    print("="*60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("Open in browser: http://0.0.0.0:8010")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8010, threaded=True)
