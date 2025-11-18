# Pinpoint Requirements

This file captures the **exact package versions** that pip resolved during the successful installation run on Nov 18, 2025. Use it as the single source of truth whenever you need to re-create the working environment.

| Category | Package | Version |
| --- | --- | --- |
| Core DL | torch | 2.8.0+cu128 |
|  | torchaudio | 2.8.0 |
|  | torchvision | 0.23.0 |
| Attention Optimizations | flash-attn | 2.8.3 |
| Transformer Stack | transformers | 4.57.1 |
|  | tokenizers | 0.22.1 |
| Training Runtime | accelerate | 1.11.0 |
|  | datasets | 4.4.1 |
| Audio Stack | numpy | 2.0.2 |
|  | scipy | 1.15.3 |
|  | librosa | 0.11.0 |
|  | soundfile | 0.13.1 |
|  | webrtcvad | 2.0.10 |
|  | pydub | 0.25.1 |
| Codec Support | einops | 0.8.1 |
|  | vector-quantize-pytorch | 1.25.2 |
| Data Collection & Alignment | yt-dlp | 2025.11.12 |
|  | whisperx | 3.7.4 |
|  | pyannote.audio | 3.4.0 |
| Web Server | fastapi | 0.121.2 |
|  | uvicorn | 0.38.0 |
|  | websockets | 15.0.1 |
|  | python-multipart | 0.0.20 |
|  | uvloop | 0.22.1 |
| Monitoring & Logging | wandb | 0.23.0 |
|  | tensorboard | 2.20.0 |
|  | tqdm | 4.67.1 |
|  | rich | 14.2.0 |
| Utilities | python-dotenv | 1.2.1 |
|  | omegaconf | 2.3.0 |
|  | hydra-core | 1.3.2 |
| Dev Tools | ipython | 8.21.0 |
|  | black | 25.11.0 |
| Deployment | onnx | 1.19.1 |
|  | onnxruntime-gpu | 1.23.2 |
|  | runpod | 1.7.13 |
| WhisperX Runtime Extras | triton | 3.4.0 |
|  | ctranslate2 | 4.6.1 |
|  | faster-whisper | 1.2.1 |
|  | pytorch-metric-learning | 2.9.0 |
|  | pyannote.core/database/metrics/pipeline | 5.0.0 / 5.1.3 / 3.2.1 / 3.0.1 |
|  | speechbrain | 1.0.3 |
|  | pytorch-lightning | 2.5.6 |

## Usage
1. Keep `requirements_new.txt` with `>=` ranges for compatibility checks.
2. When you want deterministic installs, pin to the versions above (e.g., via a lock file or direct `pip install pkg==version`).
3. If a future package upgrade is required, update both this table and the deployment scripts after verifying the new versions together.

This document deliberately excludes transient build dependencies (e.g., `antlr4-python3-runtime` for OmegaConf) unless they are part of the runtime stack.
