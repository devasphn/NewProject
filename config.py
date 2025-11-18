"""
Configuration for Telugu S2S Voice Agent
"""

# GPU Configuration (RTX A6000)
GPU_NAME = "RTX A6000"
GPU_MEMORY = 48  # GB
GPU_COST_PER_HOUR = 0.49  # USD

# Model Paths
WORKSPACE_DIR = "/workspace/NewProject"
MODELS_DIR = f"{WORKSPACE_DIR}/models"
DATA_DIR = f"{WORKSPACE_DIR}/telugu_data"
OUTPUT_DIR = f"{WORKSPACE_DIR}/outputs"

# Model Configuration
WHISPER_MODEL = "openai/whisper-large-v3"
LLAMA_MODEL = "meta-llama/Llama-3.2-1B"  # 1B for faster inference
SPEECHT5_MODEL = "microsoft/speecht5_tts"
SPEECHT5_VOCODER = "microsoft/speecht5_hifigan"

# Training Configuration
TRAINING_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 100
LOGGING_STEPS = 10
FP16 = True

# Inference Configuration
ASR_LANGUAGE = "en"  # English (change to "te" for Telugu after training)
MAX_NEW_TOKENS = 30  # Shorter for faster response
TEMPERATURE = 0.8
TOP_P = 0.9

# Audio Configuration
SAMPLE_RATE = 16000
CODEC_BITRATE = 6.0  # kbps

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000

# Latency Targets (milliseconds)
TARGET_TOTAL_LATENCY = 400  # <400ms target
TARGET_ASR_LATENCY = 150
TARGET_LLM_LATENCY = 100
TARGET_TTS_LATENCY = 150

# Telugu Training
TELUGU_TRAINING_HOURS = 20
MIN_AUDIO_DURATION = 2.0  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds

# Speaker Configuration
SPEAKER_EMBEDDING_PATH = f"{MODELS_DIR}/speaker_embeddings"
NUM_SPEAKERS = 4  # 2 male + 2 female
