"""
Download all required pre-trained models
"""

import os
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)
from encodec import EncodecModel
from datasets import load_dataset
import torch
from config import *

def download_models():
    """Download all required models"""
    
    print("=" * 60)
    print("Downloading Pre-trained Models")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Download Whisper (ASR)
    print("\n[1/5] Downloading Whisper Large V3...")
    try:
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
        processor.save_pretrained(f"{MODELS_DIR}/whisper")
        model.save_pretrained(f"{MODELS_DIR}/whisper")
        print("✓ Whisper downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading Whisper: {e}")
        return False
    
    # 2. Download Llama 3.2 1B (LLM)
    print("\n[2/5] Downloading Llama 3.2 1B...")
    print("Note: You need HuggingFace token for Llama")
    print("Set it with: export HF_TOKEN='your_token_here'")
    
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("⚠ Warning: HF_TOKEN not set. Trying without token...")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL, 
            token=hf_token,
            trust_remote_code=True
        )
        
        # Download model with proper config
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map=None,  # Don't auto-map to devices during download
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Save to disk
        tokenizer.save_pretrained(f"{MODELS_DIR}/llama")
        model.save_pretrained(f"{MODELS_DIR}/llama")
        print("✓ Llama downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading Llama: {e}")
        print("Make sure to set HF_TOKEN environment variable")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Download SpeechT5 (TTS)
    print("\n[3/5] Downloading SpeechT5...")
    try:
        processor = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL)
        model = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL)
        vocoder = SpeechT5HifiGan.from_pretrained(SPEECHT5_VOCODER)
        
        processor.save_pretrained(f"{MODELS_DIR}/speecht5")
        model.save_pretrained(f"{MODELS_DIR}/speecht5")
        vocoder.save_pretrained(f"{MODELS_DIR}/speecht5_vocoder")
        print("✓ SpeechT5 downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading SpeechT5: {e}")
        return False
    
    # 4. Download Encodec
    print("\n[4/5] Downloading Encodec...")
    try:
        codec = EncodecModel.encodec_model_24khz()
        codec.set_target_bandwidth(CODEC_BITRATE)
        print("✓ Encodec downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading Encodec: {e}")
        return False
    
    # 5. Download speaker embeddings dataset
    print("\n[5/5] Downloading speaker embeddings...")
    try:
        os.makedirs(SPEAKER_EMBEDDING_PATH, exist_ok=True)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        
        # Save multiple speaker embeddings
        speaker_ids = [7306, 8051, 9017, 9166]  # Different speakers
        for i, speaker_id in enumerate(speaker_ids):
            embedding = torch.tensor(embeddings_dataset[speaker_id]["xvector"])
            torch.save(embedding, f"{SPEAKER_EMBEDDING_PATH}/speaker_{i}.pt")
        
        print(f"✓ Downloaded {len(speaker_ids)} speaker embeddings")
    except Exception as e:
        print(f"✗ Error downloading speaker embeddings: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Total size: ~15-20 GB")
    
    return True

if __name__ == "__main__":
    success = download_models()
    if not success:
        print("\n❌ Model download failed!")
        exit(1)
    else:
        print("\n✅ Ready to start server!")
        exit(0)
