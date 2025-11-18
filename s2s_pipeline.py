"""
Telugu Speech-to-Speech Pipeline
Optimized for RTX A6000
"""

import os
import torch
import torchaudio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)
from encodec import EncodecModel
import numpy as np
import time
from config import *

class TeluguS2SPipeline:
    def __init__(self, device="cuda", use_telugu_model=False):
        self.device = device
        self.use_telugu_model = use_telugu_model
        print("Initializing Telugu S2S Pipeline...")
        print(f"Device: {device}")
        print(f"GPU: {GPU_NAME} ({GPU_MEMORY}GB)")
        
        # Load all models
        self._load_models()
        
        print("✓ Pipeline ready!")
    
    def _load_models(self):
        """Load all required models"""
        
        # 1. ASR (Whisper)
        print("Loading Whisper ASR...")
        self.asr_processor = WhisperProcessor.from_pretrained(f"{MODELS_DIR}/whisper")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(f"{MODELS_DIR}/whisper")
        self.asr_model.to(self.device)
        self.asr_model.eval()
        
        # 2. LLM (Llama)
        print("Loading Llama LLM...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            f"{MODELS_DIR}/llama",
            trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            f"{MODELS_DIR}/llama",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.llm.eval()
        
        # Ensure pad_token is set
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        # 3. TTS (SpeechT5 - Telugu fine-tuned if available)
        print("Loading SpeechT5 TTS...")
        if self.use_telugu_model and os.path.exists(f"{MODELS_DIR}/speecht5_telugu"):
            print("Using Telugu fine-tuned model")
            tts_path = f"{MODELS_DIR}/speecht5_telugu"
        else:
            tts_path = f"{MODELS_DIR}/speecht5"
        
        self.tts_processor = SpeechT5Processor.from_pretrained(tts_path)
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_path)
        self.vocoder = SpeechT5HifiGan.from_pretrained(f"{MODELS_DIR}/speecht5_vocoder")
        
        self.tts_model.to(self.device)
        self.tts_model.eval()
        self.vocoder.to(self.device)
        self.vocoder.eval()
        
        # Load speaker embeddings
        self.speaker_embeddings = []
        for i in range(NUM_SPEAKERS):
            emb_path = f"{SPEAKER_EMBEDDING_PATH}/speaker_{i}.pt"
            if os.path.exists(emb_path):
                emb = torch.load(emb_path).unsqueeze(0).to(self.device)
                self.speaker_embeddings.append(emb)
        
        if not self.speaker_embeddings:
            print("⚠ No speaker embeddings found, using default")
            self.speaker_embeddings = [torch.zeros((1, 512)).to(self.device)]
        
        # 4. Codec (Encodec)
        print("Loading Encodec...")
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(CODEC_BITRATE)
        self.codec.to(self.device)
        self.codec.eval()
    
    @torch.no_grad()
    def speech_to_text(self, audio_array, sample_rate=16000):
        """Convert speech to text (ASR)"""
        start_time = time.time()
        
        inputs = self.asr_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        predicted_ids = self.asr_model.generate(
            inputs,
            language=ASR_LANGUAGE,  # Telugu
            task="transcribe",
            max_length=448,
            num_beams=1  # Faster
        )
        
        transcription = self.asr_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        latency_ms = int((time.time() - start_time) * 1000)
        return transcription, latency_ms
    
    @torch.no_grad()
    def generate_response(self, text):
        """Generate conversational response"""
        start_time = time.time()
        
        # Conversational prompt - respond naturally to user input
        prompt = f"""You are a helpful voice assistant. Respond briefly and naturally to what the user says.

User: {text}
Assistant:"""
        
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.llm.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # Clean up
        response = response.replace("User:", "").strip()
        
        latency_ms = int((time.time() - start_time) * 1000)
        return response, latency_ms
    
    @torch.no_grad()
    def text_to_speech(self, text, speaker_id=0):
        """Convert text to speech (TTS)"""
        start_time = time.time()
        
        # Use specified speaker embedding
        speaker_id = min(speaker_id, len(self.speaker_embeddings) - 1)
        speaker_embedding = self.speaker_embeddings[speaker_id]
        
        inputs = self.tts_processor(text=text, return_tensors="pt")
        
        speech = self.tts_model.generate_speech(
            inputs["input_ids"].to(self.device),
            speaker_embedding,
            vocoder=self.vocoder
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        return speech.cpu().numpy(), latency_ms
    
    async def process(self, audio_array, sample_rate=16000, speaker_id=0):
        """Full S2S pipeline"""
        total_start = time.time()
        
        # Step 1: ASR
        text_input, asr_latency = self.speech_to_text(audio_array, sample_rate)
        
        # Step 2: LLM
        text_output, llm_latency = self.generate_response(text_input)
        
        # Step 3: TTS
        audio_output, tts_latency = self.text_to_speech(text_output, speaker_id)
        
        total_latency = int((time.time() - total_start) * 1000)
        
        return {
            "audio": audio_output,
            "input_text": text_input,
            "output_text": text_output,
            "latency_ms": total_latency,
            "breakdown": {
                "asr_ms": asr_latency,
                "llm_ms": llm_latency,
                "tts_ms": tts_latency
            }
        }

# Test initialization
if __name__ == "__main__":
    print("Testing Telugu S2S Pipeline...")
    pipeline = TeluguS2SPipeline()
    print("✓ Pipeline initialized successfully!")
