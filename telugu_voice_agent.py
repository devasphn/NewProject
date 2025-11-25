#!/usr/bin/env python3
"""
Telugu Voice Agent - Complete Pipeline
Uses: Whisper (ASR) + Gemini/Local LLM + Codec (TTS backbone)

This is a POC demonstrating the full voice agent architecture.
"""

import torch
import torchaudio
import argparse
import logging
import os
import json
import time
from pathlib import Path
from typing import Optional, Generator
import warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not installed. Run: pip install openai-whisper")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini not installed. Run: pip install google-generativeai")

from telugu_codec_fixed import TeluCodec

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TeluguVoiceAgent:
    """
    Complete Telugu Voice Agent Pipeline
    
    Architecture:
    1. ASR: Whisper (multilingual, supports Telugu)
    2. LLM: Gemini API or local model
    3. TTS: Codec-based synthesis
    """
    
    def __init__(
        self,
        codec_path: str,
        whisper_model: str = "medium",
        gemini_api_key: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Device: {self.device}")
        
        # Load components
        self._load_asr(whisper_model)
        self._load_llm(gemini_api_key)
        self._load_codec(codec_path)
        
        # Conversation history
        self.history = []
        
        logger.info("✅ Telugu Voice Agent initialized!")
    
    def _load_asr(self, model_name: str):
        """Load Whisper for speech recognition"""
        if WHISPER_AVAILABLE:
            logger.info(f"📥 Loading Whisper {model_name}...")
            self.asr = whisper.load_model(model_name, device=str(self.device))
            logger.info("✅ ASR (Whisper) loaded")
        else:
            self.asr = None
            logger.warning("⚠️ ASR not available - using mock transcription")
    
    def _load_llm(self, api_key: Optional[str]):
        """Load LLM for response generation"""
        if GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
            self.llm_type = "gemini"
            logger.info("✅ LLM (Gemini) configured")
        else:
            self.llm = None
            self.llm_type = "mock"
            logger.warning("⚠️ LLM not available - using template responses")
    
    def _load_codec(self, path: str):
        """Load Telugu codec for audio processing"""
        logger.info("📥 Loading Telugu Codec...")
        self.codec = TeluCodec().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        if 'codec_state_dict' in checkpoint:
            self.codec.load_state_dict(checkpoint['codec_state_dict'])
        else:
            self.codec.load_state_dict(checkpoint)
        self.codec.eval()
        logger.info("✅ Codec loaded")
    
    def transcribe(self, audio_path: str) -> str:
        """
        Step 1: Speech to Text (ASR)
        Converts Telugu audio to Telugu text
        """
        logger.info("\n🎤 [ASR] Transcribing audio...")
        
        if self.asr:
            result = self.asr.transcribe(
                audio_path,
                language="te",  # Telugu
                task="transcribe"
            )
            text = result["text"].strip()
        else:
            # Mock transcription for demo
            text = "నమస్కారం, మీరు ఎలా ఉన్నారు?"  # Hello, how are you?
        
        logger.info(f"📝 Transcribed: {text}")
        return text
    
    def generate_response(self, user_text: str) -> str:
        """
        Step 2: Generate AI Response (LLM)
        Takes Telugu text, generates Telugu response
        """
        logger.info("\n🧠 [LLM] Generating response...")
        
        # System prompt for Telugu assistant
        system_prompt = """You are a helpful Telugu voice assistant. 
Respond naturally in Telugu (తెలుగు). Keep responses concise and conversational.
If the user speaks in Telugu, respond in Telugu. Be friendly and helpful."""
        
        if self.llm_type == "gemini":
            # Use Gemini API
            prompt = f"{system_prompt}\n\nUser: {user_text}\nAssistant:"
            response = self.llm.generate_content(prompt)
            response_text = response.text.strip()
        else:
            # Template responses for demo
            responses = {
                "నమస్కారం": "నమస్కారం! నేను మీ తెలుగు సహాయకుడిని. మీకు ఎలా సహాయం చేయగలను?",
                "ఎలా ఉన్నారు": "నేను బాగున్నాను, ధన్యవాదాలు! మీరు ఎలా ఉన్నారు?",
                "మీ పేరు": "నా పేరు తెలుగు AI సహాయకుడు. నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.",
                "default": "అర్థమైంది. నేను మీకు సహాయం చేయడానికి సిద్ధంగా ఉన్నాను."
            }
            
            # Simple keyword matching
            response_text = responses["default"]
            for keyword, response in responses.items():
                if keyword in user_text:
                    response_text = response
                    break
        
        logger.info(f"💬 Response: {response_text}")
        
        # Save to history
        self.history.append({"user": user_text, "assistant": response_text})
        
        return response_text
    
    @torch.no_grad()
    def synthesize_speech(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        output_path: str = "response.wav"
    ) -> str:
        """
        Step 3: Text to Speech (TTS)
        Converts Telugu text to Telugu speech
        
        Note: Current implementation uses codec for audio processing.
        For full TTS, you would need a text-to-codec model.
        """
        logger.info("\n🔊 [TTS] Synthesizing speech...")
        
        if reference_audio and Path(reference_audio).exists():
            # Use reference audio and process through codec
            waveform, sr = torchaudio.load(reference_audio)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.unsqueeze(0).to(self.device)
            
            # Encode and decode through codec
            codes = self.codec.encode(waveform)
            output = self.codec.decode(codes)
            
            # Save
            output = output.squeeze(0).cpu()
            torchaudio.save(output_path, output, 16000)
            logger.info(f"💾 Saved: {output_path}")
            
            return output_path
        else:
            logger.warning("⚠️ No reference audio - TTS requires text-to-codec model")
            logger.info(f"📝 Response text: {text}")
            return ""
    
    def process_audio(
        self,
        input_audio: str,
        output_audio: str = "response.wav"
    ) -> dict:
        """
        Complete pipeline: Audio → Text → Response → Audio
        """
        logger.info("\n" + "="*60)
        logger.info("🎯 TELUGU VOICE AGENT PROCESSING")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Step 1: ASR
        asr_start = time.time()
        user_text = self.transcribe(input_audio)
        asr_time = time.time() - asr_start
        
        # Step 2: LLM
        llm_start = time.time()
        response_text = self.generate_response(user_text)
        llm_time = time.time() - llm_start
        
        # Step 3: TTS (using input as reference for now)
        tts_start = time.time()
        output_path = self.synthesize_speech(
            response_text,
            reference_audio=input_audio,
            output_path=output_audio
        )
        tts_time = time.time() - tts_start
        
        total_time = time.time() - start_time
        
        # Results
        result = {
            "input_audio": input_audio,
            "user_text": user_text,
            "response_text": response_text,
            "output_audio": output_path,
            "timings": {
                "asr_ms": asr_time * 1000,
                "llm_ms": llm_time * 1000,
                "tts_ms": tts_time * 1000,
                "total_ms": total_time * 1000
            }
        }
        
        logger.info("\n" + "="*60)
        logger.info("📊 RESULTS")
        logger.info("="*60)
        logger.info(f"🎤 User said: {user_text}")
        logger.info(f"🤖 Agent response: {response_text}")
        logger.info(f"⏱️ ASR: {asr_time*1000:.0f}ms | LLM: {llm_time*1000:.0f}ms | TTS: {tts_time*1000:.0f}ms")
        logger.info(f"⏱️ Total: {total_time*1000:.0f}ms")
        logger.info("="*60)
        
        return result
    
    def interactive_demo(self):
        """Run interactive demo with sample files"""
        logger.info("\n" + "="*60)
        logger.info("🎤 TELUGU VOICE AGENT - INTERACTIVE DEMO")
        logger.info("="*60)
        
        # Find sample audio
        sample_dirs = [
            "/workspace/telugu_data/openslr",
            "/workspace/telugu_data/indictts/audio",
        ]
        
        sample_file = None
        for d in sample_dirs:
            if Path(d).exists():
                files = list(Path(d).glob("*.wav"))[:1]
                if files:
                    sample_file = str(files[0])
                    break
        
        if not sample_file:
            logger.error("❌ No sample audio found!")
            return
        
        logger.info(f"\n📁 Using sample: {sample_file}")
        
        # Process
        result = self.process_audio(sample_file, "agent_response.wav")
        
        logger.info("\n🎉 Demo complete! Check 'agent_response.wav'")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Telugu Voice Agent")
    parser.add_argument("--codec_path", default="/workspace/models/codec/best_codec.pt")
    parser.add_argument("--whisper_model", default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--gemini_key", type=str, help="Gemini API key")
    parser.add_argument("--input", type=str, help="Input audio file")
    parser.add_argument("--output", default="response.wav", help="Output audio file")
    args = parser.parse_args()
    
    # Get API key from env if not provided
    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    
    # Initialize agent
    agent = TeluguVoiceAgent(
        codec_path=args.codec_path,
        whisper_model=args.whisper_model,
        gemini_api_key=gemini_key
    )
    
    if args.input:
        # Process specific file
        agent.process_audio(args.input, args.output)
    else:
        # Run interactive demo
        agent.interactive_demo()


if __name__ == "__main__":
    main()
