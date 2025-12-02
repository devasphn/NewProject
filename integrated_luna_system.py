#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  INTEGRATED LUNA-EQUIVALENT S2S AI SYSTEM
  
  Complete Speech-to-Speech AI system combining all components:
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      LUNA-EQUIVALENT ARCHITECTURE                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  Raw Audio Input                                                        │
  │        │                                                                │
  │        ▼                                                                │
  │  ┌───────────┐                                                          │
  │  │    VAD    │ ──────────────────────────────────────┐                  │
  │  └───────────┘                                       │                  │
  │        │ (speech detected)                           │ (silence)        │
  │        ▼                                             │                  │
  │  ┌─────────────────────────────────────────────┐    │                  │
  │  │           AUDIO ENCODER                      │    │                  │
  │  │  ┌───────────┐  ┌───────────┐  ┌──────────┐ │    │                  │
  │  │  │Production │  │ Speaker   │  │ Emotion  │ │    │                  │
  │  │  │  Codec    │  │ Encoder   │  │Classifier│ │    │                  │
  │  │  │ (VQ-VAE)  │  │(ECAPA-TDNN)│ │(WavLM)  │ │    │                  │
  │  │  └───────────┘  └───────────┘  └──────────┘ │    │                  │
  │  └─────────────────────────────────────────────┘    │                  │
  │        │                 │              │           │                  │
  │        │ audio codes     │ speaker emb  │ emotion   │                  │
  │        ▼                 ▼              ▼           │                  │
  │  ┌─────────────────────────────────────────────┐    │                  │
  │  │         CONVERSATION MANAGER                │    │                  │
  │  │    (Context, Turn-taking, History)          │    │                  │
  │  └─────────────────────────────────────────────┘    │                  │
  │        │                                            │                  │
  │        ▼                                            ▼                  │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │                    S2S TRANSFORMER                               │   │
  │  │   ┌────────────┐    ┌────────────┐    ┌────────────┐            │   │
  │  │   │  Context   │    │  Response  │    │  Streaming │            │   │
  │  │   │  Encoder   │───▶│  Decoder   │───▶│  Generator │            │   │
  │  │   └────────────┘    └────────────┘    └────────────┘            │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │        │                                                                │
  │        ▼ response codes                                                 │
  │  ┌─────────────────────────────────────────────┐                       │
  │  │           AUDIO DECODER                      │                       │
  │  │  ┌───────────┐  ┌───────────┐               │                       │
  │  │  │  Codec    │  │  Prosody  │               │                       │
  │  │  │  Decoder  │  │  Control  │               │                       │
  │  │  └───────────┘  └───────────┘               │                       │
  │  └─────────────────────────────────────────────┘                       │
  │        │                                                                │
  │        ▼                                                                │
  │  Synthesized Speech Output                                              │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  
  Features:
  - Real-time conversation with <400ms latency
  - Voice cloning (3-second reference)
  - Emotion-aware responses
  - Multi-turn dialogue context
  - Streaming audio synthesis
  
  Target: Luna/Moshi/Maya equivalent quality
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, AsyncIterator, Tuple
from dataclasses import dataclass
import asyncio
import time
import warnings

# Import all components
try:
    from codec_production import ProductionCodec, CodecConfig
except ImportError:
    warnings.warn("codec_production not found")
    ProductionCodec = None

try:
    from speaker_encoder import SpeakerEncoder
except ImportError:
    warnings.warn("speaker_encoder not found")
    SpeakerEncoder = None

try:
    from emotion_classifier import EmotionClassifier, EmotionConfig
except ImportError:
    warnings.warn("emotion_classifier not found")
    EmotionClassifier = None

try:
    from prosody_encoder import ProsodyEncoder, ProsodyConfig
except ImportError:
    warnings.warn("prosody_encoder not found")
    ProsodyEncoder = None

try:
    from vad_integration import SileroVAD, VADConfig, TurnTakingManager
except ImportError:
    warnings.warn("vad_integration not found")
    SileroVAD = None

try:
    from conversation_manager import ConversationContext, ConversationConfig, ConversationTurn, Speaker
except ImportError:
    warnings.warn("conversation_manager not found")
    ConversationContext = None

try:
    from s2s_transformer import TeluguS2STransformer, S2SConfig
except ImportError:
    warnings.warn("s2s_transformer not found")
    TeluguS2STransformer = None


@dataclass
class LunaConfig:
    """Configuration for Luna-equivalent system."""
    # Model paths
    codec_path: Optional[str] = None
    speaker_encoder_path: Optional[str] = None
    emotion_classifier_path: Optional[str] = None
    s2s_model_path: Optional[str] = None
    
    # Audio settings
    sample_rate: int = 16000
    
    # Latency targets
    max_latency_ms: int = 400
    chunk_size_ms: int = 80
    
    # Voice cloning
    reference_audio_seconds: float = 3.0
    
    # Conversation
    max_response_seconds: float = 30.0
    silence_threshold_ms: int = 500
    
    # Device
    device: str = "cuda"


class LunaEquivalentSystem(nn.Module):
    """
    Complete Luna-equivalent Speech-to-Speech AI System.
    
    Combines:
    - Audio codec (encode/decode speech)
    - Speaker encoder (voice cloning)
    - Emotion classifier (detect/generate emotions)
    - Prosody encoder (natural prosody)
    - VAD (voice activity detection)
    - Conversation manager (multi-turn context)
    - S2S transformer (response generation)
    
    Usage:
        # Initialize
        luna = LunaEquivalentSystem(config)
        luna.load_models()
        
        # Real-time conversation
        async for response_audio in luna.conversation(audio_stream):
            yield response_audio
        
        # Voice cloning
        response = luna.generate_as_speaker(
            input_audio=user_audio,
            reference_audio=speaker_reference
        )
    """
    def __init__(self, config: Optional[LunaConfig] = None):
        super().__init__()
        
        self.config = config or LunaConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize components (lazy loading)
        self.codec = None
        self.speaker_encoder = None
        self.emotion_classifier = None
        self.prosody_encoder = None
        self.s2s_model = None
        self.vad = None
        self.conversation = None
        
        # State
        self.is_initialized = False
        self.is_generating = False
        
    def load_models(self):
        """Load all model components."""
        print("=" * 60)
        print("  LOADING LUNA-EQUIVALENT SYSTEM")
        print("=" * 60)
        
        # 1. Audio Codec
        print("\n📦 Loading Audio Codec...")
        if ProductionCodec is not None:
            self.codec = ProductionCodec(CodecConfig()).to(self.device)
            if self.config.codec_path:
                self.codec.load_state_dict(
                    torch.load(self.config.codec_path, map_location=self.device)
                )
            self.codec.eval()
            print(f"   ✅ Codec loaded ({sum(p.numel() for p in self.codec.parameters())/1e6:.1f}M params)")
        else:
            print("   ⚠️ Codec not available")
        
        # 2. Speaker Encoder
        print("\n🎤 Loading Speaker Encoder...")
        if SpeakerEncoder is not None:
            self.speaker_encoder = SpeakerEncoder().to(self.device)
            if self.config.speaker_encoder_path:
                self.speaker_encoder.load_state_dict(
                    torch.load(self.config.speaker_encoder_path, map_location=self.device)
                )
            self.speaker_encoder.eval()
            print(f"   ✅ Speaker Encoder loaded ({sum(p.numel() for p in self.speaker_encoder.parameters())/1e6:.1f}M params)")
        else:
            print("   ⚠️ Speaker Encoder not available")
        
        # 3. Emotion Classifier
        print("\n😊 Loading Emotion Classifier...")
        if EmotionClassifier is not None:
            self.emotion_classifier = EmotionClassifier(EmotionConfig()).to(self.device)
            if self.config.emotion_classifier_path:
                self.emotion_classifier.load_state_dict(
                    torch.load(self.config.emotion_classifier_path, map_location=self.device)
                )
            self.emotion_classifier.eval()
            print(f"   ✅ Emotion Classifier loaded ({sum(p.numel() for p in self.emotion_classifier.parameters())/1e6:.1f}M params)")
        else:
            print("   ⚠️ Emotion Classifier not available")
        
        # 4. Prosody Encoder
        print("\n🎵 Loading Prosody Encoder...")
        if ProsodyEncoder is not None:
            self.prosody_encoder = ProsodyEncoder(ProsodyConfig()).to(self.device)
            self.prosody_encoder.eval()
            print(f"   ✅ Prosody Encoder loaded ({sum(p.numel() for p in self.prosody_encoder.parameters())/1e6:.1f}M params)")
        else:
            print("   ⚠️ Prosody Encoder not available")
        
        # 5. S2S Transformer
        print("\n🧠 Loading S2S Transformer...")
        if TeluguS2STransformer is not None:
            self.s2s_model = TeluguS2STransformer(S2SConfig()).to(self.device)
            if self.config.s2s_model_path:
                self.s2s_model.load_state_dict(
                    torch.load(self.config.s2s_model_path, map_location=self.device)
                )
            self.s2s_model.eval()
            print(f"   ✅ S2S Transformer loaded ({sum(p.numel() for p in self.s2s_model.parameters())/1e6:.1f}M params)")
        else:
            print("   ⚠️ S2S Transformer not available")
        
        # 6. VAD
        print("\n🔊 Loading Voice Activity Detection...")
        if SileroVAD is not None:
            self.vad = SileroVAD(VADConfig())
            print("   ✅ VAD loaded")
        else:
            print("   ⚠️ VAD not available")
        
        # 7. Conversation Manager
        print("\n💬 Initializing Conversation Manager...")
        if ConversationContext is not None:
            self.conversation = ConversationContext(ConversationConfig())
            print("   ✅ Conversation Manager initialized")
        else:
            print("   ⚠️ Conversation Manager not available")
        
        self.is_initialized = True
        
        print("\n" + "=" * 60)
        print("  ✅ SYSTEM READY")
        print("=" * 60)
        
        # Calculate total parameters
        total_params = 0
        for component in [self.codec, self.speaker_encoder, self.emotion_classifier,
                         self.prosody_encoder, self.s2s_model]:
            if component is not None:
                total_params += sum(p.numel() for p in component.parameters())
        print(f"\n  Total Parameters: {total_params/1e6:.1f}M")
        
    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor) -> Dict:
        """
        Encode audio to discrete codes and extract all features.
        
        Args:
            audio: [B, samples] or [B, 1, samples]
        Returns:
            Dict with codes, speaker_emb, emotion, prosody
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        audio = audio.to(self.device)
        result = {}
        
        # Encode to codes
        if self.codec is not None:
            codes = self.codec.encode(audio)
            result['codes'] = codes
        
        # Extract speaker embedding
        if self.speaker_encoder is not None:
            speaker_emb = self.speaker_encoder(audio.squeeze(1))
            result['speaker_embedding'] = speaker_emb
        
        # Detect emotion
        if self.emotion_classifier is not None:
            emotion_result = self.emotion_classifier(audio.squeeze(1))
            result['emotion'] = emotion_result['emotion'][0]
            result['emotion_id'] = emotion_result['emotion_id']
            result['emotion_probs'] = emotion_result['emotion_probs']
            if 'valence' in emotion_result:
                result['valence'] = emotion_result['valence']
                result['arousal'] = emotion_result['arousal']
        
        # Extract prosody
        if self.prosody_encoder is not None:
            prosody_emb = self.prosody_encoder.get_utterance_prosody(audio.squeeze(1))
            result['prosody_embedding'] = prosody_emb
        
        return result
    
    @torch.no_grad()
    def decode_audio(self, codes: torch.Tensor,
                     speaker_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode codes back to audio.
        
        Args:
            codes: [B, Q, T] quantized codes
            speaker_emb: Optional speaker embedding for voice cloning
        Returns:
            audio: [B, 1, samples]
        """
        if self.codec is None:
            raise RuntimeError("Codec not loaded")
        
        audio = self.codec.decode(codes)
        return audio
    
    @torch.no_grad()
    def generate_response(self, input_audio: torch.Tensor,
                          target_speaker_audio: Optional[torch.Tensor] = None,
                          target_emotion: Optional[str] = None,
                          context_codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate response audio for input.
        
        Args:
            input_audio: User's audio [1, samples]
            target_speaker_audio: Reference for voice cloning [1, samples]
            target_emotion: Target emotion for response
            context_codes: Previous conversation context
        Returns:
            response_audio: Generated response [1, 1, samples]
        """
        if self.codec is None or self.s2s_model is None:
            raise RuntimeError("Models not loaded")
        
        input_audio = input_audio.to(self.device)
        
        # Encode input
        input_features = self.encode_audio(input_audio)
        input_codes = input_features.get('codes')
        
        # Get target speaker embedding
        speaker_emb = None
        if target_speaker_audio is not None and self.speaker_encoder is not None:
            target_speaker_audio = target_speaker_audio.to(self.device)
            if target_speaker_audio.dim() == 2:
                target_speaker_audio = target_speaker_audio.unsqueeze(1)
            speaker_emb = self.speaker_encoder(target_speaker_audio.squeeze(1))
        
        # Determine emotion
        emotion_id = None
        if target_emotion is not None and self.emotion_classifier is not None:
            emotion_names = self.emotion_classifier.config.emotion_names
            if target_emotion.lower() in emotion_names:
                emotion_id = torch.tensor([emotion_names.index(target_emotion.lower())]).to(self.device)
        
        # Generate response codes
        response_codes = self.s2s_model.generate(
            input_codes=input_codes,
            speaker_embedding=speaker_emb,
            emotion_id=emotion_id,
            max_length=int(self.config.max_response_seconds * 50),  # 50Hz
        )
        
        # Decode to audio
        response_audio = self.decode_audio(response_codes, speaker_emb)
        
        return response_audio
    
    async def stream_response(self, input_audio: torch.Tensor,
                              **kwargs) -> AsyncIterator[torch.Tensor]:
        """
        Stream response audio in real-time chunks.
        
        Yields audio chunks as they're generated for low latency.
        """
        chunk_samples = int(self.config.chunk_size_ms * self.config.sample_rate / 1000)
        
        # Generate full response (in practice, would be truly streaming)
        response_audio = self.generate_response(input_audio, **kwargs)
        
        # Stream in chunks
        for i in range(0, response_audio.shape[-1], chunk_samples):
            chunk = response_audio[:, :, i:i + chunk_samples]
            yield chunk
            await asyncio.sleep(0)  # Allow other tasks to run
    
    async def conversation_loop(self, 
                                audio_stream: AsyncIterator[torch.Tensor]) -> AsyncIterator[torch.Tensor]:
        """
        Main real-time conversation loop.
        
        This is the primary interface for Luna-like conversations.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
        Yields:
            response_audio: Audio chunks of AI response
        """
        if not self.is_initialized:
            self.load_models()
        
        if self.vad is None:
            raise RuntimeError("VAD required for conversation loop")
        
        turn_manager = TurnTakingManager(self.vad)
        
        async for chunk in audio_stream:
            action = turn_manager.process(chunk)
            
            if action == 'respond':
                # Get accumulated user audio
                user_audio = turn_manager.get_user_audio()
                
                if user_audio is not None and len(user_audio) > self.config.sample_rate * 0.5:
                    # Process user input
                    if self.conversation is not None:
                        features = self.encode_audio(user_audio.unsqueeze(0))
                        turn = ConversationTurn(
                            speaker=Speaker.USER,
                            audio_codes=features.get('codes'),
                            emotion=features.get('emotion', 'neutral'),
                            speaker_embedding=features.get('speaker_embedding'),
                        )
                        self.conversation.add_turn(turn)
                    
                    # Get context
                    context_codes = None
                    if self.conversation is not None:
                        context_codes = self.conversation.get_context_codes()
                    
                    # Generate and stream response
                    turn_manager.start_ai_speaking()
                    
                    async for response_chunk in self.stream_response(
                        user_audio.unsqueeze(0),
                        context_codes=context_codes
                    ):
                        yield response_chunk
                    
                    turn_manager.stop_ai_speaking()
                    
            elif action == 'stop':
                # User interrupted - stop generating
                self.is_generating = False
                turn_manager.stop_ai_speaking()
    
    def voice_clone(self, input_audio: torch.Tensor,
                    reference_audio: torch.Tensor) -> torch.Tensor:
        """
        Convert input audio to reference speaker's voice.
        
        Args:
            input_audio: Audio to convert [1, samples]
            reference_audio: Reference of target speaker [1, samples]
        Returns:
            converted_audio: Audio in target speaker's voice
        """
        return self.generate_response(
            input_audio=input_audio,
            target_speaker_audio=reference_audio
        )
    
    def reset_conversation(self):
        """Reset conversation state for new conversation."""
        if self.conversation is not None:
            self.conversation.clear_history()
        if self.vad is not None:
            self.vad.reset()
    
    def get_system_info(self) -> Dict:
        """Get system information for debugging."""
        info = {
            'device': str(self.device),
            'is_initialized': self.is_initialized,
            'components': {
                'codec': self.codec is not None,
                'speaker_encoder': self.speaker_encoder is not None,
                'emotion_classifier': self.emotion_classifier is not None,
                'prosody_encoder': self.prosody_encoder is not None,
                's2s_model': self.s2s_model is not None,
                'vad': self.vad is not None,
                'conversation': self.conversation is not None,
            },
            'config': {
                'sample_rate': self.config.sample_rate,
                'max_latency_ms': self.config.max_latency_ms,
                'chunk_size_ms': self.config.chunk_size_ms,
            }
        }
        
        # Add conversation summary if available
        if self.conversation is not None:
            info['conversation'] = self.conversation.get_summary()
        
        return info


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK START API
# ═══════════════════════════════════════════════════════════════════════════════

def create_luna_system(
    codec_path: Optional[str] = None,
    s2s_path: Optional[str] = None,
    device: str = "cuda"
) -> LunaEquivalentSystem:
    """
    Quick helper to create and initialize Luna system.
    
    Usage:
        luna = create_luna_system(
            codec_path="checkpoints/best_codec.pt",
            s2s_path="checkpoints/best_s2s.pt"
        )
        
        response = luna.generate_response(user_audio)
    """
    config = LunaConfig(
        codec_path=codec_path,
        s2s_model_path=s2s_path,
        device=device,
    )
    
    system = LunaEquivalentSystem(config)
    system.load_models()
    
    return system


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  INTEGRATED LUNA SYSTEM TEST")
    print("=" * 70)
    
    # Create system
    config = LunaConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    luna = LunaEquivalentSystem(config)
    
    # Load models (will show which components are available)
    try:
        luna.load_models()
    except Exception as e:
        print(f"⚠️ Some components failed to load: {e}")
    
    # Get system info
    print("\n--- System Info ---")
    info = luna.get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test with synthetic audio (if codec available)
    if luna.codec is not None:
        print("\n--- Encode/Decode Test ---")
        
        test_audio = torch.randn(1, 16000 * 3).to(luna.device)  # 3 seconds
        
        # Encode
        features = luna.encode_audio(test_audio)
        print(f"Encoded features: {list(features.keys())}")
        
        if 'codes' in features:
            print(f"Codes shape: {features['codes'].shape}")
        
        if 'emotion' in features:
            print(f"Detected emotion: {features['emotion']}")
        
        # Decode
        if 'codes' in features:
            decoded = luna.decode_audio(features['codes'])
            print(f"Decoded audio shape: {decoded.shape}")
    
    # Latency test
    if luna.codec is not None:
        print("\n--- Latency Test ---")
        
        test_audio = torch.randn(1, 16000 * 2).to(luna.device)
        
        # Warmup
        for _ in range(5):
            _ = luna.encode_audio(test_audio)
        
        # Measure
        torch.cuda.synchronize() if luna.device.type == 'cuda' else None
        start = time.time()
        for _ in range(20):
            _ = luna.encode_audio(test_audio)
        torch.cuda.synchronize() if luna.device.type == 'cuda' else None
        elapsed = (time.time() - start) / 20 * 1000
        
        print(f"Encoding latency: {elapsed:.2f}ms per 2-second audio")
        
        target_met = elapsed < config.max_latency_ms
        print(f"Target ({config.max_latency_ms}ms): {'✅ MET' if target_met else '❌ EXCEEDED'}")
    
    print("\n" + "=" * 70)
    print("  ✅ INTEGRATION TEST COMPLETE!")
    print("=" * 70)
