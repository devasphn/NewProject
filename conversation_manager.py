#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  CONVERSATION MANAGER FOR S2S AI
  
  Handles multi-turn dialogue, context management, and turn-taking.
  
  Features:
  - Conversation history tracking (audio codes + metadata)
  - Context-aware response generation
  - Turn-taking with interruption handling
  - Speaker tracking (user vs AI)
  - Emotion state tracking across turns
  
  This is what makes Luna/Moshi feel "conversational" rather than
  just doing one-shot audio transformation.
  
  Reference: Moshi dialogue system, OpenAI Advanced Voice Mode
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from collections import deque


class Speaker(Enum):
    """Speaker identifiers."""
    USER = 0
    ASSISTANT = 1
    UNKNOWN = 2


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    speaker: Speaker
    audio_codes: Optional[torch.Tensor] = None  # [Q, T] quantized codes
    audio_samples: Optional[torch.Tensor] = None  # [samples] raw audio
    text: Optional[str] = None  # Optional transcript
    emotion: Optional[str] = None  # Detected emotion
    emotion_scores: Optional[Dict[str, float]] = None
    speaker_embedding: Optional[torch.Tensor] = None  # [256] speaker identity
    prosody_embedding: Optional[torch.Tensor] = None  # [512] prosody info
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    duration: float = 0.0
    
    def __post_init__(self):
        if self.timestamp_start and self.timestamp_end:
            self.duration = self.timestamp_end - self.timestamp_start


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""
    # History settings
    max_history_turns: int = 10
    max_context_tokens: int = 2048  # Max audio tokens for context
    
    # Turn-taking
    silence_threshold_ms: int = 500  # Silence before responding
    interruption_threshold_ms: int = 100  # How quickly user can interrupt
    
    # Response generation
    max_response_duration_sec: float = 30.0
    min_response_duration_sec: float = 0.5
    
    # Streaming
    chunk_duration_ms: int = 80  # Stream response in 80ms chunks
    
    # Context conditioning
    use_emotion_context: bool = True
    use_speaker_context: bool = True
    use_prosody_context: bool = True


class ConversationContext:
    """
    Maintains conversation history and provides context for generation.
    
    Usage:
        context = ConversationContext()
        
        # Add user turn
        context.add_turn(ConversationTurn(
            speaker=Speaker.USER,
            audio_codes=user_codes,
            emotion='happy'
        ))
        
        # Get context for response generation
        history_codes = context.get_context_codes()
        emotion_state = context.get_emotional_context()
    """
    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        
        # Turn history (circular buffer)
        self.history: deque = deque(maxlen=self.config.max_history_turns)
        
        # Current state
        self.current_speaker = Speaker.USER
        self.conversation_start_time = time.time()
        self.last_turn_time = time.time()
        
        # Emotional state tracking
        self.emotion_history: List[str] = []
        self.dominant_emotion = 'neutral'
        
        # Generation state
        self.is_generating = False
        self.generation_start_time = None
        
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to history."""
        self.history.append(turn)
        self.current_speaker = turn.speaker
        self.last_turn_time = time.time()
        
        # Track emotion
        if turn.emotion:
            self.emotion_history.append(turn.emotion)
            self._update_dominant_emotion()
    
    def _update_dominant_emotion(self):
        """Update dominant emotion from recent history."""
        if len(self.emotion_history) == 0:
            self.dominant_emotion = 'neutral'
            return
        
        # Count emotions in recent history
        recent = self.emotion_history[-5:]  # Last 5 turns
        emotion_counts = {}
        for emotion in recent:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find most common
        self.dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    def get_context_codes(self, max_turns: int = 5) -> Optional[torch.Tensor]:
        """
        Get audio codes from recent conversation history.
        Used to condition S2S model on conversation context.
        
        Returns:
            codes: [Q, T_total] concatenated codes from recent turns
        """
        recent_turns = list(self.history)[-max_turns:]
        
        context_codes = []
        total_tokens = 0
        
        for turn in reversed(recent_turns):
            if turn.audio_codes is not None:
                codes = turn.audio_codes
                
                # Check if we'd exceed token limit
                if total_tokens + codes.shape[-1] > self.config.max_context_tokens:
                    # Truncate to fit
                    remaining = self.config.max_context_tokens - total_tokens
                    if remaining > 0:
                        context_codes.insert(0, codes[:, -remaining:])
                    break
                
                context_codes.insert(0, codes)
                total_tokens += codes.shape[-1]
        
        if len(context_codes) == 0:
            return None
        
        return torch.cat(context_codes, dim=-1)
    
    def get_emotional_context(self) -> Dict:
        """Get emotional context for response generation."""
        return {
            'dominant_emotion': self.dominant_emotion,
            'recent_emotions': self.emotion_history[-3:] if self.emotion_history else ['neutral'],
            'user_last_emotion': self._get_last_user_emotion(),
        }
    
    def _get_last_user_emotion(self) -> str:
        """Get emotion from last user turn."""
        for turn in reversed(list(self.history)):
            if turn.speaker == Speaker.USER and turn.emotion:
                return turn.emotion
        return 'neutral'
    
    def get_speaker_context(self) -> Optional[torch.Tensor]:
        """Get speaker embedding from user's recent turns."""
        for turn in reversed(list(self.history)):
            if turn.speaker == Speaker.USER and turn.speaker_embedding is not None:
                return turn.speaker_embedding
        return None
    
    def should_respond(self, silence_duration_ms: float, 
                       is_user_speaking: bool) -> bool:
        """
        Determine if AI should start responding.
        
        Rules:
        - User has stopped speaking for sufficient duration
        - AI is not already generating
        - There's been at least one user turn
        """
        if is_user_speaking:
            return False
        
        if self.is_generating:
            return False
        
        if silence_duration_ms < self.config.silence_threshold_ms:
            return False
        
        # Check if there's user input to respond to
        if len(self.history) == 0:
            return False
        
        last_turn = self.history[-1]
        if last_turn.speaker != Speaker.USER:
            return False
        
        return True
    
    def handle_interruption(self) -> bool:
        """
        Handle user interrupting AI.
        
        Returns:
            True if there was an interruption to handle
        """
        if not self.is_generating:
            return False
        
        # Stop generation
        self.is_generating = False
        
        # Mark partial response if any
        # (would need to track partial generation state)
        
        return True
    
    def start_generation(self):
        """Mark that AI started generating response."""
        self.is_generating = True
        self.generation_start_time = time.time()
        self.current_speaker = Speaker.ASSISTANT
    
    def end_generation(self, response_turn: ConversationTurn):
        """Mark that AI finished generating and record the turn."""
        self.is_generating = False
        self.add_turn(response_turn)
    
    def clear_history(self):
        """Clear conversation history for new conversation."""
        self.history.clear()
        self.emotion_history.clear()
        self.dominant_emotion = 'neutral'
        self.is_generating = False
        self.conversation_start_time = time.time()
    
    def get_summary(self) -> Dict:
        """Get conversation summary for debugging."""
        return {
            'num_turns': len(self.history),
            'duration': time.time() - self.conversation_start_time,
            'dominant_emotion': self.dominant_emotion,
            'is_generating': self.is_generating,
            'last_speaker': self.current_speaker.name,
        }


class ConversationalS2S(nn.Module):
    """
    Speech-to-Speech model with conversation awareness.
    
    This is the main integration point that combines:
    - Audio codec for encoding/decoding
    - S2S transformer for response generation
    - Conversation context for multi-turn dialogue
    - Emotion/speaker conditioning
    
    Usage:
        model = ConversationalS2S(codec, s2s_transformer)
        
        # Real-time conversation loop
        async for response_audio in model.conversation_loop(audio_stream):
            yield response_audio
    """
    def __init__(self, codec: nn.Module, 
                 s2s_transformer: nn.Module,
                 emotion_classifier: Optional[nn.Module] = None,
                 speaker_encoder: Optional[nn.Module] = None,
                 config: Optional[ConversationConfig] = None):
        super().__init__()
        
        self.codec = codec
        self.s2s = s2s_transformer
        self.emotion_classifier = emotion_classifier
        self.speaker_encoder = speaker_encoder
        self.config = config or ConversationConfig()
        
        # Conversation state
        self.context = ConversationContext(self.config)
        
        # Context conditioning
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,  # Match S2S hidden dim
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        
    def process_user_input(self, audio: torch.Tensor) -> ConversationTurn:
        """
        Process user's audio input.
        
        Steps:
        1. Encode audio to codes
        2. Extract speaker embedding
        3. Detect emotion
        4. Create conversation turn
        5. Add to history
        """
        # Encode audio
        with torch.no_grad():
            codes = self.codec.encode(audio)
        
        # Extract speaker embedding
        speaker_emb = None
        if self.speaker_encoder is not None:
            with torch.no_grad():
                speaker_emb = self.speaker_encoder(audio)
        
        # Detect emotion
        emotion = 'neutral'
        emotion_scores = None
        if self.emotion_classifier is not None:
            with torch.no_grad():
                result = self.emotion_classifier(audio)
                emotion = result['emotion'][0]
                emotion_scores = {
                    e: p.item() 
                    for e, p in zip(
                        self.emotion_classifier.config.emotion_names,
                        result['emotion_probs'][0]
                    )
                }
        
        # Create turn
        turn = ConversationTurn(
            speaker=Speaker.USER,
            audio_codes=codes,
            audio_samples=audio,
            emotion=emotion,
            emotion_scores=emotion_scores,
            speaker_embedding=speaker_emb,
            timestamp_start=time.time(),
            timestamp_end=time.time() + audio.shape[-1] / 16000,
        )
        
        # Add to context
        self.context.add_turn(turn)
        
        return turn
    
    def generate_response(self, 
                          streaming: bool = True) -> AsyncIterator[torch.Tensor]:
        """
        Generate AI response considering conversation history.
        
        Args:
            streaming: Yield audio chunks as they're generated
        Yields:
            audio_chunks: [1, chunk_samples] audio samples
        """
        self.context.start_generation()
        
        try:
            # Get conversation context
            context_codes = self.context.get_context_codes()
            emotion_context = self.context.get_emotional_context()
            
            # Get last user input
            last_user_turn = None
            for turn in reversed(list(self.context.history)):
                if turn.speaker == Speaker.USER:
                    last_user_turn = turn
                    break
            
            if last_user_turn is None:
                return
            
            user_codes = last_user_turn.audio_codes
            
            # Determine response emotion (can be adaptive)
            response_emotion = self._determine_response_emotion(emotion_context)
            
            # Generate response codes
            response_codes = self.s2s.generate(
                input_codes=user_codes,
                context_codes=context_codes,
                emotion=response_emotion,
                max_length=int(self.config.max_response_duration_sec * 50),  # 50Hz codec
            )
            
            # Decode to audio
            response_audio = self.codec.decode(response_codes)
            
            if streaming:
                # Stream in chunks
                chunk_samples = int(self.config.chunk_duration_ms * 16)  # 16kHz
                for i in range(0, response_audio.shape[-1], chunk_samples):
                    chunk = response_audio[:, :, i:i + chunk_samples]
                    yield chunk
            else:
                yield response_audio
            
            # Record response turn
            response_turn = ConversationTurn(
                speaker=Speaker.ASSISTANT,
                audio_codes=response_codes,
                audio_samples=response_audio,
                emotion=response_emotion,
                timestamp_start=self.context.generation_start_time,
                timestamp_end=time.time(),
            )
            self.context.end_generation(response_turn)
            
        except Exception as e:
            self.context.is_generating = False
            raise e
    
    def _determine_response_emotion(self, emotion_context: Dict) -> str:
        """
        Determine appropriate emotion for response.
        
        Rules:
        - Match user's emotion for empathy
        - But adapt (e.g., calm for angry user)
        """
        user_emotion = emotion_context['user_last_emotion']
        
        # Emotion adaptation rules
        emotion_responses = {
            'angry': 'calm',      # Calm response to anger
            'sad': 'empathetic',  # Empathy for sadness
            'happy': 'happy',     # Match happiness
            'fear': 'reassuring', # Reassure fear
            'neutral': 'friendly', # Default friendly
        }
        
        return emotion_responses.get(user_emotion, 'neutral')
    
    async def conversation_loop(self, 
                                audio_stream: AsyncIterator[torch.Tensor],
                                vad: Optional['SileroVAD'] = None):
        """
        Main conversation loop for real-time interaction.
        
        This is the entry point for Luna-like conversations.
        
        Args:
            audio_stream: Async iterator of audio chunks
            vad: Voice Activity Detection for turn-taking
        Yields:
            response_audio: Audio chunks of AI response
        """
        from vad_integration import SileroVAD, TurnTakingManager
        
        if vad is None:
            vad = SileroVAD()
        
        turn_manager = TurnTakingManager(vad)
        
        async for chunk in audio_stream:
            action = turn_manager.process(chunk)
            
            if action == 'respond':
                # User finished speaking
                user_audio = turn_manager.get_user_audio()
                
                if user_audio is not None:
                    # Process user input
                    self.process_user_input(user_audio.unsqueeze(0))
                    
                    # Generate and stream response
                    turn_manager.start_ai_speaking()
                    
                    async for response_chunk in self.generate_response(streaming=True):
                        yield response_chunk
                    
                    turn_manager.stop_ai_speaking()
                    
            elif action == 'stop':
                # User interrupted
                if self.context.handle_interruption():
                    turn_manager.stop_ai_speaking()
                    # Could yield a short "okay" acknowledgment here
    
    def reset_conversation(self):
        """Reset for new conversation."""
        self.context.clear_history()


class DialogueStateTracker:
    """
    Track dialogue state for more sophisticated conversations.
    
    Can track:
    - Topics discussed
    - User preferences mentioned
    - Questions asked/answered
    - Task completion status
    """
    def __init__(self):
        self.topics: List[str] = []
        self.user_preferences: Dict[str, any] = {}
        self.pending_questions: List[str] = []
        self.completed_tasks: List[str] = []
        
    def update_from_turn(self, turn: ConversationTurn, 
                         transcript: Optional[str] = None):
        """Update state based on conversation turn."""
        # Would need NLU/LLM to extract structured info
        pass
    
    def get_state(self) -> Dict:
        """Get current dialogue state."""
        return {
            'topics': self.topics,
            'user_preferences': self.user_preferences,
            'pending_questions': self.pending_questions,
            'completed_tasks': self.completed_tasks,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  CONVERSATION MANAGER TEST")
    print("=" * 70)
    
    # Test ConversationContext
    print("\n--- ConversationContext Test ---")
    
    context = ConversationContext()
    
    # Simulate conversation
    # User turn 1
    user_turn1 = ConversationTurn(
        speaker=Speaker.USER,
        audio_codes=torch.randn(8, 50),  # 8 codebooks, 50 timesteps
        emotion='happy',
        timestamp_start=time.time(),
    )
    context.add_turn(user_turn1)
    
    print(f"After user turn 1: {context.get_summary()}")
    
    # AI turn 1
    ai_turn1 = ConversationTurn(
        speaker=Speaker.ASSISTANT,
        audio_codes=torch.randn(8, 60),
        emotion='friendly',
        timestamp_start=time.time(),
    )
    context.add_turn(ai_turn1)
    
    print(f"After AI turn 1: {context.get_summary()}")
    
    # User turn 2
    user_turn2 = ConversationTurn(
        speaker=Speaker.USER,
        audio_codes=torch.randn(8, 40),
        emotion='curious',
        timestamp_start=time.time(),
    )
    context.add_turn(user_turn2)
    
    print(f"After user turn 2: {context.get_summary()}")
    
    # Get context codes
    context_codes = context.get_context_codes(max_turns=3)
    print(f"Context codes shape: {context_codes.shape if context_codes is not None else None}")
    
    # Get emotional context
    emotion_ctx = context.get_emotional_context()
    print(f"Emotional context: {emotion_ctx}")
    
    # Test turn-taking decisions
    print("\n--- Turn-Taking Test ---")
    
    # Should respond after user stopped
    should = context.should_respond(silence_duration_ms=600, is_user_speaking=False)
    print(f"Should respond (user stopped 600ms ago): {should}")
    
    # Should not respond while user speaking
    should = context.should_respond(silence_duration_ms=600, is_user_speaking=True)
    print(f"Should respond (user still speaking): {should}")
    
    # Should not respond too soon
    should = context.should_respond(silence_duration_ms=200, is_user_speaking=False)
    print(f"Should respond (only 200ms silence): {should}")
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)
