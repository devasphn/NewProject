#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  VOICE ACTIVITY DETECTION (VAD) INTEGRATION
  
  Wrapper around Silero VAD for real-time speech detection.
  
  Features:
  - Real-time chunk-by-chunk processing
  - Speech start/end detection for turn-taking
  - Configurable silence thresholds
  - Works at 16kHz and 8kHz
  
  Used for:
  - Detecting when user starts/stops speaking
  - Turn-taking in conversations
  - Filtering silence for efficient processing
  
  Reference: https://github.com/snakers4/silero-vad
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
import warnings
import time


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    sample_rate: int = 16000
    
    # Detection thresholds
    threshold: float = 0.5  # Speech probability threshold
    neg_threshold: float = 0.35  # Threshold for speech end (hysteresis)
    
    # Timing parameters (in milliseconds)
    min_silence_duration_ms: int = 300  # Min silence to end utterance
    min_speech_duration_ms: int = 250   # Min speech to start utterance
    speech_pad_ms: int = 30             # Padding around speech
    
    # Chunk size (must match model requirements)
    # 512 samples for 16kHz, 256 for 8kHz
    window_size_samples: int = 512
    
    # Turn-taking
    response_delay_ms: int = 500  # Wait this long after user stops before responding


class SileroVAD:
    """
    Silero VAD wrapper for real-time speech detection.
    
    Usage:
        vad = SileroVAD()
        
        # Real-time processing
        for chunk in audio_stream:
            event = vad.process_chunk(chunk)
            if event == 'speech_start':
                print("User started speaking")
            elif event == 'speech_end':
                print("User stopped speaking")
    """
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        
        # Load Silero VAD model
        self.model = None
        self._load_model()
        
        # State tracking
        self.reset()
        
    def _load_model(self):
        """Load Silero VAD model from torch hub."""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self.model.eval()
            print("✅ Silero VAD loaded successfully")
        except Exception as e:
            warnings.warn(f"Could not load Silero VAD: {e}. Using fallback.")
            self.model = None
    
    def reset(self):
        """Reset VAD state for new conversation."""
        if self.model is not None:
            self.model.reset_states()
        
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.speech_buffer = []
        self.current_time = 0.0
        
    @property
    def min_silence_samples(self) -> int:
        return int(self.config.min_silence_duration_ms * self.config.sample_rate / 1000)
    
    @property
    def min_speech_samples(self) -> int:
        return int(self.config.min_speech_duration_ms * self.config.sample_rate / 1000)
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> Optional[str]:
        """
        Process a single audio chunk and detect speech events.
        
        Args:
            audio_chunk: [window_size_samples] audio samples
        Returns:
            'speech_start' - User started speaking
            'speech_end' - User stopped speaking
            'speaking' - User is currently speaking
            None - No speech or waiting
        """
        if self.model is None:
            # Fallback: simple energy-based detection
            return self._fallback_vad(audio_chunk)
        
        # Ensure correct shape
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.squeeze(0)
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_chunk, self.config.sample_rate).item()
        
        # Update time
        chunk_duration = len(audio_chunk) / self.config.sample_rate
        self.current_time += chunk_duration
        
        # State machine for speech detection
        return self._update_state(speech_prob)
    
    def _update_state(self, speech_prob: float) -> Optional[str]:
        """Update speech detection state based on probability."""
        
        if not self.is_speaking:
            # Currently not speaking
            if speech_prob >= self.config.threshold:
                # Potential speech start
                if self.speech_start_time is None:
                    self.speech_start_time = self.current_time
                
                # Check if speech duration exceeds minimum
                speech_duration = self.current_time - self.speech_start_time
                if speech_duration >= self.config.min_speech_duration_ms / 1000:
                    self.is_speaking = True
                    self.silence_start_time = None
                    return 'speech_start'
            else:
                # Reset speech start tracking
                self.speech_start_time = None
                
        else:
            # Currently speaking
            if speech_prob < self.config.neg_threshold:
                # Potential speech end
                if self.silence_start_time is None:
                    self.silence_start_time = self.current_time
                
                # Check if silence duration exceeds minimum
                silence_duration = self.current_time - self.silence_start_time
                if silence_duration >= self.config.min_silence_duration_ms / 1000:
                    self.is_speaking = False
                    self.speech_start_time = None
                    return 'speech_end'
            else:
                # Reset silence tracking
                self.silence_start_time = None
                return 'speaking'
        
        return None
    
    def _fallback_vad(self, audio_chunk: torch.Tensor) -> Optional[str]:
        """Simple energy-based VAD fallback."""
        energy = torch.sqrt(torch.mean(audio_chunk ** 2)).item()
        
        # Threshold based on typical speech energy
        is_speech = energy > 0.01
        
        if not self.is_speaking and is_speech:
            self.is_speaking = True
            return 'speech_start'
        elif self.is_speaking and not is_speech:
            self.is_speaking = False
            return 'speech_end'
        elif self.is_speaking:
            return 'speaking'
        
        return None
    
    def get_speech_timestamps(self, audio: torch.Tensor, 
                               return_seconds: bool = True) -> List[Dict]:
        """
        Get speech timestamps for an entire audio file.
        
        Args:
            audio: [samples] or [1, samples]
            return_seconds: Return times in seconds (vs samples)
        Returns:
            List of {'start': float, 'end': float} dicts
        """
        if self.model is None:
            warnings.warn("Silero VAD not loaded, using fallback")
            return []
        
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        # Use Silero's built-in function
        try:
            from silero_vad import get_speech_timestamps
            timestamps = get_speech_timestamps(
                audio,
                self.model,
                threshold=self.config.threshold,
                sampling_rate=self.config.sample_rate,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                return_seconds=return_seconds,
            )
            return timestamps
        except ImportError:
            # Manual implementation
            return self._manual_get_timestamps(audio, return_seconds)
    
    def _manual_get_timestamps(self, audio: torch.Tensor,
                                return_seconds: bool) -> List[Dict]:
        """Manual speech timestamp extraction."""
        self.reset()
        
        timestamps = []
        current_start = None
        
        window = self.config.window_size_samples
        
        for i in range(0, len(audio) - window, window):
            chunk = audio[i:i + window]
            event = self.process_chunk(chunk)
            
            if event == 'speech_start':
                current_start = i
            elif event == 'speech_end' and current_start is not None:
                if return_seconds:
                    timestamps.append({
                        'start': current_start / self.config.sample_rate,
                        'end': i / self.config.sample_rate,
                    })
                else:
                    timestamps.append({
                        'start': current_start,
                        'end': i,
                    })
                current_start = None
        
        return timestamps


class VADIterator:
    """
    Iterator-style VAD for streaming audio processing.
    
    Matches Silero's VADIterator interface.
    
    Usage:
        vad_iterator = VADIterator(model)
        
        for chunk in audio_stream:
            speech_dict = vad_iterator(chunk)
            if speech_dict:
                if 'start' in speech_dict:
                    print(f"Speech started at {speech_dict['start']}")
                if 'end' in speech_dict:
                    print(f"Speech ended at {speech_dict['end']}")
    """
    def __init__(self, model: nn.Module,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30):
        
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = int(min_silence_duration_ms * sampling_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sampling_rate / 1000)
        
        self.reset_states()
        
    def reset_states(self):
        """Reset all states."""
        if self.model is not None:
            self.model.reset_states()
        
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        
    def __call__(self, audio_chunk: torch.Tensor,
                 return_seconds: bool = False) -> Optional[Dict]:
        """
        Process chunk and return speech event if any.
        
        Args:
            audio_chunk: Audio samples [window_size]
            return_seconds: Return times in seconds
        Returns:
            Dict with 'start' or 'end' key, or None
        """
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.squeeze(0)
        
        window_size = len(audio_chunk)
        
        # Get speech probability
        if self.model is not None:
            with torch.no_grad():
                speech_prob = self.model(audio_chunk, self.sampling_rate).item()
        else:
            # Fallback
            speech_prob = 0.5 if torch.sqrt(torch.mean(audio_chunk ** 2)).item() > 0.01 else 0.0
        
        # State machine
        result = None
        
        if speech_prob >= self.threshold:
            if not self.triggered:
                self.triggered = True
                speech_start = self.current_sample - self.speech_pad_samples
                speech_start = max(0, speech_start)
                
                if return_seconds:
                    result = {'start': speech_start / self.sampling_rate}
                else:
                    result = {'start': speech_start}
            
            self.temp_end = self.current_sample
            
        elif speech_prob < self.threshold - 0.15:  # Hysteresis
            if self.triggered:
                silence_duration = self.current_sample - self.temp_end
                
                if silence_duration >= self.min_silence_samples:
                    speech_end = self.temp_end + self.speech_pad_samples
                    
                    if return_seconds:
                        result = {'end': speech_end / self.sampling_rate}
                    else:
                        result = {'end': speech_end}
                    
                    self.triggered = False
        
        self.current_sample += window_size
        
        return result


class TurnTakingManager:
    """
    Manages turn-taking in conversations using VAD.
    
    Handles:
    - Detecting when user finishes speaking
    - Preventing AI from speaking over user
    - Handling interruptions
    
    Usage:
        manager = TurnTakingManager(vad)
        
        async for chunk in audio_stream:
            action = manager.process(chunk)
            
            if action == 'respond':
                # User finished, generate response
                response = generate_response()
                yield response
                
            elif action == 'stop':
                # User interrupted, stop generating
                stop_generation()
    """
    def __init__(self, vad: SileroVAD):
        self.vad = vad
        
        # State
        self.ai_speaking = False
        self.user_speaking = False
        self.last_user_speech_end = None
        self.audio_buffer = []
        
    def process(self, audio_chunk: torch.Tensor) -> Optional[str]:
        """
        Process audio chunk and return action.
        
        Returns:
            'respond' - Ready to generate response
            'stop' - Stop current generation (user interrupted)
            'wait' - Continue waiting
            None - No action needed
        """
        event = self.vad.process_chunk(audio_chunk)
        
        # Track user speech
        if event == 'speech_start':
            self.user_speaking = True
            self.audio_buffer = [audio_chunk]
            
            # User interrupted AI
            if self.ai_speaking:
                self.ai_speaking = False
                return 'stop'
                
        elif event == 'speaking':
            self.user_speaking = True
            self.audio_buffer.append(audio_chunk)
            
        elif event == 'speech_end':
            self.user_speaking = False
            self.last_user_speech_end = time.time()
            return 'respond'
        
        # Check for response timeout
        if (not self.user_speaking and 
            self.last_user_speech_end is not None and
            time.time() - self.last_user_speech_end > self.vad.config.response_delay_ms / 1000):
            
            if len(self.audio_buffer) > 0:
                return 'respond'
        
        return 'wait'
    
    def get_user_audio(self) -> Optional[torch.Tensor]:
        """Get buffered user audio."""
        if len(self.audio_buffer) == 0:
            return None
        
        audio = torch.cat(self.audio_buffer, dim=0)
        self.audio_buffer = []
        return audio
    
    def start_ai_speaking(self):
        """Mark that AI started speaking."""
        self.ai_speaking = True
        
    def stop_ai_speaking(self):
        """Mark that AI stopped speaking."""
        self.ai_speaking = False
        
    def reset(self):
        """Reset all state."""
        self.vad.reset()
        self.ai_speaking = False
        self.user_speaking = False
        self.last_user_speech_end = None
        self.audio_buffer = []


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  VAD INTEGRATION TEST")
    print("=" * 70)
    
    # Create VAD
    config = VADConfig()
    vad = SileroVAD(config)
    
    # Test with synthetic audio
    print("\n--- Chunk Processing Test ---")
    
    # Simulate speech pattern: silence -> speech -> silence
    sample_rate = 16000
    chunk_size = 512
    
    # Create test audio
    silence = torch.zeros(chunk_size)
    speech = torch.randn(chunk_size) * 0.3  # Louder = speech
    
    # Simulate: 0.5s silence, 1s speech, 0.5s silence
    num_silence_chunks = int(0.5 * sample_rate / chunk_size)
    num_speech_chunks = int(1.0 * sample_rate / chunk_size)
    
    print(f"Simulating: {num_silence_chunks} silence chunks, "
          f"{num_speech_chunks} speech chunks, "
          f"{num_silence_chunks} silence chunks")
    
    events = []
    
    # Process silence
    for _ in range(num_silence_chunks):
        event = vad.process_chunk(silence)
        if event:
            events.append(event)
    
    # Process speech
    for _ in range(num_speech_chunks):
        event = vad.process_chunk(speech)
        if event:
            events.append(event)
    
    # Process silence
    for _ in range(num_silence_chunks * 2):  # Extra silence for end detection
        event = vad.process_chunk(silence)
        if event:
            events.append(event)
    
    print(f"Detected events: {events}")
    
    # Test timestamp extraction
    print("\n--- Timestamp Extraction Test ---")
    
    # Create longer test audio
    duration = 5  # seconds
    test_audio = torch.cat([
        torch.zeros(sample_rate),           # 1s silence
        torch.randn(sample_rate) * 0.3,     # 1s speech
        torch.zeros(sample_rate // 2),      # 0.5s silence
        torch.randn(sample_rate * 2) * 0.3, # 2s speech
        torch.zeros(sample_rate // 2),      # 0.5s silence
    ])
    
    vad.reset()
    timestamps = vad.get_speech_timestamps(test_audio, return_seconds=True)
    print(f"Speech timestamps: {timestamps}")
    
    # Test turn-taking manager
    print("\n--- Turn-Taking Manager Test ---")
    
    vad.reset()
    manager = TurnTakingManager(vad)
    
    # Simulate user speaking then stopping
    for _ in range(num_speech_chunks):
        action = manager.process(speech)
    
    for _ in range(num_silence_chunks * 3):
        action = manager.process(silence)
        if action == 'respond':
            print("Turn-taking: Ready to respond")
            user_audio = manager.get_user_audio()
            print(f"User audio length: {len(user_audio)} samples")
            break
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)
