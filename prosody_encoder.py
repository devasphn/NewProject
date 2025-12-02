#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  PROSODY ENCODER FOR NATURAL SPEECH SYNTHESIS
  
  Extracts and encodes prosodic features from speech:
  - F0 (Pitch) contour
  - Energy (Loudness) contour
  - Duration (Speaking rate)
  - Pause patterns
  
  Used for:
  - Prosody transfer ("speak like this")
  - Emotion-aware synthesis
  - Speaking rate control
  - Natural speech generation
  
  Reference: FastSpeech2, Meta Voicebox prosody modeling
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math
import warnings

# Try to import librosa for pitch extraction
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not installed. Using torch-based pitch extraction.")


@dataclass
class ProsodyConfig:
    """Configuration for prosody encoder."""
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    
    # F0 extraction
    f0_min: float = 50.0   # Hz (low male voice)
    f0_max: float = 600.0  # Hz (high female voice)
    
    # Feature dimensions
    pitch_dim: int = 256
    energy_dim: int = 256
    duration_dim: int = 256
    
    # Prosody embedding dimension
    prosody_dim: int = 512
    
    # Quantization
    num_pitch_bins: int = 256
    num_energy_bins: int = 256


class PitchExtractor(nn.Module):
    """
    Extract F0 (pitch) contour from audio.
    
    Uses:
    - PYIN algorithm (if librosa available)
    - Autocorrelation-based method (fallback)
    """
    def __init__(self, config: ProsodyConfig):
        super().__init__()
        
        self.config = config
        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.f0_min = config.f0_min
        self.f0_max = config.f0_max
        
        # For torch-based pitch estimation
        self.frame_length = 1024
        
    def extract_librosa(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract pitch using librosa PYIN (most accurate)."""
        device = audio.device
        audio_np = audio.cpu().numpy()
        
        batch_size = audio_np.shape[0]
        results = []
        voiced_flags = []
        
        for i in range(batch_size):
            f0, voiced, _ = librosa.pyin(
                audio_np[i],
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sample_rate,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
            
            # Replace NaN with 0
            f0 = torch.from_numpy(f0).float()
            f0 = torch.nan_to_num(f0, nan=0.0)
            
            voiced = torch.from_numpy(voiced).float()
            
            results.append(f0)
            voiced_flags.append(voiced)
        
        pitch = torch.stack(results, dim=0).to(device)
        voiced = torch.stack(voiced_flags, dim=0).to(device)
        
        return pitch, voiced
    
    def extract_torch(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pitch using torchaudio (faster, GPU-compatible).
        Uses autocorrelation-based method.
        """
        # Simple zero-crossing based pitch estimation
        # Not as accurate as PYIN but works without librosa
        
        B, T = audio.shape
        frame_length = self.frame_length
        hop_length = self.hop_length
        
        # Compute number of frames
        num_frames = (T - frame_length) // hop_length + 1
        
        # Frame the audio
        frames = audio.unfold(1, frame_length, hop_length)  # [B, num_frames, frame_length]
        
        # Apply window
        window = torch.hann_window(frame_length, device=audio.device)
        frames = frames * window
        
        # Autocorrelation
        fft = torch.fft.rfft(frames, n=frame_length * 2)
        autocorr = torch.fft.irfft(fft * fft.conj())[:, :, :frame_length]
        
        # Find pitch period
        min_period = int(self.sample_rate / self.f0_max)
        max_period = int(self.sample_rate / self.f0_min)
        
        autocorr_search = autocorr[:, :, min_period:max_period]
        peak_idx = autocorr_search.argmax(dim=-1) + min_period
        
        # Convert to F0
        pitch = self.sample_rate / peak_idx.float()
        
        # Voiced detection (autocorr peak strength)
        peak_values = autocorr[:, :, 0].abs()
        voiced = (peak_values > 0.1).float()
        
        # Apply voiced mask
        pitch = pitch * voiced
        
        return pitch, voiced
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pitch contour.
        
        Args:
            audio: [B, samples]
        Returns:
            pitch: [B, T] - F0 in Hz (0 for unvoiced)
            voiced: [B, T] - voiced flag
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        if HAS_LIBROSA and not audio.is_cuda:
            return self.extract_librosa(audio)
        else:
            return self.extract_torch(audio)


class EnergyExtractor(nn.Module):
    """
    Extract energy (loudness) contour from audio.
    """
    def __init__(self, config: ProsodyConfig):
        super().__init__()
        
        self.hop_length = config.hop_length
        self.frame_length = 1024
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract energy contour.
        
        Args:
            audio: [B, samples]
        Returns:
            energy: [B, T] - RMS energy per frame
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        B, T = audio.shape
        
        # Frame the audio
        frames = audio.unfold(1, self.frame_length, self.hop_length)
        
        # Apply window
        window = torch.hann_window(self.frame_length, device=audio.device)
        frames = frames * window
        
        # RMS energy
        energy = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)
        
        # Convert to dB
        energy_db = 20 * torch.log10(energy + 1e-8)
        
        # Normalize to [0, 1] (assuming -80dB to 0dB range)
        energy_norm = (energy_db + 80) / 80
        energy_norm = energy_norm.clamp(0, 1)
        
        return energy_norm


class DurationPredictor(nn.Module):
    """
    Predict phone/frame durations for speaking rate control.
    Based on FastSpeech2 duration predictor.
    """
    def __init__(self, hidden_dim: int = 256, kernel_size: int = 3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict log durations.
        
        Args:
            x: [B, T, D] - hidden features
        Returns:
            log_durations: [B, T]
        """
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # [B, T, D]
        
        log_durations = self.linear(x).squeeze(-1)
        
        return log_durations


class ProsodyEncoder(nn.Module):
    """
    Complete Prosody Encoder for speech synthesis.
    
    Extracts and encodes:
    - Pitch (F0) contour
    - Energy contour
    - Implied duration/speaking rate
    
    Usage:
        encoder = ProsodyEncoder()
        prosody_emb = encoder(audio)  # [B, T, prosody_dim]
        
        # For conditioning
        prosody_emb = encoder.get_utterance_prosody(audio)  # [B, prosody_dim]
    """
    def __init__(self, config: Optional[ProsodyConfig] = None):
        super().__init__()
        
        self.config = config or ProsodyConfig()
        
        # Feature extractors
        self.pitch_extractor = PitchExtractor(self.config)
        self.energy_extractor = EnergyExtractor(self.config)
        
        # Pitch embedding (quantized or continuous)
        self.pitch_embedding = nn.Sequential(
            nn.Linear(1, self.config.pitch_dim),
            nn.ReLU(),
            nn.Linear(self.config.pitch_dim, self.config.pitch_dim),
        )
        
        # Energy embedding
        self.energy_embedding = nn.Sequential(
            nn.Linear(1, self.config.energy_dim),
            nn.ReLU(),
            nn.Linear(self.config.energy_dim, self.config.energy_dim),
        )
        
        # Prosody encoder (processes pitch + energy together)
        self.prosody_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.pitch_dim + self.config.energy_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # Project to prosody embedding
        self.prosody_proj = nn.Linear(
            self.config.pitch_dim + self.config.energy_dim,
            self.config.prosody_dim
        )
        
        # Attention pooling for utterance-level prosody
        self.attention_pool = nn.Sequential(
            nn.Linear(self.config.prosody_dim, 1),
            nn.Softmax(dim=1),
        )
        
    def normalize_pitch(self, pitch: torch.Tensor, 
                        voiced: torch.Tensor) -> torch.Tensor:
        """Normalize pitch to log scale, handling unvoiced frames."""
        # Log pitch (only for voiced frames)
        log_pitch = torch.log(pitch + 1.0)  # +1 to handle 0
        
        # Normalize per utterance
        voiced_mask = voiced > 0.5
        
        # Compute mean/std only on voiced frames
        batch_size = pitch.shape[0]
        normalized = torch.zeros_like(log_pitch)
        
        for i in range(batch_size):
            voiced_pitch = log_pitch[i, voiced_mask[i]]
            if voiced_pitch.numel() > 0:
                mean = voiced_pitch.mean()
                std = voiced_pitch.std() + 1e-8
                normalized[i] = (log_pitch[i] - mean) / std
        
        # Zero out unvoiced frames
        normalized = normalized * voiced
        
        return normalized
    
    def forward(self, audio: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Extract prosody embedding from audio.
        
        Args:
            audio: [B, samples] or [B, 1, samples]
            return_features: Whether to also return raw pitch/energy
        Returns:
            prosody_emb: [B, T, prosody_dim] - frame-level prosody
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Extract features
        pitch, voiced = self.pitch_extractor(audio)
        energy = self.energy_extractor(audio)
        
        # Match lengths (use minimum)
        min_len = min(pitch.shape[1], energy.shape[1])
        pitch = pitch[:, :min_len]
        voiced = voiced[:, :min_len]
        energy = energy[:, :min_len]
        
        # Normalize pitch
        pitch_norm = self.normalize_pitch(pitch, voiced)
        
        # Embed pitch and energy
        pitch_emb = self.pitch_embedding(pitch_norm.unsqueeze(-1))
        energy_emb = self.energy_embedding(energy.unsqueeze(-1))
        
        # Concatenate
        prosody_features = torch.cat([pitch_emb, energy_emb], dim=-1)
        
        # Encode with transformer
        prosody_encoded = self.prosody_encoder(prosody_features)
        
        # Project to prosody embedding
        prosody_emb = self.prosody_proj(prosody_encoded)
        
        if return_features:
            return prosody_emb, {
                'pitch': pitch,
                'pitch_normalized': pitch_norm,
                'energy': energy,
                'voiced': voiced,
            }
        
        return prosody_emb
    
    def get_utterance_prosody(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Get utterance-level prosody embedding.
        
        Args:
            audio: [B, samples]
        Returns:
            prosody_emb: [B, prosody_dim]
        """
        frame_prosody = self(audio)  # [B, T, prosody_dim]
        
        # Attention pooling
        attn_weights = self.attention_pool(frame_prosody)  # [B, T, 1]
        utterance_prosody = (frame_prosody * attn_weights).sum(dim=1)  # [B, prosody_dim]
        
        return utterance_prosody


class ProsodyTransfer(nn.Module):
    """
    Module for prosody transfer between utterances.
    
    Given source content and target prosody, generate speech with
    source content but target speaker's prosody.
    """
    def __init__(self, prosody_encoder: ProsodyEncoder, 
                 hidden_dim: int = 512):
        super().__init__()
        
        self.prosody_encoder = prosody_encoder
        
        # Prosody conditioning layer
        self.prosody_condition = nn.Sequential(
            nn.Linear(prosody_encoder.config.prosody_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def extract_prosody(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract prosody embedding from reference audio."""
        return self.prosody_encoder.get_utterance_prosody(audio)
    
    def condition_hidden(self, hidden: torch.Tensor, 
                         prosody_emb: torch.Tensor) -> torch.Tensor:
        """
        Add prosody conditioning to hidden states.
        
        Args:
            hidden: [B, T, D] - content hidden states
            prosody_emb: [B, prosody_dim] - target prosody
        Returns:
            conditioned: [B, T, D]
        """
        prosody_cond = self.prosody_condition(prosody_emb)  # [B, D]
        prosody_cond = prosody_cond.unsqueeze(1)  # [B, 1, D]
        
        return hidden + prosody_cond


class ProsodyPredictor(nn.Module):
    """
    Predict prosody features for synthesis (text-to-prosody).
    
    Given hidden representations, predict pitch and energy contours.
    """
    def __init__(self, hidden_dim: int = 512, config: Optional[ProsodyConfig] = None):
        super().__init__()
        
        self.config = config or ProsodyConfig()
        
        # Pitch predictor
        self.pitch_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        
        # Energy predictor
        self.energy_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        
        # Voiced/unvoiced predictor
        self.vuv_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, 1),
        )
        
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict prosody features.
        
        Args:
            hidden: [B, T, D]
        Returns:
            Dict with pitch, energy, vuv predictions
        """
        x = hidden.transpose(1, 2)  # [B, D, T]
        
        pitch = self.pitch_predictor(x).squeeze(1)  # [B, T]
        energy = self.energy_predictor(x).squeeze(1)  # [B, T]
        vuv = torch.sigmoid(self.vuv_predictor(x).squeeze(1))  # [B, T]
        
        return {
            'pitch': pitch,
            'energy': energy,
            'vuv': vuv,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  PROSODY ENCODER TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Librosa available: {HAS_LIBROSA}")
    
    # Create model
    config = ProsodyConfig()
    model = ProsodyEncoder(config).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
    
    # Test with synthetic audio
    print("\n--- Feature Extraction Test ---")
    duration_sec = 3
    audio = torch.randn(4, 16000 * duration_sec).to(device)
    
    with torch.no_grad():
        prosody_emb, features = model(audio, return_features=True)
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Prosody embedding shape: {prosody_emb.shape}")
    print(f"Pitch shape: {features['pitch'].shape}")
    print(f"Energy shape: {features['energy'].shape}")
    
    # Test utterance-level prosody
    print("\n--- Utterance-Level Prosody Test ---")
    with torch.no_grad():
        utt_prosody = model.get_utterance_prosody(audio)
    
    print(f"Utterance prosody shape: {utt_prosody.shape}")
    
    # Latency test
    print("\n--- Latency Test ---")
    import time
    
    audio = torch.randn(1, 16000 * 2).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(audio)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(audio)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) / 50 * 1000
    
    print(f"Latency: {elapsed:.2f}ms per 2-second audio")
    
    # Test prosody predictor
    print("\n--- Prosody Predictor Test ---")
    predictor = ProsodyPredictor(hidden_dim=512).to(device)
    hidden = torch.randn(4, 100, 512).to(device)
    
    with torch.no_grad():
        pred = predictor(hidden)
    
    print(f"Hidden shape: {hidden.shape}")
    print(f"Predicted pitch shape: {pred['pitch'].shape}")
    print(f"Predicted energy shape: {pred['energy'].shape}")
    print(f"Predicted VUV shape: {pred['vuv'].shape}")
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)
