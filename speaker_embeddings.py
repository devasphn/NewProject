#!/usr/bin/env python3
"""
Speaker Embedding System for Telugu S2S
Implements 4 distinct speaker voices with emotional characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class SpeakerEmbeddingSystem(nn.Module):
    """
    Advanced speaker embedding system with:
    - 4 pre-trained speaker voices (2 male, 2 female)
    - Emotional modulation capabilities
    - Telugu accent control
    - Voice characteristic preservation
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Speaker definitions with characteristics
        self.speakers = {
            "male_young": {
                "id": 0,
                "name": "Arjun",
                "age_range": "25-30",
                "pitch": 120,  # Hz
                "characteristics": {
                    "energy": 0.8,
                    "speaking_rate": 1.1,
                    "formality": 0.4,
                    "emotion_range": 0.9
                },
                "description": "Young professional with energetic voice"
            },
            "male_mature": {
                "id": 1,
                "name": "Ravi",
                "age_range": "35-45",
                "pitch": 100,  # Hz
                "characteristics": {
                    "energy": 0.6,
                    "speaking_rate": 0.9,
                    "formality": 0.8,
                    "emotion_range": 0.7
                },
                "description": "Mature narrator with calm, authoritative voice"
            },
            "female_young": {
                "id": 2,
                "name": "Priya",
                "age_range": "22-28",
                "pitch": 220,  # Hz
                "characteristics": {
                    "energy": 0.85,
                    "speaking_rate": 1.15,
                    "formality": 0.3,
                    "emotion_range": 0.95
                },
                "description": "Young conversational voice with high expressiveness"
            },
            "female_professional": {
                "id": 3,
                "name": "Lakshmi",
                "age_range": "30-40",
                "pitch": 190,  # Hz
                "characteristics": {
                    "energy": 0.7,
                    "speaking_rate": 1.0,
                    "formality": 0.9,
                    "emotion_range": 0.6
                },
                "description": "Professional news anchor with clear articulation"
            }
        }
        
        # Learnable speaker embeddings
        self.speaker_embeddings = nn.Embedding(4, embedding_dim)
        
        # Prosody control layers
        self.pitch_control = nn.Linear(1, 64)
        self.energy_control = nn.Linear(1, 64)
        self.rate_control = nn.Linear(1, 64)
        
        # Emotion modulation layer
        self.emotion_modulator = nn.Linear(embedding_dim, embedding_dim)
        
        # Telugu accent control
        self.accent_embeddings = nn.Embedding(3, 64)  # mild, moderate, heavy
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim + 64*3 + 64, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize embeddings with distinct characteristics
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize speaker embeddings with distinct characteristics"""
        with torch.no_grad():
            # Initialize each speaker with unique pattern
            for speaker_name, speaker_info in self.speakers.items():
                speaker_id = speaker_info["id"]
                
                # Create base embedding
                base_embedding = torch.randn(self.embedding_dim)
                
                # Add speaker-specific patterns
                if "male" in speaker_name:
                    base_embedding[::2] *= 1.2  # Lower frequency emphasis
                else:
                    base_embedding[1::2] *= 1.2  # Higher frequency emphasis
                
                if "young" in speaker_name:
                    base_embedding[:self.embedding_dim//2] *= 1.1  # More energy
                else:
                    base_embedding[self.embedding_dim//2:] *= 1.1  # More stability
                
                # Normalize and assign
                base_embedding = F.normalize(base_embedding, dim=0)
                self.speaker_embeddings.weight[speaker_id] = base_embedding
    
    def forward(
        self,
        speaker_id: torch.Tensor,
        emotion_embedding: Optional[torch.Tensor] = None,
        accent_level: int = 1,
        custom_prosody: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Generate speaker embedding with emotional and prosodic control
        
        Args:
            speaker_id: Tensor of speaker IDs [batch_size]
            emotion_embedding: Optional emotion embeddings [batch_size, embedding_dim]
            accent_level: Telugu accent level (0=mild, 1=moderate, 2=heavy)
            custom_prosody: Optional prosody overrides (pitch, energy, rate)
        
        Returns:
            Speaker embeddings [batch_size, embedding_dim]
        """
        batch_size = speaker_id.size(0)
        device = speaker_id.device
        
        # Get base speaker embedding
        speaker_embed = self.speaker_embeddings(speaker_id)
        
        # Get speaker characteristics
        prosody_features = []
        for i in range(batch_size):
            sid = speaker_id[i].item()
            speaker_info = list(self.speakers.values())[sid]
            chars = speaker_info["characteristics"]
            
            # Apply custom prosody if provided
            if custom_prosody:
                pitch = custom_prosody.get("pitch", speaker_info["pitch"]) / 200.0
                energy = custom_prosody.get("energy", chars["energy"])
                rate = custom_prosody.get("rate", chars["speaking_rate"])
            else:
                pitch = speaker_info["pitch"] / 200.0
                energy = chars["energy"]
                rate = chars["speaking_rate"]
            
            # Process prosody features
            pitch_feat = self.pitch_control(torch.tensor([pitch], device=device))
            energy_feat = self.energy_control(torch.tensor([energy], device=device))
            rate_feat = self.rate_control(torch.tensor([rate], device=device))
            
            prosody_features.append(torch.cat([pitch_feat, energy_feat, rate_feat]))
        
        prosody_features = torch.stack(prosody_features)
        
        # Add accent embedding
        accent_embed = self.accent_embeddings(
            torch.tensor([accent_level] * batch_size, device=device)
        )
        
        # Combine all features
        combined = torch.cat([speaker_embed, prosody_features, accent_embed], dim=1)
        speaker_output = self.feature_fusion(combined)
        
        # Apply emotion modulation if provided
        if emotion_embedding is not None:
            speaker_output = speaker_output + 0.3 * self.emotion_modulator(emotion_embedding)
        
        return F.normalize(speaker_output, dim=-1)
    
    def get_speaker_info(self, speaker_id: int) -> Dict:
        """Get detailed information about a speaker"""
        for speaker_info in self.speakers.values():
            if speaker_info["id"] == speaker_id:
                return speaker_info
        return None
    
    def interpolate_speakers(
        self,
        speaker1_id: int,
        speaker2_id: int,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between two speakers for voice morphing
        
        Args:
            speaker1_id: First speaker ID
            speaker2_id: Second speaker ID
            alpha: Interpolation factor (0 = speaker1, 1 = speaker2)
        
        Returns:
            Interpolated speaker embedding
        """
        embed1 = self.speaker_embeddings.weight[speaker1_id]
        embed2 = self.speaker_embeddings.weight[speaker2_id]
        
        # Spherical linear interpolation for better quality
        omega = torch.acos(torch.clamp(torch.dot(embed1, embed2), -1, 1))
        so = torch.sin(omega)
        
        if so < 1e-6:  # Vectors are parallel
            return embed1 * (1 - alpha) + embed2 * alpha
        
        return (torch.sin((1.0 - alpha) * omega) / so) * embed1 + \
               (torch.sin(alpha * omega) / so) * embed2
    
    def save_embeddings(self, path: str):
        """Save speaker embeddings to file"""
        save_dict = {
            "embeddings": self.speaker_embeddings.weight.detach().cpu().numpy().tolist(),
            "speakers": self.speakers,
            "embedding_dim": self.embedding_dim
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Speaker embeddings saved to {path}")
    
    def load_embeddings(self, path: str):
        """Load speaker embeddings from file"""
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        embeddings = torch.tensor(save_dict["embeddings"])
        self.speaker_embeddings.weight.data = embeddings
        self.speakers = save_dict["speakers"]
        
        print(f"Speaker embeddings loaded from {path}")


class SpeakerDataAugmentation:
    """
    Data augmentation for speaker diversity during training
    """
    
    def __init__(self):
        self.pitch_shift_range = (-5, 5)  # semitones
        self.tempo_range = (0.9, 1.1)
        self.formant_shift_range = (0.95, 1.05)
    
    def augment_for_speaker(
        self,
        audio: torch.Tensor,
        speaker_id: int,
        augment_prob: float = 0.3
    ) -> torch.Tensor:
        """
        Apply speaker-specific augmentations
        
        Args:
            audio: Input audio tensor
            speaker_id: Target speaker ID
            augment_prob: Probability of applying augmentation
        
        Returns:
            Augmented audio
        """
        if torch.rand(1).item() > augment_prob:
            return audio
        
        # Apply pitch shift
        if torch.rand(1).item() < 0.5:
            pitch_shift = torch.randint(
                self.pitch_shift_range[0],
                self.pitch_shift_range[1],
                (1,)
            ).item()
            audio = self._pitch_shift(audio, pitch_shift)
        
        # Apply tempo change
        if torch.rand(1).item() < 0.3:
            tempo_factor = torch.rand(1).item() * \
                          (self.tempo_range[1] - self.tempo_range[0]) + \
                          self.tempo_range[0]
            audio = self._tempo_change(audio, tempo_factor)
        
        return audio
    
    def _pitch_shift(self, audio: torch.Tensor, semitones: int) -> torch.Tensor:
        """Apply pitch shift using phase vocoder"""
        # Simplified implementation - would use librosa in production
        shift_factor = 2 ** (semitones / 12)
        return F.interpolate(
            audio.unsqueeze(0),
            scale_factor=shift_factor,
            mode='linear'
        ).squeeze(0)
    
    def _tempo_change(self, audio: torch.Tensor, factor: float) -> torch.Tensor:
        """Change tempo without affecting pitch"""
        # Simplified implementation
        return F.interpolate(
            audio.unsqueeze(0),
            scale_factor=1/factor,
            mode='linear'
        ).squeeze(0)


def create_speaker_dataset_splits(
    data_dir: Path,
    speaker_mapping: Dict[str, int]
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Create dataset splits with speaker labels
    
    Args:
        data_dir: Directory containing audio data
        speaker_mapping: Mapping from data source to speaker ID
    
    Returns:
        Dictionary with train/val/test splits
    """
    splits = {"train": [], "val": [], "test": []}
    
    # Map data sources to speakers
    source_to_speaker = {
        "raw_talks_male": 0,      # Young male podcaster
        "news_anchor_male": 1,    # Mature male news
        "raw_talks_female": 2,    # Young female podcaster
        "news_anchor_female": 3,  # Professional female news
    }
    
    # Process each data source
    for source, speaker_id in source_to_speaker.items():
        source_dir = data_dir / source
        if source_dir.exists():
            audio_files = list(source_dir.glob("*.wav"))
            
            # Split 80/10/10
            n_files = len(audio_files)
            n_train = int(0.8 * n_files)
            n_val = int(0.1 * n_files)
            
            for i, audio_file in enumerate(audio_files):
                if i < n_train:
                    splits["train"].append((str(audio_file), speaker_id))
                elif i < n_train + n_val:
                    splits["val"].append((str(audio_file), speaker_id))
                else:
                    splits["test"].append((str(audio_file), speaker_id))
    
    return splits


if __name__ == "__main__":
    # Test speaker embedding system
    system = SpeakerEmbeddingSystem(embedding_dim=256)
    
    # Test forward pass
    batch_size = 4
    speaker_ids = torch.tensor([0, 1, 2, 3])
    
    embeddings = system(speaker_ids, accent_level=1)
    print(f"Speaker embeddings shape: {embeddings.shape}")
    
    # Test speaker interpolation
    interpolated = system.interpolate_speakers(0, 1, alpha=0.5)
    print(f"Interpolated embedding shape: {interpolated.shape}")
    
    # Save embeddings
    system.save_embeddings("speaker_embeddings.json")
    
    # Print speaker info
    print("\nAvailable speakers:")
    for speaker_name, info in system.speakers.items():
        print(f"  {speaker_name}: {info['description']}")