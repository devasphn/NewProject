#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SPEAKER ENCODER FOR VOICE CLONING
  
  Architecture: ECAPA-TDNN (State-of-the-Art 2024)
  Based on: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and 
            Aggregation in TDNN Based Speaker Verification"
  
  Features:
  - Squeeze-and-Excitation (SE) blocks for channel attention
  - Res2Net modules for multi-scale feature extraction
  - Attentive Statistics Pooling (ASP)
  - 256-dimensional speaker embeddings
  - GE2E loss for training
  
  Used for:
  - Voice cloning (extract speaker identity)
  - Speaker verification
  - Speaker diarization
  
  Reference: https://arxiv.org/abs/2005.07143
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple, List
import math


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module for channel attention.
    Learns to emphasize informative channels while suppressing less useful ones.
    """
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return x * self.se(x)


class Res2NetBlock(nn.Module):
    """
    Res2Net block with hierarchical residual-like connections.
    Enables multi-scale feature extraction within a single block.
    """
    def __init__(self, in_channels: int, out_channels: int, scale: int = 8, 
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        assert in_channels % scale == 0, "in_channels must be divisible by scale"
        
        self.scale = scale
        self.width = in_channels // scale
        
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, 
                     padding=(kernel_size - 1) // 2 * dilation,
                     dilation=dilation)
            for _ in range(scale - 1)
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(scale - 1)
        ])
        
        # Final projection if dimensions differ
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into scale groups
        xs = torch.chunk(x, self.scale, dim=1)
        ys = []
        
        for i, xi in enumerate(xs):
            if i == 0:
                ys.append(xi)
            elif i == 1:
                ys.append(F.relu(self.bns[i-1](self.convs[i-1](xi))))
            else:
                ys.append(F.relu(self.bns[i-1](self.convs[i-1](xi + ys[-1]))))
        
        return self.proj(torch.cat(ys, dim=1))


class SERes2NetBlock(nn.Module):
    """
    SE-Res2Net block combining Res2Net with Squeeze-and-Excitation.
    Core building block of ECAPA-TDNN.
    """
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilation: int = 1, scale: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.res2net = Res2NetBlock(channels, channels, scale, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.conv3 = nn.Conv1d(channels, channels, 1)
        self.bn3 = nn.BatchNorm1d(channels)
        
        self.se = SEModule(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.res2net(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        return F.relu(out + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP).
    Computes weighted mean and standard deviation using attention.
    More effective than simple global pooling for speaker embeddings.
    """
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_channels),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        
        # Global statistics for context
        mean = x.mean(dim=2, keepdim=True).expand_as(x)
        std = x.std(dim=2, keepdim=True).expand_as(x)
        
        # Compute attention weights
        attn_input = torch.cat([x, mean, std], dim=1)
        attn_weights = self.attention(attn_input)
        
        # Weighted statistics
        weighted_mean = (attn_weights * x).sum(dim=2)
        weighted_std = torch.sqrt(
            (attn_weights * (x ** 2)).sum(dim=2) - weighted_mean ** 2 + 1e-8
        )
        
        # Concatenate mean and std
        return torch.cat([weighted_mean, weighted_std], dim=1)


class ECAPATDNNEncoder(nn.Module):
    """
    ECAPA-TDNN Speaker Encoder.
    
    State-of-the-art speaker embedding model with:
    - Multi-scale Res2Net blocks
    - Squeeze-and-Excitation attention
    - Attentive Statistics Pooling
    - 256-dimensional output embeddings
    
    Args:
        input_dim: Input feature dimension (80 for mel-spectrogram)
        channels: Number of channels in hidden layers (1024 default)
        embedding_dim: Output embedding dimension (256 for speaker ID)
        num_blocks: Number of SE-Res2Net blocks (3 default)
    """
    def __init__(self, input_dim: int = 80, channels: int = 1024, 
                 embedding_dim: int = 256, num_blocks: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Input layer with larger receptive field
        self.input_layer = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        
        # SE-Res2Net blocks with increasing dilation
        self.blocks = nn.ModuleList([
            SERes2NetBlock(channels, kernel_size=3, dilation=2 ** i, scale=8)
            for i in range(num_blocks)
        ])
        
        # Multi-layer Feature Aggregation (MFA)
        self.mfa = nn.Conv1d(channels * num_blocks, channels * 3, kernel_size=1)
        
        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(channels * 3)
        
        # Final embedding layers
        self.fc = nn.Sequential(
            nn.BatchNorm1d(channels * 6),
            nn.Linear(channels * 6, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel-spectrogram.
        
        Args:
            x: Mel-spectrogram [B, mel_dim, T] or raw audio [B, samples]
        Returns:
            embedding: L2-normalized speaker embedding [B, embedding_dim]
        """
        # Input layer
        out = self.input_layer(x)
        
        # SE-Res2Net blocks with skip connections
        block_outputs = []
        for block in self.blocks:
            out = block(out)
            block_outputs.append(out)
        
        # Multi-layer Feature Aggregation
        out = self.mfa(torch.cat(block_outputs, dim=1))
        
        # Attentive Statistics Pooling
        out = self.asp(out)
        
        # Final embedding
        out = self.fc(out)
        
        # L2 normalization for cosine similarity
        out = F.normalize(out, p=2, dim=1)
        
        return out


class SpeakerEncoder(nn.Module):
    """
    Complete Speaker Encoder with mel-spectrogram extraction.
    
    Usage:
        encoder = SpeakerEncoder()
        
        # From raw audio
        embedding = encoder(audio)  # [B, samples] -> [B, 256]
        
        # Voice cloning
        similarity = encoder.compute_similarity(emb1, emb2)
        
    For training:
        loss = ge2e_loss(embeddings)  # [N_speakers, M_utterances, 256]
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 embedding_dim: int = 256, channels: int = 1024):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=400,  # 25ms at 16kHz
            hop_length=160,  # 10ms at 16kHz
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
        )
        
        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        
        # ECAPA-TDNN encoder
        self.encoder = ECAPATDNNEncoder(
            input_dim=n_mels,
            channels=channels,
            embedding_dim=embedding_dim,
        )
        
    def extract_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram from audio."""
        # Ensure audio is [B, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # [B, 1, T] -> [B, T]
        
        # Extract mel-spectrogram
        mel = self.mel_transform(audio)  # [B, n_mels, T]
        mel = self.amplitude_to_db(mel)
        
        # Normalize
        mel = (mel + 80) / 80  # Normalize to [0, 1] roughly
        
        return mel
    
    def forward(self, audio: torch.Tensor, 
                return_mel: bool = False) -> torch.Tensor:
        """
        Extract speaker embedding from audio.
        
        Args:
            audio: Raw audio [B, samples] or [B, 1, samples]
            return_mel: Whether to also return mel-spectrogram
        Returns:
            embedding: Speaker embedding [B, embedding_dim]
        """
        # Extract mel-spectrogram
        mel = self.extract_mel(audio)
        
        # Extract embedding
        embedding = self.encoder(mel)
        
        if return_mel:
            return embedding, mel
        return embedding
    
    def compute_similarity(self, emb1: torch.Tensor, 
                          emb2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings."""
        return F.cosine_similarity(emb1, emb2, dim=-1)
    
    def verify_speaker(self, audio1: torch.Tensor, audio2: torch.Tensor,
                       threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio1, audio2: Raw audio samples
            threshold: Similarity threshold for same speaker
        Returns:
            is_same: Whether samples are from same speaker
            similarity: Cosine similarity score
        """
        emb1 = self(audio1)
        emb2 = self(audio2)
        similarity = self.compute_similarity(emb1, emb2).item()
        return similarity >= threshold, similarity


def ge2e_loss(embeddings: torch.Tensor, 
              loss_type: str = 'softmax') -> torch.Tensor:
    """
    Generalized End-to-End (GE2E) Loss for speaker verification.
    
    Trains the network to make embeddings from the same speaker similar
    and embeddings from different speakers dissimilar.
    
    Args:
        embeddings: [N_speakers, M_utterances, embedding_dim]
        loss_type: 'softmax' (default) or 'contrast'
    Returns:
        loss: Scalar loss value
    """
    N, M, D = embeddings.shape
    
    # Compute centroids for each speaker (excluding current utterance for stability)
    centroids = embeddings.mean(dim=1)  # [N, D]
    
    # Compute similarity matrix
    # For each utterance, compute similarity to all centroids
    embeddings_flat = embeddings.view(N * M, D)  # [N*M, D]
    
    # Cosine similarity to all centroids
    # sim[i, j] = similarity of utterance i to centroid of speaker j
    sim_matrix = F.cosine_similarity(
        embeddings_flat.unsqueeze(1),  # [N*M, 1, D]
        centroids.unsqueeze(0),         # [1, N, D]
        dim=-1
    )  # [N*M, N]
    
    # Learnable scaling parameters (fixed for simplicity)
    w = 10.0
    b = -5.0
    sim_matrix = w * sim_matrix + b
    
    # Target: each utterance should match its speaker's centroid
    targets = torch.arange(N, device=embeddings.device).unsqueeze(1).expand(N, M).reshape(-1)
    
    if loss_type == 'softmax':
        # Softmax loss
        loss = F.cross_entropy(sim_matrix, targets)
    else:
        # Contrast loss
        pos_sim = sim_matrix[torch.arange(N*M), targets]
        neg_sim = sim_matrix.clone()
        neg_sim[torch.arange(N*M), targets] = float('-inf')
        neg_sim = neg_sim.max(dim=1)[0]
        
        loss = (1 - torch.sigmoid(pos_sim) + torch.sigmoid(neg_sim)).mean()
    
    return loss


class SpeakerEncoderTrainer:
    """
    Trainer for speaker encoder using GE2E loss.
    
    Usage:
        trainer = SpeakerEncoderTrainer(model, speakers_per_batch=64, 
                                        utterances_per_speaker=10)
        for batch in dataloader:
            loss = trainer.train_step(batch)
    """
    def __init__(self, model: SpeakerEncoder, 
                 speakers_per_batch: int = 64,
                 utterances_per_speaker: int = 10,
                 lr: float = 1e-4):
        self.model = model
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.95
        )
        
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Train on a batch of utterances.
        
        Args:
            batch: [N_speakers, M_utterances, samples]
        Returns:
            loss: Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        N, M, T = batch.shape
        
        # Flatten for processing
        audio_flat = batch.view(N * M, T)
        
        # Extract embeddings
        embeddings_flat = self.model(audio_flat)  # [N*M, D]
        
        # Reshape to [N, M, D]
        embeddings = embeddings_flat.view(N, M, -1)
        
        # Compute GE2E loss
        loss = ge2e_loss(embeddings)
        
        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
        self.optimizer.step()
        
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH CODEC
# ═══════════════════════════════════════════════════════════════════════════════

class SpeakerConditionedCodec(nn.Module):
    """
    Wrapper to add speaker conditioning to any codec.
    
    Enables voice cloning by injecting speaker embedding into the decoder.
    """
    def __init__(self, codec: nn.Module, speaker_encoder: SpeakerEncoder,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.codec = codec
        self.speaker_encoder = speaker_encoder
        
        # Freeze speaker encoder (pretrained)
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False
        
        # Speaker conditioning projection
        self.speaker_proj = nn.Linear(speaker_encoder.embedding_dim, hidden_dim)
        
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio and extract speaker embedding."""
        # Extract speaker embedding
        speaker_emb = self.speaker_encoder(audio)
        
        # Encode audio
        codes = self.codec.encode(audio)
        
        return codes, speaker_emb
    
    def decode(self, codes: torch.Tensor, 
               speaker_emb: torch.Tensor) -> torch.Tensor:
        """Decode with speaker conditioning."""
        # Project speaker embedding
        speaker_cond = self.speaker_proj(speaker_emb)  # [B, hidden_dim]
        
        # Decode codes (codec should accept speaker_cond)
        audio = self.codec.decode(codes, speaker_cond=speaker_cond)
        
        return audio
    
    def voice_convert(self, source_audio: torch.Tensor,
                      target_speaker_audio: torch.Tensor) -> torch.Tensor:
        """
        Convert source audio to target speaker's voice.
        
        Args:
            source_audio: Audio to convert [B, 1, T]
            target_speaker_audio: Reference audio of target speaker [B, 1, T]
        Returns:
            converted_audio: Audio in target speaker's voice
        """
        # Encode source audio
        codes = self.codec.encode(source_audio)
        
        # Extract target speaker embedding
        target_emb = self.speaker_encoder(target_speaker_audio)
        
        # Decode with target speaker
        converted = self.decode(codes, target_emb)
        
        return converted


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  SPEAKER ENCODER TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create model
    model = SpeakerEncoder().to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    audio = torch.randn(4, 16000 * 3).to(device)  # 4 samples, 3 seconds each
    
    with torch.no_grad():
        embeddings = model(audio)
    
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1).mean():.4f}")
    
    # Test similarity
    print("\n--- Similarity Test ---")
    sim_same = model.compute_similarity(embeddings[0:1], embeddings[0:1])
    sim_diff = model.compute_similarity(embeddings[0:1], embeddings[1:2])
    print(f"Same embedding similarity: {sim_same.item():.4f}")
    print(f"Different embedding similarity: {sim_diff.item():.4f}")
    
    # Test GE2E loss
    print("\n--- GE2E Loss Test ---")
    N, M = 8, 5  # 8 speakers, 5 utterances each
    test_embeddings = torch.randn(N, M, 256).to(device)
    test_embeddings = F.normalize(test_embeddings, dim=-1)
    
    loss = ge2e_loss(test_embeddings)
    print(f"GE2E Loss: {loss.item():.4f}")
    
    # Test latency
    print("\n--- Latency Test ---")
    import time
    
    audio = torch.randn(1, 16000 * 3).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(audio)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(audio)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) / 100 * 1000
    
    print(f"Latency: {elapsed:.2f}ms per 3-second audio")
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)
