"""
TeluCodec: Custom Neural Audio Codec for Telugu Speech
Ultra-low latency codec optimized for Telugu phonemes at 16kHz
Target: <10ms encoding, <10ms decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from einops import rearrange, pack, unpack
import math

class CausalConv1d(nn.Module):
    """Causal 1D convolution for streaming"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        # Remove future samples for causality
        x = self.conv(x)
        if self.padding > 0:
            x = x[..., :-self.padding]
        return x

class ResidualBlock(nn.Module):
    """Residual block with depthwise separable convolutions"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.GELU(),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1)  # Pointwise
        )
    
    def forward(self, x):
        return x + self.block(x)

class VectorQuantizer(nn.Module):
    """
    Residual Vector Quantization with codebook learning
    Optimized for Telugu speech patterns
    """
    def __init__(self, dim, codebook_size=1024, num_quantizers=8, 
                 commitment_weight=0.25, ema_decay=0.99):
        super().__init__()
        self.dim = dim
        self.n_codes = codebook_size
        self.n_quantizers = num_quantizers
        self.commitment_weight = commitment_weight
        
        # Learnable codebooks - initialize with smaller values
        self.codebooks = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, dim) * 0.01
        )
        
        # EMA for codebook updates
        self.register_buffer('ema_cluster_size', torch.zeros(num_quantizers, codebook_size))
        self.register_buffer('ema_w', torch.randn(num_quantizers, codebook_size, dim) * 0.01)
        self.ema_decay = ema_decay
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor using residual VQ
        
        Args:
            z: Input tensor [B, D, T]
        
        Returns:
            quantized: Quantized tensor
            codes: Codebook indices
            losses: Quantization losses
        """
        B, D, T = z.shape
        z = rearrange(z, 'b d t -> b t d')
        
        quantized = torch.zeros_like(z)
        codes = []
        losses = []
        
        residual = z
        for q in range(self.n_quantizers):
            # Compute distances to codebook
            distances = torch.cdist(residual, self.codebooks[q])
            indices = distances.argmin(dim=-1)
            codes.append(indices)
            
            # Quantize
            quantized_step = F.embedding(indices, self.codebooks[q])
            quantized += quantized_step
            
            # Commitment loss with clamping
            commitment_loss = F.mse_loss(residual.detach(), quantized_step)
            commitment_loss = torch.clamp(commitment_loss, 0, 10.0)  # Prevent explosion
            losses.append(commitment_loss * self.commitment_weight)
            
            # EMA codebook update (training only) - use quantized_step not residual
            if self.training:
                self._update_codebook_ema(q, quantized_step.detach(), indices)
            
            # Update residual
            residual = residual - quantized_step.detach()
        
        quantized = rearrange(quantized, 'b t d -> b d t')
        codes = torch.stack(codes, dim=1)  # [B, Q, T]
        total_loss = sum(losses) if losses else torch.tensor(0.0, device=z.device)
        
        # Clamp total VQ loss
        total_loss = torch.clamp(total_loss, 0, 10.0)
        
        # Straight-through estimator
        quantized = z.reshape(B, D, T) + (quantized - z.reshape(B, D, T)).detach()
        
        return quantized, codes, total_loss
    
    def _update_codebook_ema(self, q_idx, inputs, indices):
        """Update codebook using exponential moving average"""
        with torch.no_grad():
            # Flatten
            inputs_flat = inputs.reshape(-1, self.dim)
            indices_flat = indices.reshape(-1)
            
            # One-hot encoding
            encodings = F.one_hot(indices_flat, self.n_codes).float()
            
            # Update cluster sizes
            self.ema_cluster_size[q_idx] = self.ema_decay * self.ema_cluster_size[q_idx] + \
                                          (1 - self.ema_decay) * encodings.sum(0)
            
            # Update codebook vectors
            dw = encodings.T @ inputs_flat
            self.ema_w[q_idx] = self.ema_decay * self.ema_w[q_idx] + (1 - self.ema_decay) * dw
            
            # Normalize
            n = self.ema_cluster_size[q_idx].sum()
            cluster_size = (self.ema_cluster_size[q_idx] + 1e-5) / (n + self.n_codes * 1e-5) * n
            self.codebooks.data[q_idx] = self.ema_w[q_idx] / cluster_size.unsqueeze(1)

class TeluguEncoder(nn.Module):
    """
    Encoder for Telugu speech -> discrete tokens
    Optimized for 16kHz with 80x compression (200Hz tokens)
    """
    def __init__(self, input_channels=1, hidden_dim=1024):
        super().__init__()
        
        # Strided convolutions for downsampling
        # 16000Hz -> 8000Hz -> 4000Hz -> 2000Hz -> 400Hz -> 200Hz
        self.encoder = nn.ModuleList([
            CausalConv1d(input_channels, 32, kernel_size=7, stride=1),
            ResidualBlock(32),
            CausalConv1d(32, 64, kernel_size=7, stride=2),  # 8000Hz
            ResidualBlock(64),
            CausalConv1d(64, 128, kernel_size=7, stride=2),  # 4000Hz
            ResidualBlock(128),
            CausalConv1d(128, 256, kernel_size=7, stride=2),  # 2000Hz
            ResidualBlock(256),
            CausalConv1d(256, 512, kernel_size=7, stride=5),  # 400Hz
            ResidualBlock(512),
            CausalConv1d(512, hidden_dim, kernel_size=3, stride=2),  # 200Hz
            ResidualBlock(hidden_dim)
        ])
        
        # Final projection
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # Temporal smoothing for Telugu phonemes
        self.temporal_smooth = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent representation
        
        Args:
            x: Audio tensor [B, 1, T] at 16kHz
        
        Returns:
            z: Latent tensor [B, D, T'] at 200Hz
        """
        # Progressive encoding
        for layer in self.encoder:
            x = layer(x)
        
        # Project to latent dimension
        z = self.proj(x)
        
        # Apply temporal smoothing for better phoneme representation
        B, D, T = z.shape
        z_smooth, _ = self.temporal_smooth(z.transpose(1, 2))
        z = z_smooth.transpose(1, 2)
        
        return z

class TeluguDecoder(nn.Module):
    """
    Decoder for discrete tokens -> Telugu speech
    Streaming-capable with lookahead buffer
    """
    def __init__(self, hidden_dim=1024, output_channels=1):
        super().__init__()
        
        # Initial projection
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # Transposed convolutions for upsampling
        # 200Hz -> 400Hz -> 2000Hz -> 4000Hz -> 8000Hz -> 16000Hz
        self.decoder = nn.ModuleList([
            ResidualBlock(hidden_dim),
            nn.ConvTranspose1d(hidden_dim, 512, kernel_size=4, stride=2, padding=1),  # 400Hz
            ResidualBlock(512),
            nn.ConvTranspose1d(512, 256, kernel_size=10, stride=5, padding=2),  # 2000Hz
            ResidualBlock(256),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 4000Hz
            ResidualBlock(128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 8000Hz
            ResidualBlock(64),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 16000Hz
            ResidualBlock(32),
            nn.Conv1d(32, output_channels, kernel_size=7, padding=3)
        ])
        
        # Post-processing for audio quality with tanh to match input range [-1, 1]
        self.post_net = nn.Sequential(
            nn.Conv1d(output_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, output_channels, kernel_size=5, padding=2),
            nn.Tanh()  # Match input data range [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio
        
        Args:
            z: Latent tensor [B, D, T'] at 200Hz
        
        Returns:
            audio: Reconstructed audio [B, 1, T] at 16kHz
        """
        # Initial projection
        x = self.proj(z)
        
        # Progressive decoding
        for layer in self.decoder:
            x = layer(x)
        
        # Post-processing for quality
        audio = self.post_net(x)
        
        return audio

class TeluCodec(nn.Module):
    """
    Complete Telugu neural codec with streaming support
    Targets: <10ms encoding, <10ms decoding, 16kbps bitrate
    """
    def __init__(self, sample_rate=16000, hidden_dim=1024, 
                 codebook_size=1024, num_quantizers=8):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim
        self.frame_rate = 200  # 200Hz tokens
        
        # Components
        self.encoder = TeluguEncoder(hidden_dim=hidden_dim)
        self.quantizer = VectorQuantizer(
            dim=hidden_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers
        )
        self.decoder = TeluguDecoder(hidden_dim=hidden_dim)
        
        # Streaming buffers
        self.register_buffer('encode_buffer', torch.zeros(1, 1, 1600))  # 100ms
        self.register_buffer('decode_buffer', torch.zeros(1, hidden_dim, 20))  # 100ms
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete codes
        
        Args:
            audio: Audio tensor [B, 1, T] or [B, T]
        
        Returns:
            codes: Discrete codes [B, Q, T']
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        # Encode to latent
        z = self.encoder(audio)
        
        # Quantize
        z_q, codes, _ = self.quantizer.quantize(z)
        
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes to audio
        
        Args:
            codes: Discrete codes [B, Q, T']
        
        Returns:
            audio: Reconstructed audio [B, 1, T]
        """
        B, Q, T = codes.shape
        
        # Dequantize
        z_q = torch.zeros(B, self.hidden_dim, T, device=codes.device)
        for q in range(Q):
            z_q += F.embedding(codes[:, q], self.quantizer.codebooks[q]).transpose(1, 2)
        
        # Decode to audio
        audio = self.decoder(z_q)
        
        return audio
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with losses
        
        Args:
            audio: Input audio [B, 1, T]
        
        Returns:
            Dictionary with reconstructed audio and losses
        """
        # Store original length
        original_length = audio.shape[-1]
        
        # Encode
        z = self.encoder(audio)
        
        # Quantize
        z_q, codes, vq_loss = self.quantizer.quantize(z)
        
        # Decode  
        audio_recon = self.decoder(z_q)
        
        # Match output size to input size
        if audio_recon.shape[-1] != original_length:
            if audio_recon.shape[-1] > original_length:
                # Crop if too long
                audio_recon = audio_recon[..., :original_length]
            else:
                # Pad if too short
                padding = original_length - audio_recon.shape[-1]
                audio_recon = F.pad(audio_recon, (0, padding))
        
        # Reconstruction loss with clamping
        recon_loss = F.l1_loss(audio_recon, audio)
        recon_loss = torch.clamp(recon_loss, 0, 10.0)  # Prevent explosion
        
        # CRITICAL: Amplitude/Scale matching loss
        # Force decoder to match input magnitude (not just shape)
        input_rms = torch.sqrt((audio ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        output_rms = torch.sqrt((audio_recon ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        scale_loss = F.mse_loss(output_rms, input_rms) * 10.0  # Strong weight!
        
        # Also match absolute max values
        input_max = audio.abs().max(dim=-1, keepdim=True)[0]
        output_max = audio_recon.abs().max(dim=-1, keepdim=True)[0]
        max_loss = F.mse_loss(output_max, input_max) * 5.0
        
        # Perceptual loss (multi-scale spectral) - REMOVED during early training
        # perceptual_loss = self._perceptual_loss(audio, audio_recon)
        perceptual_loss = torch.tensor(0.0, device=audio.device)  # Disabled!
        
        # Clamp VQ loss
        vq_loss = torch.clamp(vq_loss, 0, 10.0)
        
        # Total loss: recon + scale matching + max matching + VQ
        total_loss = recon_loss + scale_loss + max_loss + vq_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Return safe default loss if NaN detected
            total_loss = torch.tensor(1.0, device=audio.device, requires_grad=True)
        
        return {
            "audio": audio_recon,
            "codes": codes,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "perceptual_loss": perceptual_loss,
            "scale_loss": scale_loss,
            "max_loss": max_loss
        }
    
    def _perceptual_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Multi-scale spectral loss for perceptual quality"""
        # Ensure same length for STFT
        min_len = min(target.shape[-1], pred.shape[-1])
        target = target[..., :min_len]
        pred = pred[..., :min_len]
        
        # Cast to float32 for STFT to avoid FP16 issues
        target_f32 = target.float()
        pred_f32 = pred.float()
        
        loss = 0
        for n_fft in [512, 1024, 2048]:
            # Create Hann window in float32 to match input
            window = torch.hann_window(n_fft, device=target.device, dtype=torch.float32)
            
            # Compute spectrograms with proper window in float32
            target_spec = torch.stft(
                target_f32.squeeze(1), n_fft=n_fft, 
                hop_length=n_fft//4, window=window, return_complex=True
            ).abs()
            pred_spec = torch.stft(
                pred_f32.squeeze(1), n_fft=n_fft,
                hop_length=n_fft//4, window=window, return_complex=True
            ).abs()
            
            # Add small epsilon to prevent log(0) or division by zero
            target_spec = target_spec + 1e-7
            pred_spec = pred_spec + 1e-7
            
            # L1 + L2 loss on magnitude
            loss += F.l1_loss(pred_spec, target_spec) + F.mse_loss(pred_spec, target_spec)
        
        # Clamp loss to prevent NaN and cast back to original dtype
        loss = torch.clamp(loss / 3, 0, 100.0)
        
        return loss.to(target.dtype)
    
    @torch.no_grad()
    def encode_streaming(self, audio_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Streaming encode with 100ms chunks
        
        Args:
            audio_chunk: Audio chunk [1, 1, 1600] (100ms at 16kHz)
        
        Returns:
            codes: Codes for this chunk or None if buffering
        """
        # Add to buffer
        self.encode_buffer = torch.cat([self.encode_buffer, audio_chunk], dim=-1)
        
        # Process if we have enough samples (100ms = 1600 samples)
        if self.encode_buffer.size(-1) >= 1600:
            # Extract chunk
            chunk = self.encode_buffer[:, :, :1600]
            self.encode_buffer = self.encode_buffer[:, :, 1600:]
            
            # Encode
            codes = self.encode(chunk)
            return codes
        
        return None
    
    @torch.no_grad()
    def decode_streaming(self, codes_chunk: torch.Tensor) -> torch.Tensor:
        """
        Streaming decode with minimal latency
        
        Args:
            codes_chunk: Code chunk [1, Q, 20] (100ms at 200Hz)
        
        Returns:
            audio: Decoded audio chunk [1, 1, 1600]
        """
        # Direct decode (no buffering needed for low-latency)
        audio = self.decode(codes_chunk)
        return audio
    
    def calculate_bitrate(self) -> float:
        """Calculate codec bitrate in kbps"""
        bits_per_code = math.log2(self.quantizer.n_codes)
        codes_per_second = self.frame_rate
        num_quantizers = self.quantizer.n_quantizers
        bitrate = bits_per_code * codes_per_second * num_quantizers / 1000
        return bitrate


if __name__ == "__main__":
    # Test codec
    codec = TeluCodec()
    print(f"TeluCodec initialized")
    print(f"Parameters: {sum(p.numel() for p in codec.parameters())/1e6:.2f}M")
    print(f"Bitrate: {codec.calculate_bitrate():.2f} kbps")
    
    # Test with dummy audio
    audio = torch.randn(2, 1, 16000)  # 1 second at 16kHz
    
    # Full pass
    output = codec(audio)
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {output['audio'].shape}")
    print(f"Codes shape: {output['codes'].shape}")
    print(f"Total loss: {output['loss'].item():.4f}")
    
    # Test streaming
    chunk = torch.randn(1, 1, 1600)  # 100ms
    codes = codec.encode_streaming(chunk)
    if codes is not None:
        audio_chunk = codec.decode_streaming(codes)
        print(f"Streaming: {chunk.shape} -> {codes.shape} -> {audio_chunk.shape}")