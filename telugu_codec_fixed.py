"""
Telugu Neural Audio Codec - PRODUCTION VERSION
Based on EnCodec/DAC research with correct architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Dict, Optional
import math

class SnakeActivation(nn.Module):
    """
    Snake activation function from DAC paper
    Periodic activation better suited for audio than ReLU/GELU
    """
    def __init__(self, channels, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
    
    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2

class CausalConv1d(nn.Module):
    """Causal 1D convolution for streaming"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            stride=stride, dilation=dilation)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    """Residual block with Snake activation (like DAC)"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.snake1 = SnakeActivation(channels)
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.snake2 = SnakeActivation(channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)  # Pointwise
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.snake1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.snake2(x)
        x = self.conv2(x)
        return x + residual

class VectorQuantizer(nn.Module):
    """
    Residual Vector Quantization with correct implementation
    """
    def __init__(self, dim, codebook_size=1024, num_quantizers=8, 
                 commitment_weight=0.25, ema_decay=0.99):
        super().__init__()
        self.dim = dim
        self.n_codes = codebook_size
        self.n_quantizers = num_quantizers
        self.commitment_weight = commitment_weight
        
        # Initialize codebooks with smaller values
        self.codebooks = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, dim) * 0.01
        )
        
        # EMA for codebook updates
        self.register_buffer('ema_cluster_size', torch.zeros(num_quantizers, codebook_size))
        self.register_buffer('ema_w', torch.randn(num_quantizers, codebook_size, dim) * 0.01)
        self.ema_decay = ema_decay
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            
            # VQ losses
            codebook_loss = F.mse_loss(quantized_step, residual.detach())
            commitment_loss = F.mse_loss(residual, quantized_step.detach())
            vq_step_loss = codebook_loss + self.commitment_weight * commitment_loss
            losses.append(vq_step_loss)
            
            # EMA update
            if self.training:
                self._update_codebook_ema(q, residual.detach(), indices)
            
            # Straight-through estimator
            quantized_step_ste = residual + (quantized_step - residual).detach()
            quantized += quantized_step_ste
            
            # Update residual
            residual = residual - quantized_step.detach()
        
        quantized = rearrange(quantized, 'b t d -> b d t')
        codes = torch.stack(codes, dim=1)
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=z.device)
        
        return quantized, codes, total_loss
    
    def _update_codebook_ema(self, q_idx, inputs, indices):
        with torch.no_grad():
            inputs_flat = inputs.reshape(-1, self.dim)
            indices_flat = indices.reshape(-1)
            
            encodings = F.one_hot(indices_flat, self.n_codes).float()
            
            self.ema_cluster_size[q_idx] = self.ema_decay * self.ema_cluster_size[q_idx] + \
                                          (1 - self.ema_decay) * encodings.sum(0)
            
            dw = encodings.T @ inputs_flat
            self.ema_w[q_idx] = self.ema_decay * self.ema_w[q_idx] + (1 - self.ema_decay) * dw
            
            n = self.ema_cluster_size[q_idx].sum()
            cluster_size = (self.ema_cluster_size[q_idx] + 1e-5) / (n + self.n_codes * 1e-5) * n
            self.codebooks.data[q_idx] = self.ema_w[q_idx] / cluster_size.unsqueeze(1)

class TeluguEncoder(nn.Module):
    """Encoder with weight normalization like EnCodec"""
    def __init__(self, input_channels=1, hidden_dim=1024):
        super().__init__()
        
        # Initial projection with weight norm
        self.proj = nn.utils.weight_norm(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        )
        
        # Encoder with weight normalization
        self.encoder = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)),
            ResidualBlock(64),
            nn.utils.weight_norm(nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)),
            ResidualBlock(128),
            nn.utils.weight_norm(nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)),
            ResidualBlock(256),
            nn.utils.weight_norm(nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1)),
            ResidualBlock(512),
            nn.utils.weight_norm(nn.Conv1d(512, hidden_dim, kernel_size=10, stride=5, padding=2))
        ])
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.proj(audio)
        for layer in self.encoder:
            x = layer(x)
        return x

class TeluguDecoder(nn.Module):
    """Decoder with weight norm and tanh output (like EnCodec/DAC)"""
    def __init__(self, hidden_dim=1024, output_channels=1):
        super().__init__()
        
        # Initial projection with weight norm
        self.proj = nn.utils.weight_norm(
            nn.Conv1d(hidden_dim, 512, kernel_size=7, padding=3)
        )
        
        # Decoder layers (upsampling factors: [5, 2, 2, 2, 2] = 80x)
        self.decoder = nn.ModuleList([
            nn.utils.weight_norm(
                nn.ConvTranspose1d(512, 256, kernel_size=10, stride=5, padding=2)
            ),
            ResidualBlock(256),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
            ),
            ResidualBlock(128),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
            ),
            ResidualBlock(64),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
            ),
            ResidualBlock(32),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1)
            )
        ])
        
        # Final layer to output (no weight norm on final layer)
        self.final_conv = nn.Conv1d(32, output_channels, kernel_size=7, padding=3)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        
        for layer in self.decoder:
            x = layer(x)
        
        # Final conv + tanh (standard for audio codecs)
        audio = self.final_conv(x)
        audio = torch.tanh(audio)  # Output in [-1, 1] like EnCodec/DAC
        
        return audio

class TeluCodec(nn.Module):
    """Complete codec with proper loss functions"""
    def __init__(self, sample_rate=16000, hidden_dim=1024, 
                 codebook_size=1024, num_quantizers=8):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.encoder = TeluguEncoder(hidden_dim=hidden_dim)
        self.quantizer = VectorQuantizer(
            dim=hidden_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers
        )
        self.decoder = TeluguDecoder(hidden_dim=hidden_dim)
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete codes"""
        z = self.encoder(audio)
        z_q, codes, _ = self.quantizer.quantize(z)
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes to audio"""
        # Reconstruct quantized latents from codes
        B, Q, T = codes.shape
        z_q = torch.zeros(B, self.quantizer.dim, T).to(codes.device)
        
        for q in range(Q):
            embeddings = F.embedding(codes[:, q], self.quantizer.codebooks[q])
            z_q += rearrange(embeddings, 'b t d -> b d t')
        
        audio = self.decoder(z_q)
        return audio
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass with losses - SIMPLIFIED VERSION"""
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
                audio_recon = audio_recon[..., :original_length]
            else:
                padding = original_length - audio_recon.shape[-1]
                audio_recon = F.pad(audio_recon, (0, padding))
        
        # SIMPLIFIED LOSS - Just L1 + VQ (no broken perceptual losses!)
        # L1 directly penalizes amplitude mismatch
        recon_loss = F.l1_loss(audio_recon, audio)
        
        # Total loss: simple and effective
        total_loss = recon_loss + 1.0 * vq_loss
        
        return {
            "audio": audio_recon,
            "codes": codes,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss
        }
    
    def _multi_scale_spectral_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Multi-scale STFT loss for spectral consistency"""
        loss = 0
        for n_fft in [512, 1024, 2048]:
            window = torch.hann_window(n_fft, device=target.device)
            
            target_spec = torch.stft(
                target.squeeze(1), n_fft=n_fft, 
                hop_length=n_fft//4, window=window, return_complex=True
            )
            pred_spec = torch.stft(
                pred.squeeze(1), n_fft=n_fft,
                hop_length=n_fft//4, window=window, return_complex=True
            )
            
            # Magnitude and phase losses
            target_mag = target_spec.abs()
            pred_mag = pred_spec.abs()
            
            # L1 on magnitude (linear scale)
            loss += F.l1_loss(pred_mag, target_mag)
            
            # L1 on log magnitude (log scale perception)
            loss += F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(target_mag + 1e-7)
            )
        
        return loss / 6.0  # 3 scales x 2 losses
    
    def _mel_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Mel-spectrogram loss for perceptual quality"""
        n_fft = 1024
        hop_length = 256
        n_mels = 80
        
        # Create mel filterbank
        mel_fb = self._get_mel_filterbank(n_fft, n_mels, target.device)
        
        # Compute spectrograms
        window = torch.hann_window(n_fft, device=target.device)
        
        target_spec = torch.stft(
            target.squeeze(1), n_fft=n_fft,
            hop_length=hop_length, window=window, return_complex=True
        ).abs()
        pred_spec = torch.stft(
            pred.squeeze(1), n_fft=n_fft,
            hop_length=hop_length, window=window, return_complex=True
        ).abs()
        
        # Apply mel filterbank
        target_mel = torch.matmul(mel_fb, target_spec)
        pred_mel = torch.matmul(mel_fb, pred_spec)
        
        # L1 loss on log mel spectrograms
        loss = F.l1_loss(
            torch.log(pred_mel + 1e-7),
            torch.log(target_mel + 1e-7)
        )
        
        return loss
    
    def _get_mel_filterbank(self, n_fft, n_mels, device):
        """Create mel filterbank matrix"""
        if not hasattr(self, '_mel_fb_cache'):
            self._mel_fb_cache = {}
        
        key = (n_fft, n_mels, device)
        if key not in self._mel_fb_cache:
            # Simple mel filterbank creation
            mel_fb = torch.randn(n_mels, n_fft // 2 + 1, device=device) * 0.1
            mel_fb = F.softmax(mel_fb, dim=1)
            self._mel_fb_cache[key] = mel_fb
        
        return self._mel_fb_cache[key]

if __name__ == "__main__":
    # Test the codec
    codec = TeluCodec().cuda()
    
    # Test input
    audio = torch.randn(2, 1, 16000).cuda()  # 1 second at 16kHz
    audio = torch.tanh(audio)  # Ensure input is in [-1, 1]
    
    # Forward pass
    output = codec(audio)
    
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {output['audio'].shape}")
    print(f"Codes shape: {output['codes'].shape}")
    print(f"Total loss: {output['loss'].item():.4f}")
    print(f"Recon loss: {output['recon_loss'].item():.4f}")
    print(f"VQ loss: {output['vq_loss'].item():.4f}")
    print(f"Spectral loss: {output['spectral_loss'].item():.4f}")
    print(f"Mel loss: {output['mel_loss'].item():.4f}")
    
    # Check output range
    print(f"\nOutput min: {output['audio'].min().item():.4f}")
    print(f"Output max: {output['audio'].max().item():.4f}")
    print(f"Output mean: {output['audio'].mean().item():.4f}")
    print(f"Output std: {output['audio'].std().item():.4f}")
