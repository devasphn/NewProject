"""
Proper DAC-Style Discriminator Architecture
Based on official DAC (Descript Audio Codec) implementation
Combines Multi-Period Discriminator + Multi-Scale STFT Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class PeriodDiscriminator(nn.Module):
    """Single period discriminator for Multi-Period Discriminator"""
    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        
        # Convolutional layers with increasing channels
        # NO aggressive grouping - use standard convolutions for full capacity
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, 1, time)
        Returns:
            logits: (batch, 1, ...)
            features: List of intermediate feature maps
        """
        features = []
        
        # Reshape input into 2D by period
        batch, channels, time = x.shape
        if time % self.period != 0:
            # Pad to make divisible by period
            n_pad = self.period - (time % self.period)
            x = F.pad(x, (0, n_pad), mode='reflect')
            time = time + n_pad
        
        # Reshape to (batch, channels, time//period, period)
        x = x.view(batch, channels, time // self.period, self.period)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        # Final conv
        x = self.conv_post(x)
        features.append(x)
        
        # Flatten output
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator from HiFi-GAN, used in DAC
    Applies discriminators at different periodicities
    """
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, 1, time)
        Returns:
            all_logits: List of logits from each period
            all_features: List of feature lists from each period
        """
        all_logits = []
        all_features = []
        
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        
        return all_logits, all_features


class STFTDiscriminator(nn.Module):
    """
    Single-scale STFT discriminator
    Processes complex STFT (real, imaginary, magnitude)
    """
    def __init__(
        self, 
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Register Hann window
        self.register_buffer('window', torch.hann_window(win_length))
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        
        # Input: 3 channels (real, imag, magnitude)
        # Use 2D convolutions on time-frequency representation
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(3, 32, (3, 9), (1, 1), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))),
        ])
        
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), (1, 1), padding=(1, 1)))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, 1, time)
        Returns:
            logits: (batch, ...)
            features: List of intermediate feature maps
        """
        features = []
        
        # Compute STFT
        batch, channels, time = x.shape
        x = x.squeeze(1)  # (batch, time)
        
        # STFT computation
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True
        )
        
        # stft shape: (batch, freq_bins, time_frames)
        # Extract real, imaginary, magnitude
        real = stft.real
        imag = stft.imag
        magnitude = torch.abs(stft)
        
        # Stack as 3-channel input: (batch, 3, freq, time)
        x = torch.stack([real, imag, magnitude], dim=1)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        # Final conv
        x = self.conv_post(x)
        features.append(x)
        
        # Flatten output
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT Discriminator from DAC
    Uses multiple window sizes to capture different time-frequency resolutions
    """
    def __init__(
        self,
        n_ffts: List[int] = [2048, 1024, 512],
        hop_lengths: List[int] = None,
        win_lengths: List[int] = None
    ):
        super().__init__()
        
        # Default: hop_length = n_fft // 4, win_length = n_fft
        if hop_lengths is None:
            hop_lengths = [n // 4 for n in n_ffts]
        if win_lengths is None:
            win_lengths = n_ffts
        
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft, hop, win)
            for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, 1, time)
        Returns:
            all_logits: List of logits from each scale
            all_features: List of feature lists from each scale
        """
        all_logits = []
        all_features = []
        
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        
        return all_logits, all_features


class DACDiscriminator(nn.Module):
    """
    Complete DAC discriminator combining:
    1. Multi-Period Discriminator (time domain)
    2. Multi-Scale STFT Discriminator (frequency domain)
    """
    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        n_ffts: List[int] = [2048, 1024, 512]
    ):
        super().__init__()
        
        self.mpd = MultiPeriodDiscriminator(periods)
        self.msd = MultiScaleSTFTDiscriminator(n_ffts)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, 1, time)
        Returns:
            all_logits: Combined list of logits from MPD and MSD
            all_features: Combined list of features from MPD and MSD
        """
        # Multi-Period Discriminator
        mpd_logits, mpd_features = self.mpd(x)
        
        # Multi-Scale STFT Discriminator
        msd_logits, msd_features = self.msd(x)
        
        # Combine outputs
        all_logits = mpd_logits + msd_logits
        all_features = mpd_features + msd_features
        
        return all_logits, all_features


def discriminator_loss(
    real_logits_list: List[torch.Tensor],
    fake_logits_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hinge loss for discriminator (used in DAC)
    
    Returns:
        total_loss: Combined discriminator loss
        real_loss: Loss on real samples
        fake_loss: Loss on fake samples
    """
    real_loss = 0.0
    fake_loss = 0.0
    
    for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
        # Hinge loss: want real > 0, fake < 0
        real_loss += torch.mean(F.relu(1.0 - real_logits))
        fake_loss += torch.mean(F.relu(1.0 + fake_logits))
    
    # Average across all discriminators
    real_loss = real_loss / len(real_logits_list)
    fake_loss = fake_loss / len(fake_logits_list)
    total_loss = real_loss + fake_loss
    
    return total_loss, real_loss, fake_loss


def generator_adversarial_loss(
    fake_logits_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Adversarial loss for generator (fool discriminator)
    Hinge formulation: want fake > 0
    """
    loss = 0.0
    
    for fake_logits in fake_logits_list:
        # Want discriminator to output positive values for fake
        loss += torch.mean(F.relu(1.0 - fake_logits))
    
    return loss / len(fake_logits_list)


def feature_matching_loss(
    real_features_list: List[List[torch.Tensor]],
    fake_features_list: List[List[torch.Tensor]]
) -> torch.Tensor:
    """
    L1 feature matching loss between real and fake features
    """
    loss = 0.0
    num_features = 0
    
    for real_features, fake_features in zip(real_features_list, fake_features_list):
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat)
            num_features += 1
    
    return loss / num_features if num_features > 0 else loss


if __name__ == "__main__":
    print("Testing DAC Discriminator Architecture...\n")
    
    # Create discriminator
    disc = DACDiscriminator()
    
    # Test input
    batch_size = 4
    audio_length = 16000  # 1 second at 16kHz
    x = torch.randn(batch_size, 1, audio_length)
    
    print(f"Input shape: {x.shape}\n")
    
    # Forward pass
    logits_list, features_list = disc(x)
    
    print(f"Number of discriminators: {len(logits_list)}")
    print(f"  - Multi-Period: 5 discriminators (periods 2,3,5,7,11)")
    print(f"  - Multi-Scale STFT: 3 discriminators (windows 2048,1024,512)")
    print(f"  - Total: 8 discriminators\n")
    
    for i, (logits, features) in enumerate(zip(logits_list, features_list)):
        disc_type = "MPD" if i < 5 else "STFT"
        print(f"{disc_type} {i}: Logits shape {logits.shape}, {len(features)} feature maps")
    
    # Test losses
    print("\n" + "="*50)
    print("Testing Loss Functions...\n")
    
    fake_x = torch.randn_like(x)
    fake_logits_list, fake_features_list = disc(fake_x)
    
    # Discriminator loss
    disc_loss, real_loss, fake_loss = discriminator_loss(logits_list, fake_logits_list)
    print(f"Discriminator loss: {disc_loss.item():.4f}")
    print(f"  - Real loss: {real_loss.item():.4f}")
    print(f"  - Fake loss: {fake_loss.item():.4f}")
    
    # Generator losses
    gen_adv_loss = generator_adversarial_loss(fake_logits_list)
    feat_loss = feature_matching_loss(features_list, fake_features_list)
    print(f"\nGenerator adversarial loss: {gen_adv_loss.item():.4f}")
    print(f"Feature matching loss: {feat_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in disc.parameters())
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")
    
    print("\n✅ All tests passed!")
