"""
Multi-Scale Discriminator for Neural Audio Codec
Based on DAC (Descript Audio Codec) and EnCodec architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DiscriminatorBlock(nn.Module):
    """Single discriminator with strided convolutions"""
    def __init__(self):
        super().__init__()
        
        # Use grouped convolutions for efficiency (like DAC)
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
        ])
        
        # Final layer outputs logits
        self.final_conv = nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        
        # LeakyReLU activation
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with feature extraction
        
        Args:
            x: Input audio (batch, 1, time)
            
        Returns:
            logits: Real/fake logits (batch, 1, time)
            features: List of intermediate feature maps for feature matching
        """
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            features.append(x)
        
        # Final logits (no activation)
        logits = self.final_conv(x)
        
        return logits, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator operating at different resolutions
    Based on DAC and EnCodec multi-scale architectures
    """
    def __init__(self, num_scales=3):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList([
            DiscriminatorBlock() for _ in range(num_scales)
        ])
        
        # Average pooling for downsampling (scale 2, 4, etc.)
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through all scales
        
        Args:
            x: Input audio (batch, 1, time)
            
        Returns:
            all_logits: List of logits from each scale
            all_features: List of feature lists from each scale
        """
        all_logits = []
        all_features = []
        
        for i, disc in enumerate(self.discriminators):
            # Apply discriminator at current scale
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
            
            # Downsample for next scale (except last)
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        return all_logits, all_features


def discriminator_loss(
    real_logits_list: List[torch.Tensor],
    fake_logits_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute discriminator loss (hinge loss)
    
    Args:
        real_logits_list: List of logits from real audio at each scale
        fake_logits_list: List of logits from fake audio at each scale
        
    Returns:
        Total discriminator loss
    """
    loss = 0.0
    
    for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
        # Hinge loss for discriminator
        # Real: max(0, 1 - real_logits)
        # Fake: max(0, 1 + fake_logits)
        real_loss = torch.mean(torch.relu(1.0 - real_logits))
        fake_loss = torch.mean(torch.relu(1.0 + fake_logits))
        loss += real_loss + fake_loss
    
    return loss / len(real_logits_list)


def generator_adversarial_loss(
    fake_logits_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute generator adversarial loss (fool discriminator)
    
    Args:
        fake_logits_list: List of logits from fake audio at each scale
        
    Returns:
        Generator adversarial loss
    """
    loss = 0.0
    
    for fake_logits in fake_logits_list:
        # Generator wants discriminator to output high values for fake
        # Hinge loss: max(0, 1 - fake_logits) → minimize by making fake_logits large
        loss += torch.mean(torch.relu(1.0 - fake_logits))
    
    return loss / len(fake_logits_list)


def feature_matching_loss(
    real_features_list: List[List[torch.Tensor]],
    fake_features_list: List[List[torch.Tensor]]
) -> torch.Tensor:
    """
    Compute feature matching loss (L1 distance between features)
    
    Args:
        real_features_list: List of feature lists from real audio
        fake_features_list: List of feature lists from fake audio
        
    Returns:
        Feature matching loss
    """
    loss = 0.0
    num_features = 0
    
    for real_features, fake_features in zip(real_features_list, fake_features_list):
        for real_feat, fake_feat in zip(real_features, fake_features):
            # L1 distance between features
            loss += F.l1_loss(fake_feat, real_feat.detach())
            num_features += 1
    
    return loss / num_features


if __name__ == "__main__":
    # Test discriminator
    print("Testing Multi-Scale Discriminator...")
    
    disc = MultiScaleDiscriminator(num_scales=3)
    
    # Test input
    batch_size = 4
    audio_length = 16000  # 1 second at 16kHz
    x = torch.randn(batch_size, 1, audio_length)
    
    # Forward pass
    logits_list, features_list = disc(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"\nNumber of scales: {len(logits_list)}")
    
    for i, (logits, features) in enumerate(zip(logits_list, features_list)):
        print(f"\nScale {i}:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Number of feature maps: {len(features)}")
        for j, feat in enumerate(features):
            print(f"    Feature {j} shape: {feat.shape}")
    
    # Test losses
    print("\n" + "="*50)
    print("Testing Loss Functions...")
    
    # Create fake audio
    fake_x = torch.randn_like(x)
    fake_logits_list, fake_features_list = disc(fake_x)
    
    # Discriminator loss
    disc_loss = discriminator_loss(logits_list, fake_logits_list)
    print(f"\nDiscriminator loss: {disc_loss.item():.4f}")
    
    # Generator adversarial loss
    gen_adv_loss = generator_adversarial_loss(fake_logits_list)
    print(f"Generator adversarial loss: {gen_adv_loss.item():.4f}")
    
    # Feature matching loss
    feat_loss = feature_matching_loss(features_list, fake_features_list)
    print(f"Feature matching loss: {feat_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in disc.parameters())
    trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ All tests passed!")
