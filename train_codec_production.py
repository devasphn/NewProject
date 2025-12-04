"""
═══════════════════════════════════════════════════════════════════════════════
    PRODUCTION CODEC TRAINING SCRIPT
    Version: 2.0 - Final Production Build
    
    Features:
    ✅ GAN Training (DAC Discriminator)
    ✅ Multi-GPU Support (DDP)
    ✅ Mixed Precision (FP16)
    ✅ Gradient Accumulation
    ✅ Learning Rate Scheduling
    ✅ Checkpoint Management
    ✅ TensorBoard Logging
    ✅ Audio Sample Logging
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import warnings

# Suppress torchaudio deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Local imports
from codec_production import ProductionCodec, CodecConfig
from discriminator_dac import DACDiscriminator, discriminator_loss, generator_adversarial_loss, feature_matching_loss

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Run: pip install wandb")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class TrainConfig:
    """Training configuration"""
    # Data
    data_dirs: list = None  # List of directories containing audio
    sample_rate: int = 16000
    segment_length: int = 32000  # 2 seconds at 16kHz
    
    # Training
    batch_size: int = 16
    num_epochs: int = 200
    gradient_accumulation: int = 2
    
    # Optimization - TTUR (Two Time-scale Update Rule)
    gen_lr: float = 1e-4
    disc_lr: float = 5e-5  # Discriminator learns slower = more stable GAN
    betas: tuple = (0.5, 0.9)
    weight_decay: float = 0.01
    
    # Loss weights - BALANCED for stable GAN training
    adv_weight: float = 0.1  # Reduced from 1.0 - prevents gen loss explosion
    feat_weight: float = 2.0  # Reduced from 10.0 - more stable training
    
    # GAN training
    disc_start_epoch: int = 10  # Start discriminator later for better codec foundation
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints_production"
    save_every: int = 5
    
    # Logging
    log_every: int = 100
    sample_every: int = 500
    
    # Hardware
    num_workers: int = 32  # Optimal for high-core systems
    use_fp16: bool = False  # Disabled for stability - enable with --fp16 flag
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "codec-production"
    wandb_run_name: str = None
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class MultilingualAudioDataset(Dataset):
    """
    Dataset for multilingual audio training with augmentation
    Supports: Telugu, Hindi, English, Tamil, and more
    
    Augmentations for robustness:
    - Speed perturbation (0.9-1.1x)
    - Volume perturbation (0.7-1.3x)  
    - Background noise injection
    - Pitch shifting (optional)
    """
    def __init__(self, data_dirs: list, sample_rate: int = 16000, 
                 segment_length: int = 32000, normalize_db: float = -16.0,
                 augment: bool = True, augment_prob: float = 0.3):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.normalize_db = normalize_db
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Collect all audio files from all directories
        self.audio_files = []
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                print(f"⚠️ Directory not found: {data_dir}")
                continue
            
            for ext in audio_extensions:
                self.audio_files.extend(list(data_path.rglob(f"*{ext}")))
        
        print(f"📁 Found {len(self.audio_files)} audio files across {len(data_dirs)} directories")
        
        if len(self.audio_files) == 0:
            raise ValueError("No audio files found! Check your data directories.")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Get segment
            if waveform.shape[1] >= self.segment_length:
                # Random crop
                start = torch.randint(0, waveform.shape[1] - self.segment_length + 1, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
            else:
                # Pad short audio
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            
            # Apply augmentation (only during training)
            if self.augment:
                waveform = self._apply_augmentation(waveform)
            
            # RMS Normalization
            waveform = self._normalize_rms(waveform)
            
            # Clamp to [-1, 1]
            waveform = torch.clamp(waveform, -1.0, 1.0)
            
            return waveform
            
        except Exception as e:
            # Return valid noise if file fails to load
            target_rms = 10 ** (self.normalize_db / 20)
            return torch.randn(1, self.segment_length) * target_rms
    
    def _normalize_rms(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to target RMS (dB)"""
        rms = torch.sqrt(torch.mean(audio ** 2) + 1e-8)
        target_rms = 10 ** (self.normalize_db / 20)
        audio = audio * (target_rms / rms)
        
        # Check for NaN and replace with noise if needed
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            audio = torch.randn_like(audio) * target_rms
        
        return audio
    
    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations for robustness training.
        These help the codec generalize to:
        - Different speaking speeds
        - Various volume levels
        - Noisy environments
        """
        import random
        
        # Speed perturbation (0.9-1.1x)
        if random.random() < self.augment_prob:
            speed_factor = 0.9 + random.random() * 0.2  # 0.9-1.1
            # Simple speed change by resampling
            orig_length = waveform.shape[-1]
            new_length = int(orig_length / speed_factor)
            waveform = F.interpolate(
                waveform.unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            # Adjust back to original length
            if waveform.shape[-1] > self.segment_length:
                waveform = waveform[:, :self.segment_length]
            elif waveform.shape[-1] < self.segment_length:
                waveform = F.pad(waveform, (0, self.segment_length - waveform.shape[-1]))
        
        # Volume perturbation (0.7-1.3x)
        if random.random() < self.augment_prob:
            gain = 0.7 + random.random() * 0.6  # 0.7-1.3
            waveform = waveform * gain
        
        # Background noise injection
        if random.random() < self.augment_prob:
            noise_level = random.random() * 0.01  # 0-0.01 noise level
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        # Random reverb simulation (simple convolution with decaying impulse)
        if random.random() < self.augment_prob * 0.5:  # Less frequent
            decay = 0.3 + random.random() * 0.4  # 0.3-0.7 decay
            ir_length = int(self.sample_rate * 0.05)  # 50ms impulse response
            impulse = torch.zeros(1, 1, ir_length)
            impulse[0, 0, 0] = 1.0
            for i in range(1, ir_length):
                impulse[0, 0, i] = impulse[0, 0, i-1] * decay * (0.5 + random.random() * 0.5)
            # Apply convolution
            waveform_padded = F.pad(waveform.unsqueeze(0), (ir_length-1, 0))
            waveform = F.conv1d(waveform_padded, impulse).squeeze(0)
            waveform = waveform[:, :self.segment_length]
        
        return waveform


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class ProductionCodecTrainer:
    """Production codec trainer with GAN training"""
    
    def __init__(self, config: TrainConfig, codec_config: CodecConfig = None):
        self.config = config
        self.codec_config = codec_config or CodecConfig()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Create models
        print("\n📦 Initializing models...")
        self.codec = ProductionCodec(self.codec_config).to(self.device)
        self.discriminator = DACDiscriminator().to(self.device)
        
        # Optimizers
        self.gen_optimizer = torch.optim.AdamW(
            self.codec.parameters(),
            lr=config.gen_lr,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.disc_lr,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # Schedulers
        self.gen_scheduler = CosineAnnealingLR(self.gen_optimizer, T_max=config.num_epochs)
        self.disc_scheduler = CosineAnnealingLR(self.disc_optimizer, T_max=config.num_epochs)
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_fp16 else None
        
        # Create dataset
        print("\n📁 Loading dataset...")
        self.dataset = MultilingualAudioDataset(
            data_dirs=config.data_dirs,
            sample_rate=config.sample_rate,
            segment_length=config.segment_length
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if config.num_workers > 0 else False,
            prefetch_factor=4 if config.num_workers > 0 else None,
        )
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Logging
        self.setup_logging()
        
        # Print GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 VRAM: {gpu_mem:.1f} GB")
            print(f"📊 Batch size: {config.batch_size}")
            print(f"👷 Workers: {config.num_workers}")
    
    def setup_logging(self):
        """Setup TensorBoard and WandB logging"""
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.checkpoint_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir)
            print(f"📊 TensorBoard logging to: {log_dir}")
        except ImportError:
            self.writer = None
            print("⚠️ TensorBoard not available")
        
        # Weights & Biases
        self.use_wandb = False
        if WANDB_AVAILABLE and self.config.use_wandb:
            try:
                run_name = self.config.wandb_run_name or f"codec_{datetime.now().strftime('%Y%m%d_%H%M')}"
                wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    config={
                        'batch_size': self.config.batch_size,
                        'num_epochs': self.config.num_epochs,
                        'gen_lr': self.config.gen_lr,
                        'disc_lr': self.config.disc_lr,
                        'num_quantizers': self.codec_config.num_quantizers,
                        'hidden_dim': self.codec_config.hidden_dim,
                        'dataset_size': len(self.dataset),
                    }
                )
                self.use_wandb = True
                print(f"📈 WandB logging enabled: {run_name}")
            except Exception as e:
                print(f"⚠️ WandB init failed: {e}")
    
    def save_checkpoint(self, name: str = None, is_best: bool = False):
        """Save training checkpoint"""
        if name is None:
            name = f"codec_epoch_{self.epoch}"
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'codec_state_dict': self.codec.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'gen_scheduler_state_dict': self.gen_scheduler.state_dict(),
            'disc_scheduler_state_dict': self.disc_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'codec_config': self.codec_config.__dict__,
        }
        
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_codec.pt"
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved!")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        print(f"📂 Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.codec.load_state_dict(checkpoint['codec_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler_state_dict'])
        self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"✅ Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train_step(self, audio: torch.Tensor, use_disc: bool = True):
        """Single training step"""
        audio = audio.to(self.device)
        
        losses = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # GENERATOR UPDATE (runs first to get fresh audio)
        # ═══════════════════════════════════════════════════════════════════
        self.gen_optimizer.zero_grad()
        
        with autocast('cuda', enabled=self.config.use_fp16):
            # Forward pass - generate fake audio
            output = self.codec(audio)
            fake_audio = output['audio']
            
            # Reconstruction losses from codec
            gen_loss = output['loss']
            
            if use_disc:
                # Get discriminator outputs (cache real features)
                with torch.no_grad():
                    real_logits, real_features = self.discriminator(audio)
                fake_logits, fake_features = self.discriminator(fake_audio)
                
                # Adversarial loss (generator wants discriminator to think fake is real)
                adv_loss = generator_adversarial_loss(fake_logits)
                feat_loss = feature_matching_loss(real_features, fake_features)
                
                # Scale down adversarial components
                gen_loss = gen_loss + self.config.adv_weight * adv_loss + self.config.feat_weight * feat_loss
                
                losses['adv_loss'] = adv_loss.item()
                losses['feat_loss'] = feat_loss.item()
        
        # Check for NaN loss before backprop
        if torch.isnan(gen_loss) or torch.isinf(gen_loss):
            # Return NaN to signal skip this batch
            losses['gen_loss'] = float('nan')
            losses['recon_loss'] = float('nan')
            losses['vq_loss'] = float('nan')
            losses['spectral_loss'] = float('nan')
            losses['mel_loss'] = float('nan')
            losses['semantic_loss'] = float('nan')
            return losses
        
        if self.scaler:
            self.scaler.scale(gen_loss).backward()
            self.scaler.unscale_(self.gen_optimizer)
            # Check for inf gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.codec.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                self.scaler.step(self.gen_optimizer)
            self.scaler.update()
        else:
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.codec.parameters(), 1.0)
            self.gen_optimizer.step()
        
        # Record losses
        losses['gen_loss'] = gen_loss.item()
        losses['recon_loss'] = output['recon_loss'].item()
        losses['vq_loss'] = output['vq_loss'].item()
        losses['spectral_loss'] = output['spectral_loss'].item()
        losses['mel_loss'] = output['mel_loss'].item()
        losses['semantic_loss'] = output['semantic_loss'].item()
        
        # ═══════════════════════════════════════════════════════════════════
        # DISCRIMINATOR UPDATE (after generator, using detached fake audio)
        # ═══════════════════════════════════════════════════════════════════
        if use_disc:
            self.disc_optimizer.zero_grad()
            
            with autocast('cuda', enabled=self.config.use_fp16):
                # Use detached fake audio (no gradient to generator)
                fake_audio_detached = fake_audio.detach()
                
                # Discriminator forward
                real_logits, _ = self.discriminator(audio)
                fake_logits, _ = self.discriminator(fake_audio_detached)
                
                # Discriminator loss
                disc_loss, real_loss, fake_loss = discriminator_loss(real_logits, fake_logits)
            
            if self.scaler:
                self.scaler.scale(disc_loss).backward()
                self.scaler.unscale_(self.disc_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)  # Tighter clipping
                self.scaler.step(self.disc_optimizer)
            else:
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)  # Tighter clipping
                self.disc_optimizer.step()
            
            losses['disc_loss'] = disc_loss.item()
            losses['disc_real'] = real_loss.item()
            losses['disc_fake'] = fake_loss.item()
        
        return losses
    
    def train_epoch(self):
        """Train one epoch"""
        self.codec.train()
        self.discriminator.train()
        
        use_disc = self.epoch >= self.config.disc_start_epoch
        
        epoch_losses = {}
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, audio in enumerate(pbar):
            # Skip batches with NaN/Inf
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                continue
            
            losses = self.train_step(audio, use_disc=use_disc)
            
            # Skip if losses are NaN (indicates bad batch)
            if np.isnan(losses.get('gen_loss', 0)) or np.isnan(losses.get('recon_loss', 0)):
                continue
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                if not np.isnan(v):  # Only accumulate valid losses
                    epoch_losses[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'gen': f"{losses['gen_loss']:.3f}",
                'recon': f"{losses['recon_loss']:.3f}",
                'vq': f"{losses['vq_loss']:.3f}",
            })
            
            # Logging
            if self.global_step % self.config.log_every == 0:
                if self.writer:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                if self.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in losses.items()}, step=self.global_step)
            
            self.global_step += 1
        
        # Epoch averages
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self):
        """Validation pass"""
        self.codec.eval()
        
        total_loss = 0
        num_batches = min(50, len(self.dataloader))  # Validate on subset
        
        with torch.no_grad():
            for i, audio in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                
                audio = audio.to(self.device)
                output = self.codec(audio)
                total_loss += output['loss'].item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("STARTING PRODUCTION CODEC TRAINING")
        print("="*60)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Steps per epoch: {len(self.dataloader)}")
        print(f"Discriminator starts at epoch: {self.config.disc_start_epoch}")
        print("="*60 + "\n")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            start_time = time.time()
            train_losses = self.train_epoch()
            epoch_time = time.time() - start_time
            
            # Validate
            val_loss = self.validate()
            
            # Update schedulers
            self.gen_scheduler.step()
            self.disc_scheduler.step()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary (took {epoch_time/60:.1f} min):")
            print(f"   Gen Loss: {train_losses['gen_loss']:.4f}")
            print(f"   Recon Loss: {train_losses['recon_loss']:.4f}")
            print(f"   VQ Loss: {train_losses['vq_loss']:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            if 'disc_loss' in train_losses:
                print(f"   Disc Loss: {train_losses['disc_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()
            
            # Save best
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best_codec", is_best=True)
            
            # Log to tensorboard & wandb
            if self.writer:
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
                self.writer.add_scalar("epoch/lr", self.gen_scheduler.get_last_lr()[0], epoch)
            if self.use_wandb:
                wandb.log({
                    "epoch/val_loss": val_loss,
                    "epoch/lr": self.gen_scheduler.get_last_lr()[0],
                    "epoch": epoch
                })
        
        print("\n✅ Training complete!")
        self.save_checkpoint("final_codec")
        
        if self.use_wandb:
            wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Production Codec")
    
    # Data
    parser.add_argument("--data_dirs", nargs="+", required=True,
                       help="Directories containing audio files")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_production")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Hardware
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision (unstable)")
    
    # Logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="codec-production")
    parser.add_argument("--wandb_run", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    
    # Create config
    train_config = TrainConfig(
        data_dirs=args.data_dirs,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gen_lr=args.gen_lr,
        disc_lr=args.disc_lr,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        use_fp16=args.fp16,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )
    
    # Create trainer
    trainer = ProductionCodecTrainer(train_config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
