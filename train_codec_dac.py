"""
CORRECTED GAN Training with Proper DAC Discriminators
Uses Multi-Period + Multi-Scale STFT discriminators
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchaudio

# Import codec and PROPER discriminator
from telugu_codec_fixed import TeluCodec
from discriminator_dac import (
    DACDiscriminator,
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional: WandB logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")


class AudioDataset(Dataset):
    """Dataset for loading audio files with fixed normalization"""
    def __init__(self, data_dir, segment_length=16000, sample_rate=16000):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Find all audio files recursively
        self.audio_files = list(self.data_dir.rglob("*.wav")) + \
                          list(self.data_dir.rglob("*.mp3")) + \
                          list(self.data_dir.rglob("*.flac"))
        
        logger.info(f"Found {len(self.audio_files)} audio files")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Get random segment
        if waveform.shape[1] > self.segment_length:
            start = random.randint(0, waveform.shape[1] - self.segment_length)
            waveform = waveform[:, start:start + self.segment_length]
        elif waveform.shape[1] < self.segment_length:
            # Pad with zeros
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # CRITICAL: Fixed -16 dB RMS normalization
        rms = torch.sqrt(torch.mean(waveform ** 2))
        
        if rms > 1e-8:
            target_rms = 0.158  # -16 dB
            waveform = waveform * (target_rms / rms)
            
            # Clip to prevent overflow
            max_val = waveform.abs().max()
            if max_val > 1.0:
                waveform = waveform / max_val * 0.95
        
        return waveform


def train_discriminator_step(
    codec, discriminator, batch, disc_optimizer, scaler
):
    """Single discriminator training step"""
    discriminator.train()
    codec.eval()
    
    with autocast():
        real_audio = batch.to(next(discriminator.parameters()).device)
        
        # Generate fake audio
        with torch.no_grad():
            codec_output = codec(real_audio)
            fake_audio = codec_output["audio"]
        
        # Discriminator forward
        real_logits, _ = discriminator(real_audio)
        fake_logits, _ = discriminator(fake_audio.detach())
        
        # Discriminator loss
        loss, real_loss, fake_loss = discriminator_loss(real_logits, fake_logits)
    
    # Backward
    disc_optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(disc_optimizer)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    scaler.step(disc_optimizer)
    scaler.update()
    
    return {
        "total": loss.item(),
        "real": real_loss.item(),
        "fake": fake_loss.item()
    }


def train_generator_step(
    codec, discriminator, batch, gen_optimizer, scaler, args
):
    """Single generator training step"""
    codec.train()
    discriminator.eval()
    
    with autocast():
        real_audio = batch.to(next(codec.parameters()).device)
        
        # Generator forward
        codec_output = codec(real_audio)
        fake_audio = codec_output["audio"]
        recon_loss = codec_output["recon_loss"]
        vq_loss = codec_output["vq_loss"]
        
        # Discriminator evaluation
        fake_logits, fake_features = discriminator(fake_audio)
        real_logits, real_features = discriminator(real_audio)
        
        # Generator losses
        adv_loss = generator_adversarial_loss(fake_logits)
        feat_loss = feature_matching_loss(real_features, fake_features)
        
        # Combined loss with STRONGER feature matching
        total_loss = (
            args.adv_weight * adv_loss +
            args.feat_weight * feat_loss +      # INCREASED weight!
            args.recon_weight * recon_loss +
            args.vq_weight * vq_loss
        )
    
    # Backward
    gen_optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(gen_optimizer)
    torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
    scaler.step(gen_optimizer)
    scaler.update()
    
    return {
        "total_loss": total_loss.item(),
        "adv_loss": adv_loss.item(),
        "feat_loss": feat_loss.item(),
        "recon_loss": recon_loss.item(),
        "vq_loss": vq_loss.item()
    }


@torch.no_grad()
def validate(codec, val_loader, device):
    """Validation loop"""
    codec.eval()
    
    total_loss = 0
    snr_values = []
    
    for batch in val_loader:
        batch = batch.to(device)
        
        output = codec(batch)
        total_loss += output["loss"].item()
        
        # Calculate SNR
        audio_recon = output["audio"]
        signal_power = torch.mean(batch ** 2)
        noise_power = torch.mean((batch - audio_recon) ** 2)
        
        if noise_power > 1e-10:
            snr = 10 * torch.log10(signal_power / noise_power)
            snr_values.append(snr.item())
    
    avg_loss = total_loss / len(val_loader)
    avg_snr = sum(snr_values) / len(snr_values) if snr_values else float('-inf')
    
    # Amplitude statistics
    input_std = batch.std().item()
    output_std = audio_recon.std().item()
    amplitude_ratio = output_std / input_std if input_std > 0 else 0
    
    return {
        "loss": avg_loss,
        "snr": avg_snr,
        "input_std": input_std,
        "output_std": output_std,
        "amplitude_ratio": amplitude_ratio
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="telugu-codec-dac",
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create dataset
    dataset = AudioDataset(
        args.data_dir,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate
    )
    
    # Split train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize models
    logger.info("Initializing models...")
    codec = TeluCodec(
        sample_rate=args.sample_rate,
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers
    ).to(device)
    
    # CRITICAL: Use proper DAC discriminator!
    discriminator = DACDiscriminator(
        periods=[2, 3, 5, 7, 11],
        n_ffts=[2048, 1024, 512]
    ).to(device)
    
    # Log model sizes
    codec_params = sum(p.numel() for p in codec.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    logger.info(f"Codec parameters: {codec_params/1e6:.2f}M")
    logger.info(f"Discriminator parameters: {disc_params/1e6:.2f}M")
    logger.info("Using DAC discriminators: 5 MPD + 3 STFT = 8 total")
    
    # Optimizers
    gen_optimizer = AdamW(
        codec.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.9),
        weight_decay=0.01
    )
    
    disc_optimizer = AdamW(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.9),
        weight_decay=0.01
    )
    
    # Learning rate schedulers
    gen_scheduler = CosineAnnealingLR(gen_optimizer, T_max=args.num_epochs)
    disc_scheduler = CosineAnnealingLR(disc_optimizer, T_max=args.num_epochs)
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_snr = float('-inf')
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Training
        codec.train()
        discriminator.train()
        
        disc_losses = {"total": [], "real": [], "fake": []}
        gen_losses = {"total": [], "adv": [], "feat": [], "recon": [], "vq": []}
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Train discriminator
            disc_loss_dict = train_discriminator_step(
                codec, discriminator, batch, disc_optimizer, scaler
            )
            for k, v in disc_loss_dict.items():
                disc_losses[k].append(v)
            
            # Train generator
            gen_loss_dict = train_generator_step(
                codec, discriminator, batch, gen_optimizer, scaler, args
            )
            for key, value in gen_loss_dict.items():
                if key == "total_loss":
                    gen_losses["total"].append(value)
                else:
                    gen_losses[key.replace("_loss", "")].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                "disc": f"{disc_loss_dict['total']:.4f}",
                "gen": f"{gen_loss_dict['total_loss']:.4f}",
                "adv": f"{gen_loss_dict['adv_loss']:.4f}",
                "feat": f"{gen_loss_dict['feat_loss']:.4f}",
                "recon": f"{gen_loss_dict['recon_loss']:.4f}",
                "vq": f"{gen_loss_dict['vq_loss']:.4f}"
            })
        
        # Calculate averages
        avg_disc_loss = {k: sum(v) / len(v) for k, v in disc_losses.items()}
        avg_gen_loss = {k: sum(v) / len(v) for k, v in gen_losses.items()}
        
        logger.info(f"Discriminator loss: {avg_disc_loss['total']:.4f}")
        logger.info(f"  - Real loss: {avg_disc_loss['real']:.4f}")
        logger.info(f"  - Fake loss: {avg_disc_loss['fake']:.4f}")
        logger.info(f"Generator loss: {avg_gen_loss['total']:.4f}")
        logger.info(f"  - Adversarial: {avg_gen_loss['adv']:.4f}")
        logger.info(f"  - Feature: {avg_gen_loss['feat']:.4f}")
        logger.info(f"  - Reconstruction: {avg_gen_loss['recon']:.4f}")
        logger.info(f"  - VQ: {avg_gen_loss['vq']:.4f}")
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            logger.info("\nValidating...")
            val_metrics = validate(codec, val_loader, device)
            
            logger.info(f"\n=== Validation Statistics ===")
            logger.info(f"Loss: {val_metrics['loss']:.4f}")
            logger.info(f"SNR: {val_metrics['snr']:.2f} dB")
            logger.info(f"Input std: {val_metrics['input_std']:.4f}")
            logger.info(f"Output std: {val_metrics['output_std']:.4f}")
            logger.info(f"Amplitude ratio: {val_metrics['amplitude_ratio']:.3f}")
            
            # Check if improving
            if epoch == 5:
                logger.info("\n🔍 CRITICAL CHECK:")
                if val_metrics['snr'] > 5.0:
                    logger.info("✅ SNR is POSITIVE and good! Training is working!")
                elif val_metrics['snr'] > 0:
                    logger.info("⚠️  SNR is positive but low. Monitor next epochs.")
                else:
                    logger.info("❌ SNR is NEGATIVE! Something is wrong!")
            
            # Save best model
            if val_metrics['snr'] > best_snr:
                best_snr = val_metrics['snr']
                logger.info(f"✅ New best model! SNR: {best_snr:.2f} dB")
                
                checkpoint_path = os.path.join(args.checkpoint_dir, "best_codec.pt")
                torch.save({
                    "epoch": epoch,
                    "codec_state_dict": codec.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                    "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                    "snr": best_snr,
                }, checkpoint_path)
            
            # Log to WandB
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train/disc_loss": avg_disc_loss['total'],
                    "train/disc_real_loss": avg_disc_loss['real'],
                    "train/disc_fake_loss": avg_disc_loss['fake'],
                    "train/gen_loss": avg_gen_loss['total'],
                    "train/adv_loss": avg_gen_loss['adv'],
                    "train/feat_loss": avg_gen_loss['feat'],
                    "train/recon_loss": avg_gen_loss['recon'],
                    "train/vq_loss": avg_gen_loss['vq'],
                    "val/loss": val_metrics['loss'],
                    "val/snr": val_metrics['snr'],
                    "val/amplitude_ratio": val_metrics['amplitude_ratio'],
                    "lr": gen_scheduler.get_last_lr()[0]
                })
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "codec_state_dict": codec.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                "disc_optimizer_state_dict": disc_optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Update learning rates
        gen_scheduler.step()
        disc_scheduler.step()
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Best SNR: {best_snr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Telugu codec with DAC discriminators")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--num_quantizers", type=int, default=8)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--segment_length", type=int, default=16000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    
    # Loss weights - UPDATED for DAC discriminators
    parser.add_argument("--adv_weight", type=float, default=1.0)
    parser.add_argument("--feat_weight", type=float, default=10.0, help="INCREASED from 2.0!")
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--vq_weight", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="telugu_codec_dac")
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    main(args)
