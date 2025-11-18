"""
Training script for Telugu Codec - PRODUCTION VERSION
Based on EnCodec/DAC training methodology
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torchaudio
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import logging
from telugu_codec_fixed import TeluCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """Simple audio dataset with proper normalization"""
    def __init__(self, data_dir, segment_length=16000, sample_rate=16000):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.audio_files.extend(list(self.data_dir.glob(ext)))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
        
        logger.info(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(1, self.segment_length)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Get random segment
        if waveform.shape[1] > self.segment_length:
            start = random.randint(0, waveform.shape[1] - self.segment_length)
            waveform = waveform[:, start:start + self.segment_length]
        elif waveform.shape[1] < self.segment_length:
            # Pad with zeros
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # CRITICAL: Normalize to [-1, 1] range (standard for audio codecs)
        # This matches the tanh output range of the decoder
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95  # Scale to 0.95 to avoid clipping
        
        return waveform

def train_step(model, batch, optimizer, scaler):
    """Single training step with mixed precision"""
    model.train()
    
    with autocast(device_type='cuda'):
        output = model(batch)
        loss = output["loss"]
    
    # Backward pass with gradient scaling
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return output

@torch.no_grad()
def validate(model, val_loader, device):
    """Validation with SNR computation"""
    model.eval()
    
    total_loss = 0
    total_snr = 0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        batch = batch.to(device)
        
        with autocast(device_type='cuda'):
            output = model(batch)
        
        total_loss += output["loss"].item()
        
        # Compute SNR
        signal_power = (batch ** 2).mean()
        noise = batch - output["audio"]
        noise_power = (noise ** 2).mean()
        
        if noise_power > 1e-10:
            snr = 10 * torch.log10(signal_power / noise_power)
            total_snr += snr.item()
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_snr = total_snr / num_batches
    
    # Log sample statistics
    logger.info(f"\n=== Validation Statistics ===")
    logger.info(f"Loss: {avg_loss:.4f}")
    logger.info(f"SNR: {avg_snr:.2f} dB")
    
    if num_batches > 0:
        # Get last batch statistics
        with torch.no_grad():
            input_std = batch.std().item()
            output_std = output["audio"].std().item()
            input_mean = batch.mean().item()
            output_mean = output["audio"].mean().item()
            
            logger.info(f"Input  stats: mean={input_mean:.4f}, std={input_std:.4f}")
            logger.info(f"Output stats: mean={output_mean:.4f}, std={output_std:.4f}")
            logger.info(f"Amplitude ratio (std): {output_std/input_std:.3f}")
    
    return avg_loss, avg_snr

def main(args):
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="telugu-codec-fixed",
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Dataset
    train_dataset = AudioDataset(
        args.data_dir,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate
    )
    
    # Use 10% for validation
    val_size = len(train_dataset) // 10
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # DataLoaders
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
    
    # Model
    model = TeluCodec(
        sample_rate=args.sample_rate,
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimizer (following EnCodec settings)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.9),  # EnCodec uses beta1=0.5
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader),
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_snr = -float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch in pbar:
            batch = batch.to(device)
            
            output = train_step(model, batch, optimizer, scaler)
            scheduler.step()
            
            train_losses.append(output["loss"].item())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": output["loss"].item(),
                "recon": output["recon_loss"].item(),
                "vq": output["vq_loss"].item(),
                "lr": scheduler.get_last_lr()[0]
            })
            
            # Log to wandb
            if args.use_wandb and len(train_losses) % 10 == 0:
                wandb.log({
                    "train/loss": output["loss"].item(),
                    "train/recon_loss": output["recon_loss"].item(),
                    "train/vq_loss": output["vq_loss"].item(),
                    "train/spectral_loss": output["spectral_loss"].item(),
                    "train/mel_loss": output["mel_loss"].item(),
                    "train/lr": scheduler.get_last_lr()[0]
                })
        
        avg_train_loss = np.mean(train_losses)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_loss, val_snr = validate(model, val_loader, device)
            
            if args.use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/snr": val_snr,
                    "epoch": epoch + 1
                })
            
            # Save best model
            if val_snr > best_snr:
                best_snr = val_snr
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_snr": best_snr,
                    "config": vars(args)
                }
                
                save_path = Path(args.checkpoint_dir) / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"Saved best model with SNR: {best_snr:.2f} dB")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": vars(args)
            }
            
            save_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    logger.info(f"\nTraining complete! Best SNR: {best_snr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
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
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--segment_length", type=int, default=16000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="telugu_codec")
    
    args = parser.parse_args()
    main(args)
