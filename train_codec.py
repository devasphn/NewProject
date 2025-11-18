#!/usr/bin/env python3
"""
Training script for TeluCodec
Optimized for H200 GPU with mixed precision training
Target: 6-8 hours training for production quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchaudio
from pathlib import Path
import json
import logging
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional
import numpy as np
from telugu_codec import TeluCodec
import argparse
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeluguAudioDataset(Dataset):
    """Dataset for Telugu audio codec training"""
    
    def __init__(self, data_dir: str, segment_length: int = 16000, split: str = "train", val_ratio: float = 0.1, test_ratio: float = 0.1):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.split = split
        
        # Find all WAV files recursively
        all_audio_files = list(self.data_dir.rglob("*.wav"))
        if not all_audio_files:
            raise ValueError(f"No WAV files found in {data_dir}")
        
        # Sort for consistent ordering
        all_audio_files.sort()
        
        # Create train/val/test splits
        total_files = len(all_audio_files)
        n_test = int(total_files * test_ratio)
        n_val = int(total_files * val_ratio)
        n_train = total_files - n_test - n_val
        
        if split == "train":
            self.audio_files = all_audio_files[:n_train]
        elif split == "validation":
            self.audio_files = all_audio_files[n_train:n_train + n_val]
        elif split == "test":
            self.audio_files = all_audio_files[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        logger.info(f"Loaded {len(self.audio_files)} audio files for {split} split")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Random crop or pad to segment_length
            if waveform.shape[1] > self.segment_length:
                # Random crop during training, deterministic during val/test
                if self.split == "train":
                    max_start = waveform.shape[1] - self.segment_length
                    start = torch.randint(0, max_start + 1, (1,)).item()
                else:
                    start = 0  # Always use beginning for validation/test
                waveform = waveform[:, start:start + self.segment_length]
            elif waveform.shape[1] < self.segment_length:
                # Pad with zeros
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            
            # NO NORMALIZATION - let model learn the actual audio scale
            # Decoder has no tanh, so it can output any range to match input
            
            return waveform
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}. Returning silence.")
            # Return silence if file is corrupted
            return torch.zeros(1, self.segment_length)

class CodecTrainer:
    """Trainer for TeluCodec with H200 optimizations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = TeluCodec(
            hidden_dim=config["hidden_dim"],
            codebook_size=config["codebook_size"],
            num_quantizers=config["num_quantizers"]
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"] * config["steps_per_epoch"],
            eta_min=1e-6
        )
        
        # Datasets
        self.train_dataset = TeluguAudioDataset(
            config["data_dir"],
            segment_length=config["segment_length"],
            split="train"
        )
        self.val_dataset = TeluguAudioDataset(
            config["data_dir"],
            segment_length=config["segment_length"],
            split="validation"
        )
        
        # DataLoaders with H200 optimizations
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize wandb
        self.use_wandb = config.get("use_wandb", True)
        if self.use_wandb:
            try:
                wandb.init(
                    project="telugu-codec",
                    config=config,
                    name=f"{config['experiment_name']}"
                )
                logger.info("WandB initialized successfully")
            except Exception as e:
                logger.warning(f"WandB initialization failed: {e}. Continuing without WandB.")
                self.use_wandb = False
        
        # Compile model for faster training (PyTorch 2.0+)
        # Disabled due to dynamic shape handling issues with decoder output
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
        #     logger.info("Model compiled with torch.compile()")
        logger.info("torch.compile disabled for compatibility")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(self.device)
            
            # Mixed precision forward pass
            with autocast():
                output = self.model(audio)
                loss = output["loss"]
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            total_recon_loss += output["recon_loss"].item()
            total_vq_loss += output["vq_loss"].item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": loss.item(),
                    "recon": output["recon_loss"].item(),
                    "vq": output["vq_loss"].item(),
                    "lr": self.scheduler.get_last_lr()[0]
                })
            
            # Wandb logging
            if batch_idx % 50 == 0 and self.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/recon_loss": output["recon_loss"].item(),
                    "train/vq_loss": output["vq_loss"].item(),
                    "train/perceptual_loss": output["perceptual_loss"].item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validation pass"""
        self.model.eval()
        total_loss = 0
        total_snr = 0
        
        for audio in tqdm(self.val_loader, desc="Validation"):
            audio = audio.to(self.device)
            
            with autocast():
                output = self.model(audio)
            
            total_loss += output["loss"].item()
            
            # Calculate SNR with DEBUG logging
            # Log actual values to understand what's happening
            audio_min = audio.min().item()
            audio_max = audio.max().item()
            audio_mean = audio.mean().item()
            audio_std = audio.std().item()
            
            output_min = output["audio"].min().item()
            output_max = output["audio"].max().item()
            output_mean = output["audio"].mean().item()
            output_std = output["audio"].std().item()
            
            # Calculate simple SNR on full signal
            signal_power = (audio ** 2).mean()
            noise_power = ((audio - output["audio"]) ** 2).mean()
            
            # Prevent division by zero
            if signal_power < 1e-10:
                signal_power = 1e-10
            if noise_power < 1e-10:
                noise_power = 1e-10
                
            snr = 10 * torch.log10(signal_power / noise_power)
            
            logger.info(f"\n=== VALIDATION SNR DEBUG ===")
            logger.info(f"Input  range: [{audio_min:.6f}, {audio_max:.6f}], mean={audio_mean:.6f}, std={audio_std:.6f}")
            logger.info(f"Output range: [{output_min:.6f}, {output_max:.6f}], mean={output_mean:.6f}, std={output_std:.6f}")
            logger.info(f"Signal power: {signal_power:.8f}")
            logger.info(f"Noise power:  {noise_power:.8f}")
            logger.info(f"SNR: {snr.item():.2f} dB")
            logger.info(f"==========================\n")
            
            total_snr += snr.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_snr = total_snr / len(self.val_loader)
        
        if self.use_wandb:
            wandb.log({
                "val/loss": avg_loss,
                "val/snr": avg_snr,
                "epoch": epoch
            })
        
        return avg_loss, avg_snr
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": self.config
        }
        
        if best:
            path = Path(self.config["checkpoint_dir"]) / "best_codec.pt"
        else:
            path = Path(self.config["checkpoint_dir"]) / f"codec_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validate
            if epoch % self.config["val_interval"] == 0:
                val_loss, val_snr = self.validate(epoch)
                logger.info(f"Val loss: {val_loss:.4f}, SNR: {val_snr:.2f} dB")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, best=True)
            
            # Regular checkpoint
            if epoch % self.config["save_interval"] == 0:
                self.save_checkpoint(epoch)
        
        logger.info("Training completed!")
        return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train TeluCodec")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Telugu data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/codec")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--experiment_name", type=str, default="telucodec_h200")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX after training")
    args = parser.parse_args()
    
    # Configuration for H200 training
    config = {
        "data_dir": args.data_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "experiment_name": args.experiment_name,
        
        # Model config
        "hidden_dim": 1024,
        "codebook_size": 1024,
        "num_quantizers": 8,
        "segment_length": 16000 * 2,  # 2 seconds
        
        # Training config
        "batch_size": args.batch_size,  # H200 can handle large batches
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "steps_per_epoch": 1000,
        "val_interval": 5,
        "save_interval": 10,
        "use_wandb": True,
        
        # H200 specific
        "gradient_accumulation_steps": 4,
        "mixed_precision": True,
    }
    
    # Create checkpoint directory
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer = CodecTrainer(config)
    best_loss = trainer.train()
    
    logger.info(f"Training completed! Best validation loss: {best_loss:.4f}")
    
    # Export to ONNX for deployment
    if args.export_onnx:
        logger.info("Exporting to ONNX...")
        dummy_input = torch.randn(1, 1, 16000).to(trainer.device)
        torch.onnx.export(
            trainer.model,
            dummy_input,
            Path(config["checkpoint_dir"]) / "telucodec.onnx",
            input_names=["audio"],
            output_names=["codes"],
            dynamic_axes={"audio": {2: "length"}, "codes": {2: "length"}},
            opset_version=14
        )
        logger.info("ONNX export complete!")

if __name__ == "__main__":
    main()