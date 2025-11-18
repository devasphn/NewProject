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
    
    def __init__(self, data_dir: str, segment_length: int = 16000, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.split = split
        
        # Load metadata
        metadata_file = self.data_dir / "metadata" / f"{split}.json"
        with open(metadata_file, 'r') as f:
            self.segments = json.load(f)
        
        logger.info(f"Loaded {len(self.segments)} segments for {split}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        audio_path = segment["audio_path"]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Random crop or pad
        if waveform.shape[1] > self.segment_length:
            # Random crop
            start = torch.randint(0, waveform.shape[1] - self.segment_length + 1, (1,))
            waveform = waveform[:, start:start + self.segment_length]
        elif waveform.shape[1] < self.segment_length:
            # Pad
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        return waveform

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
        if config.get("use_wandb", True):
            wandb.init(
                project="telugu-codec",
                config=config,
                name=f"telucodec_{config['experiment_name']}"
            )
        
        # Compile model for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile()")
    
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
            if batch_idx % 50 == 0 and wandb.run:
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
            
            # Calculate SNR
            signal_power = (audio ** 2).mean()
            noise_power = ((audio - output["audio"]) ** 2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            total_snr += snr.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_snr = total_snr / len(self.val_loader)
        
        if wandb.run:
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