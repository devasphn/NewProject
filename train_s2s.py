#!/usr/bin/env python3
"""
Training script for Telugu S2S Transformer
Optimized for H200 with DeepSpeed and Flash Attention
Target: 18-24 hours for production quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple
import argparse

# Import our models
from s2s_transformer import TeluguS2STransformer, S2SConfig, EMOTION_IDS, SPEAKER_IDS
from telugu_codec import TeluCodec

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S2SDataset(Dataset):
    """Dataset for S2S training with conversational pairs"""
    
    def __init__(self, data_dir: str, codec_model: TeluCodec, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.codec = codec_model
        
        # Load conversational pairs
        metadata_file = self.data_dir / "metadata" / f"{split}_pairs.json"
        with open(metadata_file, 'r') as f:
            self.pairs = json.load(f)
        
        logger.info(f"Loaded {len(self.pairs)} conversation pairs for {split}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load input and target audio
        input_audio = self._load_audio(pair["input_path"])
        target_audio = self._load_audio(pair["target_path"])
        
        # Encode to codes using codec
        with torch.no_grad():
            input_codes = self.codec.encode(input_audio)
            target_codes = self.codec.encode(target_audio)
        
        # Get metadata
        speaker_id = pair.get("speaker_id", 0)
        emotion_id = pair.get("emotion_id", 0)
        
        return {
            "input_codes": input_codes.squeeze(0),
            "target_codes": target_codes.squeeze(0),
            "speaker_id": speaker_id,
            "emotion_id": emotion_id
        }
    
    def _load_audio(self, path):
        # Load and preprocess audio
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        return waveform

class S2STrainer:
    """Trainer for S2S model with H200 optimizations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load codec for encoding
        self.codec = self._load_codec()
        
        # Initialize S2S model
        model_config = S2SConfig(
            hidden_dim=config["hidden_dim"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            use_flash_attn=True
        )
        self.model = TeluguS2STransformer(model_config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.98),
            weight_decay=0.01
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config["learning_rate"],
            total_steps=config["total_steps"],
            pct_start=0.05
        )
        
        # Datasets
        self.train_dataset = S2SDataset(config["data_dir"], self.codec, "train")
        self.val_dataset = S2SDataset(config["data_dir"], self.codec, "validation")
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize wandb
        if config.get("use_wandb", True):
            wandb.init(
                project="telugu-s2s",
                config=config,
                name=f"s2s_{config['experiment_name']}"
            )
        
        # Compile model
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="max-autotune")
            logger.info("Model compiled with torch.compile()")
        
        self.global_step = 0
    
    def _load_codec(self):
        """Load pre-trained codec"""
        codec_path = Path(self.config["codec_path"])
        codec = TeluCodec().to(self.device)
        
        if codec_path.exists():
            checkpoint = torch.load(codec_path, map_location=self.device)
            codec.load_state_dict(checkpoint["model_state"])
            logger.info(f"Loaded codec from {codec_path}")
        else:
            logger.warning("Codec checkpoint not found, using random initialization")
        
        codec.eval()
        return codec
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_codes = batch["input_codes"].to(self.device)
            target_codes = batch["target_codes"].to(self.device)
            speaker_ids = batch["speaker_id"].to(self.device)
            emotion_ids = batch["emotion_id"].to(self.device)
            
            # Mixed precision forward
            with autocast():
                loss = self.model(input_codes, target_codes, speaker_ids, emotion_ids)
            
            # Backward
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
            self.global_step += 1
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0]
                })
            
            if self.global_step % 100 == 0 and wandb.run:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step
                })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validation with generation quality metrics"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            input_codes = batch["input_codes"].to(self.device)
            target_codes = batch["target_codes"].to(self.device)
            speaker_ids = batch["speaker_id"].to(self.device)
            emotion_ids = batch["emotion_id"].to(self.device)
            
            with autocast():
                loss = self.model(input_codes, target_codes, speaker_ids, emotion_ids)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Test streaming generation
        test_latencies = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # Generate one sample
            test_input = input_codes[:1]
            test_speaker = speaker_ids[:1]
            test_emotion = emotion_ids[:1]
            
            start.record()
            chunks = []
            for chunk in self.model.generate_streaming(
                test_input, test_speaker, test_emotion, max_new_tokens=20
            ):
                chunks.append(chunk)
                break  # Just first chunk for latency
            end.record()
            
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
            test_latencies.append(latency)
        
        avg_latency = np.mean(test_latencies)
        
        if wandb.run:
            wandb.log({
                "val/loss": avg_loss,
                "val/latency_ms": avg_latency,
                "epoch": epoch
            })
        
        return avg_loss, avg_latency
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step
        }
        
        save_dir = Path(self.config["checkpoint_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if best:
            path = save_dir / "s2s_best.pt"
        else:
            path = save_dir / f"s2s_epoch_{epoch}.pt"
        
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
                val_loss, val_latency = self.validate(epoch)
                logger.info(f"Val loss: {val_loss:.4f}, Latency: {val_latency:.2f}ms")
                
                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, best=True)
                
                # Early stopping if latency target met
                if val_latency < 150:
                    logger.info(f"Target latency achieved: {val_latency:.2f}ms < 150ms")
            
            # Regular checkpoint
            if epoch % self.config["save_interval"] == 0:
                self.save_checkpoint(epoch)
        
        logger.info(f"Training completed! Best loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train Telugu S2S Transformer")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--codec_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/s2s")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--experiment_name", type=str, default="telugu_s2s_h200")
    args = parser.parse_args()
    
    config = {
        "data_dir": args.data_dir,
        "codec_path": args.codec_path,
        "checkpoint_dir": args.checkpoint_dir,
        "experiment_name": args.experiment_name,
        
        # Model config
        "hidden_dim": 768,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        
        # Training config  
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "total_steps": args.num_epochs * 1000,  # Approximate
        "val_interval": 5,
        "save_interval": 10,
        "use_wandb": True
    }
    
    trainer = S2STrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()