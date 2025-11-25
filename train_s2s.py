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
from telugu_codec_fixed import TeluCodec

# Check Flash Attention availability
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S2SDataset(Dataset):
    """Dataset for S2S training - works with available Telugu audio data"""
    
    def __init__(self, data_dir: str, codec_model: TeluCodec, split: str = "train", 
                 segment_length: int = 32000, sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.split = split
        self.codec = codec_model
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.device = next(codec_model.parameters()).device
        
        # Find all audio files recursively
        self.audio_files = []
        self.speaker_ids = []
        
        # Search in all subdirectories
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.audio_files.extend(list(self.data_dir.rglob(ext)))
        
        # Assign speaker IDs based on directory structure
        for audio_file in self.audio_files:
            # Determine speaker based on path
            path_str = str(audio_file).lower()
            if 'male' in path_str and 'female' not in path_str:
                speaker_id = 0  # male
            elif 'female' in path_str:
                speaker_id = 1  # female
            elif 'indictts' in path_str:
                # Split indictts between speakers 2 and 3
                speaker_id = 2 if hash(str(audio_file)) % 2 == 0 else 3
            else:
                speaker_id = hash(str(audio_file)) % 4
            self.speaker_ids.append(speaker_id)
        
        # Split into train/val (90/10)
        total = len(self.audio_files)
        split_idx = int(0.9 * total)
        
        if split == "train":
            self.audio_files = self.audio_files[:split_idx]
            self.speaker_ids = self.speaker_ids[:split_idx]
        else:
            self.audio_files = self.audio_files[split_idx:]
            self.speaker_ids = self.speaker_ids[split_idx:]
        
        logger.info(f"Loaded {len(self.audio_files)} audio files for {split}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_ids[idx]
        
        try:
            # Load audio
            waveform = self._load_audio(audio_path)
            
            # For S2S training, we use same audio as input/output (reconstruction)
            input_audio = waveform
            target_audio = waveform.clone()
            
            # Encode with codec
            with torch.no_grad():
                # Move to codec device temporarily
                input_audio_dev = input_audio.to(self.device)
                target_audio_dev = target_audio.to(self.device)
                
                input_codes = self.codec.encode(input_audio_dev)
                target_codes = self.codec.encode(target_audio_dev)
                
                # CRITICAL: Ensure codes are long (integer) tensors!
                input_codes = input_codes.long().cpu()
                target_codes = target_codes.long().cpu()
            
            # Random emotion for training diversity
            emotion_id = torch.randint(0, 7, (1,)).item()
            
            return {
                "input_codes": input_codes.squeeze(0),  # [Q, T]
                "target_codes": target_codes.squeeze(0),  # [Q, T]
                "speaker_id": speaker_id,
                "emotion_id": emotion_id
            }
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return dummy data on error - MUST BE LONG TENSORS!
            dummy_codes = torch.zeros(8, 50, dtype=torch.long)  # [num_quantizers, seq_len]
            return {
                "input_codes": dummy_codes,
                "target_codes": dummy_codes,
                "speaker_id": 0,
                "emotion_id": 0
            }
    
    def _load_audio(self, path):
        """Load and preprocess audio"""
        import torchaudio
        waveform, sr = torchaudio.load(path)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Get segment
        if waveform.shape[1] > self.segment_length:
            start = torch.randint(0, waveform.shape[1] - self.segment_length, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length]
        elif waveform.shape[1] < self.segment_length:
            # Pad with zeros
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # Normalize
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 1e-8:
            waveform = waveform * (0.1 / rms)
        
        return waveform

class S2STrainer:
    """Trainer for S2S model - optimized for POC"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load codec for encoding FIRST
        self.codec = self._load_codec()
        
        # Initialize S2S model
        logger.info("Initializing S2S Transformer...")
        model_config = S2SConfig(
            hidden_dim=config.get("hidden_dim", 512),  # Smaller for POC
            num_encoder_layers=config.get("num_encoder_layers", 6),  # Smaller for POC
            num_decoder_layers=config.get("num_decoder_layers", 6),  # Smaller for POC
            use_flash_attn=FLASH_AVAILABLE
        )
        self.model = TeluguS2STransformer(model_config).to(self.device)
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"S2S Model parameters: {total_params/1e6:.2f}M")
        
        # Datasets - use codec on device
        logger.info("Loading datasets...")
        self.train_dataset = S2SDataset(config["data_dir"], self.codec, "train")
        self.val_dataset = S2SDataset(config["data_dir"], self.codec, "validation")
        
        if len(self.train_dataset) == 0:
            raise ValueError(f"No training data found in {config['data_dir']}")
        
        logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        
        # DataLoaders with fewer workers to avoid issues
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        if len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader)
        total_steps = config["num_epochs"] * steps_per_epoch
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
        
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        # Initialize wandb (optional)
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                wandb.init(
                    project="telugu-s2s",
                    config=config,
                    name=f"s2s_{config['experiment_name']}"
                )
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")
                self.use_wandb = False
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _load_codec(self):
        """Load pre-trained codec"""
        codec_path = Path(self.config["codec_path"])
        codec = TeluCodec().to(self.device)
        
        if codec_path.exists():
            checkpoint = torch.load(codec_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'codec_state_dict' in checkpoint:
                codec.load_state_dict(checkpoint['codec_state_dict'])
            elif 'model_state' in checkpoint:
                codec.load_state_dict(checkpoint['model_state'])
            elif 'state_dict' in checkpoint:
                codec.load_state_dict(checkpoint['state_dict'])
            else:
                codec.load_state_dict(checkpoint)
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
            try:
                # Move to device with correct types
                input_codes = batch["input_codes"].long().to(self.device)
                target_codes = batch["target_codes"].long().to(self.device)
                speaker_ids = batch["speaker_id"].long().to(self.device)
                emotion_ids = batch["emotion_id"].long().to(self.device)
                
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
                
                if self.global_step % 100 == 0 and self.use_wandb:
                    try:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/step": self.global_step
                        })
                    except:
                        pass
            
            except Exception as e:
                logger.warning(f"Batch {batch_idx} error: {e}")
                continue
        
        return total_loss / max(len(self.train_loader), 1)
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validation with generation quality metrics"""
        if self.val_loader is None or len(self.val_loader) == 0:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            try:
                input_codes = batch["input_codes"].to(self.device)
                target_codes = batch["target_codes"].to(self.device)
                speaker_ids = batch["speaker_id"].to(self.device)
                emotion_ids = batch["emotion_id"].to(self.device)
                
                with autocast():
                    loss = self.model(input_codes, target_codes, speaker_ids, emotion_ids)
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.warning(f"Validation batch error: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Test streaming generation latency
        avg_latency = 0.0
        try:
            test_latencies = []
            # Use a sample from training data
            sample_batch = next(iter(self.train_loader))
            test_input = sample_batch["input_codes"][:1].to(self.device)
            test_speaker = torch.tensor([0], device=self.device)
            test_emotion = torch.tensor([0], device=self.device)
            
            for _ in range(3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for chunk in self.model.generate_streaming(
                    test_input, test_speaker, test_emotion, max_new_tokens=10
                ):
                    break  # Just first chunk for latency
                end.record()
                
                torch.cuda.synchronize()
                latency = start.elapsed_time(end)
                test_latencies.append(latency)
            
            avg_latency = np.mean(test_latencies)
        except Exception as e:
            logger.warning(f"Latency test failed: {e}")
            avg_latency = 0.0
        
        if self.use_wandb:
            try:
                wandb.log({
                    "val/loss": avg_loss,
                    "val/latency_ms": avg_latency,
                    "epoch": epoch
                })
            except:
                pass
        
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
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/models/s2s")
    parser.add_argument("--batch_size", type=int, default=4)  # Smaller for memory
    parser.add_argument("--num_epochs", type=int, default=50)  # Faster for POC
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--experiment_name", type=str, default="telugu_s2s_poc")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    config = {
        "data_dir": args.data_dir,
        "codec_path": args.codec_path,
        "checkpoint_dir": args.checkpoint_dir,
        "experiment_name": args.experiment_name,
        
        # Model config - SMALLER for POC (faster training)
        "hidden_dim": 512,           # Smaller: 512 vs 768
        "num_encoder_layers": 6,     # Smaller: 6 vs 12
        "num_decoder_layers": 6,     # Smaller: 6 vs 12
        
        # Training config  
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "val_interval": 5,
        "save_interval": 10,
        "use_wandb": args.use_wandb
    }
    
    logger.info("=" * 60)
    logger.info("TELUGU S2S TRANSFORMER TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Codec path: {args.codec_path}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)
    
    try:
        trainer = S2STrainer(config)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()