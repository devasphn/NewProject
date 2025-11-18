#!/usr/bin/env python3
"""
Training script for Speaker Embeddings
Creates distinct voice characteristics for 4 Telugu speakers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple

from speaker_embeddings import SpeakerEmbeddingSystem, SpeakerDataAugmentation
from telugu_codec import TeluCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerDataset(Dataset):
    """Dataset for training speaker embeddings"""
    
    def __init__(self, data_dir: str, codec_path: str):
        self.data_dir = Path(data_dir)
        
        # Load codec for encoding
        self.codec = TeluCodec()
        if Path(codec_path).exists():
            checkpoint = torch.load(codec_path, map_location='cpu')
            self.codec.load_state_dict(checkpoint['model_state'])
        self.codec.eval()
        
        # Speaker mapping based on data sources
        self.speaker_mapping = {
            "raw_talks_male": 0,      # Young male podcaster
            "10tv_male": 1,           # Mature male news anchor
            "raw_talks_female": 2,    # Young female podcaster
            "sakshi_female": 3        # Professional female news
        }
        
        # Load file paths
        self.samples = []
        for source, speaker_id in self.speaker_mapping.items():
            source_pattern = source.split('_')[0]  # Get base name
            
            # Find matching directories
            for subdir in self.data_dir.glob(f"*{source_pattern}*"):
                if subdir.is_dir():
                    audio_files = list(subdir.glob("*.wav")) + list(subdir.glob("*.mp3"))
                    for audio_file in audio_files[:100]:  # Limit per speaker
                        self.samples.append((audio_file, speaker_id))
        
        logger.info(f"Loaded {len(self.samples)} samples for {len(self.speaker_mapping)} speakers")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, speaker_id = self.samples[idx]
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Take 2-second segment
            segment_length = 32000  # 2 seconds at 16kHz
            if waveform.size(1) > segment_length:
                start = torch.randint(0, waveform.size(1) - segment_length, (1,)).item()
                waveform = waveform[:, start:start + segment_length]
            else:
                # Pad if too short
                waveform = F.pad(waveform, (0, segment_length - waveform.size(1)))
            
            # Encode with codec
            with torch.no_grad():
                codes = self.codec.encode(waveform)
            
            return codes.squeeze(0), speaker_id
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return random data as fallback
            return torch.randn(10, 256), speaker_id

class SpeakerTrainer:
    """Trainer for speaker embedding system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize speaker system
        self.speaker_system = SpeakerEmbeddingSystem(
            embedding_dim=config['embedding_dim']
        ).to(self.device)
        
        # Data augmentation
        self.augmenter = SpeakerDataAugmentation()
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.speaker_loss = nn.CrossEntropyLoss()
        
        # Speaker classifier for training
        self.speaker_classifier = nn.Sequential(
            nn.Linear(config['embedding_dim'], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)  # 4 speakers
        ).to(self.device)
        
        # Optimizer
        params = list(self.speaker_system.parameters()) + \
                 list(self.speaker_classifier.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config['learning_rate'],
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Dataset
        self.dataset = SpeakerDataset(
            config['data_dir'],
            config['codec_path']
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        speaker_ids: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Contrastive loss to push same speakers together, different apart
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create label matrix (1 if same speaker, 0 if different)
        labels = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)
        labels = labels.float().to(self.device)
        
        # Mask out diagonal
        mask = torch.eye(batch_size, device=self.device).bool()
        similarity = similarity.masked_fill(mask, -float('inf'))
        labels = labels.masked_fill(mask, 0)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        pos_sum = (exp_sim * labels).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)
        
        loss = -torch.log(pos_sum / total_sum + 1e-8).mean()
        
        return loss
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.speaker_system.train()
        self.speaker_classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (codes, speaker_ids) in enumerate(pbar):
            codes = codes.to(self.device)
            speaker_ids = speaker_ids.to(self.device)
            
            # Generate embeddings
            embeddings = self.speaker_system(
                speaker_ids,
                accent_level=1  # Moderate accent
            )
            
            # Classification loss
            logits = self.speaker_classifier(embeddings)
            cls_loss = self.speaker_loss(logits, speaker_ids)
            
            # Contrastive loss
            cont_loss = self.contrastive_loss(embeddings, speaker_ids)
            
            # Total loss
            loss = cls_loss + 0.5 * cont_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.speaker_system.parameters(), 1.0
            )
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == speaker_ids).sum().item()
            total += speaker_ids.size(0)
            
            if batch_idx % 10 == 0:
                accuracy = 100 * correct / total
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': f'{accuracy:.1f}%',
                    'lr': self.scheduler.get_last_lr()[0]
                })
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate speaker distinctiveness"""
        self.speaker_system.eval()
        
        with torch.no_grad():
            # Generate embeddings for all speakers
            speaker_ids = torch.arange(4).to(self.device)
            embeddings = []
            
            for sid in speaker_ids:
                embed = self.speaker_system(
                    sid.unsqueeze(0),
                    accent_level=1
                )
                embeddings.append(embed)
            
            embeddings = torch.cat(embeddings, dim=0)
            
            # Compute pairwise distances
            distances = torch.cdist(embeddings, embeddings, p=2)
            
            # Log statistics
            logger.info("Speaker embedding distances:")
            speaker_names = ["male_young", "male_mature", "female_young", "female_prof"]
            
            for i in range(4):
                for j in range(i+1, 4):
                    dist = distances[i, j].item()
                    logger.info(f"  {speaker_names[i]} <-> {speaker_names[j]}: {dist:.3f}")
            
            # Check minimum separation
            min_dist = distances[~torch.eye(4, dtype=bool, device=self.device)].min().item()
            
        return min_dist
    
    def train(self):
        """Main training loop"""
        best_accuracy = 0
        best_separation = 0
        
        for epoch in range(self.config['num_epochs']):
            # Train
            avg_loss, accuracy = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
            
            # Validate
            if epoch % 5 == 0:
                min_separation = self.validate()
                logger.info(f"Minimum speaker separation: {min_separation:.3f}")
                
                # Save best model
                if accuracy > best_accuracy and min_separation > best_separation:
                    best_accuracy = accuracy
                    best_separation = min_separation
                    self.save_embeddings(epoch, best=True)
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_embeddings(epoch)
        
        logger.info(f"Training complete! Best accuracy: {best_accuracy:.1f}%")
    
    def save_embeddings(self, epoch: int, best: bool = False):
        """Save speaker embeddings"""
        output_path = Path(self.config['output_path'])
        
        if best:
            save_path = output_path.parent / "speaker_embeddings.json"
        else:
            save_path = output_path.parent / f"speaker_embeddings_epoch_{epoch}.json"
        
        self.speaker_system.save_embeddings(str(save_path))
        logger.info(f"Saved embeddings to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Speaker Embeddings")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--codec_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/workspace/models/speaker_embeddings.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    args = parser.parse_args()
    
    config = {
        "data_dir": args.data_dir,
        "codec_path": args.codec_path,
        "output_path": args.output_path,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim
    }
    
    trainer = SpeakerTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()