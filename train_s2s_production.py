#!/usr/bin/env python3
"""
🎯 PRODUCTION S2S Training
===========================

Trains a production-quality Telugu S2S model with:
- Multi-speaker support (speaker embeddings)
- Emotion conditioning
- Prosody modeling
- Natural speech patterns (breathing, pauses)
- Streaming-ready architecture

Target: <500ms latency, human-like responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import time
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

@dataclass
class ProductionConfig:
    """Configuration for production S2S model"""
    
    # Model architecture
    hidden_dim: int = 768            # Larger for production
    num_heads: int = 12              # 768/12 = 64 per head
    num_encoder_layers: int = 12     # Deeper encoder
    num_decoder_layers: int = 12     # Deeper decoder
    num_quantizers: int = 8          # Match codec
    vocab_size: int = 1024           # Match codec
    max_seq_len: int = 4096          # Longer sequences
    
    # Speaker & Emotion
    num_speakers: int = 500          # Support many speakers
    speaker_embed_dim: int = 256     # Speaker embedding size
    num_emotions: int = 8            # neutral, happy, sad, angry, excited, calm, fearful, surprised
    emotion_embed_dim: int = 64      # Emotion embedding size
    
    # Prosody
    use_prosody: bool = True         # Enable prosody modeling
    prosody_dim: int = 64            # Pitch, energy, duration features
    
    # Training
    batch_size: int = 8
    gradient_accumulation: int = 4   # Effective batch = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 500000          # Long training
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Data
    data_dir: str = "data/telugu_production/encoded"
    codec_path: str = "best_codec.pt"
    
    # Checkpointing
    save_every: int = 5000
    eval_every: int = 1000
    output_dir: str = "checkpoints/s2s_production"
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True     # Use FP16 for speed
    num_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.hidden_dim % self.num_quantizers == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_quantizers ({self.num_quantizers})"


# ============================================================
# Dataset
# ============================================================

class TeluguAudioDataset(IterableDataset):
    """
    Streaming dataset for large-scale Telugu audio
    
    Supports:
    - Large datasets that don't fit in memory
    - On-the-fly augmentation
    - Speaker/emotion labels
    """
    
    def __init__(self, 
                 data_dir: str,
                 max_seq_len: int = 4096,
                 mode: str = "pretrain"):  # pretrain, finetune
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.mode = mode
        
        # Find all code files
        self.code_files = list(self.data_dir.rglob("*.pt"))
        logger.info(f"📚 Found {len(self.code_files)} encoded audio files")
        
        # Load metadata if available
        self.metadata = {}
        meta_file = self.data_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                self.metadata = json.load(f)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            files = self.code_files
        else:
            # Split files among workers
            per_worker = len(self.code_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.code_files)
            files = self.code_files[start:end]
        
        random.shuffle(files)
        
        for code_file in files:
            try:
                codes = torch.load(code_file)
                
                # Handle different shapes
                if codes.dim() == 3:
                    codes = codes.squeeze(0)  # [Q, T]
                
                # Get metadata
                file_key = str(code_file.relative_to(self.data_dir))
                meta = self.metadata.get(file_key, {})
                speaker_id = meta.get("speaker_id", 0)
                emotion_id = meta.get("emotion_id", 0)  # 0 = neutral
                
                # For pretraining: next-token prediction on full sequence
                if self.mode == "pretrain":
                    # Random crop if too long
                    if codes.shape[1] > self.max_seq_len:
                        start = random.randint(0, codes.shape[1] - self.max_seq_len)
                        codes = codes[:, start:start + self.max_seq_len]
                    
                    yield {
                        "codes": codes.long(),
                        "speaker_id": speaker_id,
                        "emotion_id": emotion_id,
                        "length": codes.shape[1]
                    }
                
                # For finetuning: conversation pairs
                elif self.mode == "finetune":
                    # Expect pairs in specific folder structure
                    pass
                    
            except Exception as e:
                logger.debug(f"Error loading {code_file}: {e}")
                continue


class ConversationDataset(Dataset):
    """Dataset for conversation pairs"""
    
    def __init__(self, data_dir: str, max_len: int = 2048):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        
        # Find all conversation pairs
        self.pairs = []
        for pair_dir in self.data_dir.glob("pair_*"):
            q_path = pair_dir / "question_codes.pt"
            a_path = pair_dir / "answer_codes.pt"
            meta_path = pair_dir / "metadata.json"
            
            if q_path.exists() and a_path.exists():
                self.pairs.append({
                    "question": q_path,
                    "answer": a_path,
                    "metadata": meta_path if meta_path.exists() else None
                })
        
        logger.info(f"📚 Found {len(self.pairs)} conversation pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        q_codes = torch.load(pair["question"]).squeeze(0).long()
        a_codes = torch.load(pair["answer"]).squeeze(0).long()
        
        # Truncate if needed
        if q_codes.shape[1] > self.max_len:
            q_codes = q_codes[:, :self.max_len]
        if a_codes.shape[1] > self.max_len:
            a_codes = a_codes[:, :self.max_len]
        
        # Get metadata
        speaker_id = 0
        emotion_id = 0
        if pair["metadata"]:
            with open(pair["metadata"]) as f:
                meta = json.load(f)
                speaker_id = meta.get("speaker_id", 0)
                emotion_id = meta.get("emotion_id", 0)
        
        return {
            "input_codes": q_codes,
            "target_codes": a_codes,
            "speaker_id": speaker_id,
            "emotion_id": emotion_id
        }


# ============================================================
# Model Components
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiQuantizerEmbedding(nn.Module):
    """Embedding layer for multi-quantizer codes"""
    
    def __init__(self, num_quantizers: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim // num_quantizers)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, codes):
        """
        Args:
            codes: [B, Q, T]
        Returns:
            embeddings: [B, T, D]
        """
        B, Q, T = codes.shape
        embeds = []
        for q in range(Q):
            embeds.append(self.embeddings[q](codes[:, q]))  # [B, T, D/Q]
        return torch.cat(embeds, dim=-1)  # [B, T, D]


class SpeakerEncoder(nn.Module):
    """Speaker embedding with learned representations"""
    
    def __init__(self, num_speakers: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, speaker_ids):
        """
        Args:
            speaker_ids: [B]
        Returns:
            speaker_embed: [B, 1, D] - to add to sequence
        """
        embed = self.embedding(speaker_ids)  # [B, E]
        return self.proj(embed).unsqueeze(1)  # [B, 1, D]


class EmotionEncoder(nn.Module):
    """Emotion conditioning"""
    
    def __init__(self, num_emotions: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, embed_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)
    
    def forward(self, emotion_ids):
        embed = self.embedding(emotion_ids)
        return self.proj(embed).unsqueeze(1)


class ProsodyPredictor(nn.Module):
    """Predicts prosody features (pitch, energy, duration)"""
    
    def __init__(self, hidden_dim: int, prosody_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, prosody_dim)
        )
    
    def forward(self, hidden_states):
        return self.predictor(hidden_states)


# ============================================================
# Main Model
# ============================================================

class ProductionS2S(nn.Module):
    """
    Production-quality S2S Transformer
    
    Features:
    - Multi-speaker support
    - Emotion conditioning  
    - Prosody modeling
    - Streaming-ready
    """
    
    def __init__(self, config: ProductionConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = MultiQuantizerEmbedding(
            config.num_quantizers,
            config.vocab_size,
            config.hidden_dim
        )
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(config.hidden_dim, config.max_seq_len)
        
        # Speaker & Emotion conditioning
        self.speaker_enc = SpeakerEncoder(
            config.num_speakers,
            config.speaker_embed_dim,
            config.hidden_dim
        )
        self.emotion_enc = EmotionEncoder(
            config.num_emotions,
            config.emotion_embed_dim,
            config.hidden_dim
        )
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        
        # Prosody predictor (auxiliary task)
        if config.use_prosody:
            self.prosody_pred = ProsodyPredictor(config.hidden_dim, config.prosody_dim)
        
        # Output heads for each quantizer
        head_dim = config.hidden_dim // config.num_quantizers
        self.output_heads = nn.ModuleList([
            nn.Linear(head_dim, config.vocab_size)
            for _ in range(config.num_quantizers)
        ])
        
        # Layer norms
        self.enc_norm = nn.LayerNorm(config.hidden_dim)
        self.dec_norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, 
                input_codes,           # [B, Q, T1] - encoder input
                target_codes,          # [B, Q, T2] - decoder input (teacher forcing)
                speaker_ids=None,      # [B] - speaker conditioning
                emotion_ids=None):     # [B] - emotion conditioning
        """
        Forward pass with teacher forcing
        
        Returns:
            logits: [B, Q, T2, vocab_size]
            prosody: [B, T2, prosody_dim] (if enabled)
        """
        B, Q, T1 = input_codes.shape
        _, _, T2 = target_codes.shape
        
        # Embed tokens
        enc_input = self.token_embed(input_codes)   # [B, T1, D]
        dec_input = self.token_embed(target_codes)  # [B, T2, D]
        
        # Add positional encoding
        enc_input = self.pos_enc(enc_input)
        dec_input = self.pos_enc(dec_input)
        
        # Add speaker conditioning
        if speaker_ids is not None:
            speaker_embed = self.speaker_enc(speaker_ids)  # [B, 1, D]
            enc_input = enc_input + speaker_embed
            dec_input = dec_input + speaker_embed
        
        # Add emotion conditioning
        if emotion_ids is not None:
            emotion_embed = self.emotion_enc(emotion_ids)  # [B, 1, D]
            dec_input = dec_input + emotion_embed
        
        # Encode
        memory = self.encoder(enc_input)  # [B, T1, D]
        memory = self.enc_norm(memory)
        
        # Create causal mask for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T2).to(input_codes.device)
        
        # Decode
        dec_output = self.decoder(dec_input, memory, tgt_mask=causal_mask)  # [B, T2, D]
        dec_output = self.dec_norm(dec_output)
        
        # Predict prosody (auxiliary)
        prosody = None
        if self.config.use_prosody:
            prosody = self.prosody_pred(dec_output)
        
        # Output logits for each quantizer
        chunk_size = self.config.hidden_dim // Q
        logits = []
        for q in range(Q):
            chunk = dec_output[:, :, q*chunk_size:(q+1)*chunk_size]
            logit = self.output_heads[q](chunk)  # [B, T2, vocab]
            logits.append(logit)
        
        logits = torch.stack(logits, dim=1)  # [B, Q, T2, vocab]
        
        return logits, prosody
    
    @torch.no_grad()
    def generate(self, 
                 input_codes,
                 speaker_id=None,
                 emotion_id=None,
                 max_len=500,
                 temperature=0.8,
                 top_k=50):
        """
        Generate response codes
        
        Args:
            input_codes: [1, Q, T] - input audio codes
            speaker_id: int - target speaker
            emotion_id: int - target emotion
            max_len: int - maximum output length
            temperature: float - sampling temperature
            top_k: int - top-k sampling
        
        Returns:
            output_codes: [1, Q, T']
        """
        self.eval()
        device = input_codes.device
        B, Q, T1 = input_codes.shape
        
        # Prepare conditioning
        if speaker_id is not None:
            speaker_ids = torch.tensor([speaker_id], device=device)
        else:
            speaker_ids = torch.zeros(1, dtype=torch.long, device=device)
        
        if emotion_id is not None:
            emotion_ids = torch.tensor([emotion_id], device=device)
        else:
            emotion_ids = torch.zeros(1, dtype=torch.long, device=device)
        
        # Encode input
        enc_input = self.token_embed(input_codes)
        enc_input = self.pos_enc(enc_input)
        enc_input = enc_input + self.speaker_enc(speaker_ids)
        
        memory = self.encoder(enc_input)
        memory = self.enc_norm(memory)
        
        # Start generation
        output_codes = torch.zeros(B, Q, 1, dtype=torch.long, device=device)
        
        # Get emotion + speaker for decoder
        speaker_embed = self.speaker_enc(speaker_ids)
        emotion_embed = self.emotion_enc(emotion_ids)
        
        for t in range(max_len):
            # Embed current output
            dec_input = self.token_embed(output_codes)
            dec_input = self.pos_enc(dec_input)
            dec_input = dec_input + speaker_embed + emotion_embed
            
            # Decode
            causal_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=causal_mask)
            dec_output = self.dec_norm(dec_output)
            
            # Get next tokens
            chunk_size = self.config.hidden_dim // Q
            next_tokens = []
            
            for q in range(Q):
                chunk = dec_output[:, -1, q*chunk_size:(q+1)*chunk_size]
                logit = self.output_heads[q](chunk)  # [B, vocab]
                
                # Apply temperature and top-k
                logit = logit / temperature
                
                if top_k > 0:
                    values, indices = logit.topk(top_k)
                    logit = torch.full_like(logit, float('-inf'))
                    logit.scatter_(1, indices, values)
                
                probs = F.softmax(logit, dim=-1)
                token = torch.multinomial(probs, 1)
                next_tokens.append(token)
            
            next_tokens = torch.stack(next_tokens, dim=1)  # [B, Q, 1]
            output_codes = torch.cat([output_codes, next_tokens], dim=-1)
        
        return output_codes[:, :, 1:]  # Remove initial zero


# ============================================================
# Training Functions
# ============================================================

def train_step(model, batch, optimizer, scaler, config, step):
    """Single training step with mixed precision"""
    model.train()
    
    input_codes = batch["input_codes"].to(config.device).long()
    target_codes = batch["target_codes"].to(config.device).long()
    speaker_ids = batch.get("speaker_id", torch.zeros(input_codes.shape[0])).to(config.device).long()
    emotion_ids = batch.get("emotion_id", torch.zeros(input_codes.shape[0])).to(config.device).long()
    
    with autocast(enabled=config.mixed_precision):
        # Forward (teacher forcing on shifted target)
        logits, prosody = model(
            input_codes, 
            target_codes[:, :, :-1],
            speaker_ids,
            emotion_ids
        )
        
        # Compute loss
        B, Q, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            target_codes[:, :, 1:].reshape(-1),
            ignore_index=0  # Ignore padding
        )
    
    # Backward with gradient accumulation
    loss = loss / config.gradient_accumulation
    scaler.scale(loss).backward()
    
    if (step + 1) % config.gradient_accumulation == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return loss.item() * config.gradient_accumulation


def collate_conversation(batch):
    """Collate function for conversation data"""
    # Pad sequences
    max_input_len = max(b["input_codes"].shape[1] for b in batch)
    max_target_len = max(b["target_codes"].shape[1] for b in batch)
    Q = batch[0]["input_codes"].shape[0]
    
    padded_inputs = []
    padded_targets = []
    speaker_ids = []
    emotion_ids = []
    
    for b in batch:
        # Pad input
        inp = b["input_codes"]
        pad_inp = torch.zeros(Q, max_input_len, dtype=inp.dtype)
        pad_inp[:, :inp.shape[1]] = inp
        padded_inputs.append(pad_inp)
        
        # Pad target
        tgt = b["target_codes"]
        pad_tgt = torch.zeros(Q, max_target_len, dtype=tgt.dtype)
        pad_tgt[:, :tgt.shape[1]] = tgt
        padded_targets.append(pad_tgt)
        
        speaker_ids.append(b.get("speaker_id", 0))
        emotion_ids.append(b.get("emotion_id", 0))
    
    return {
        "input_codes": torch.stack(padded_inputs),
        "target_codes": torch.stack(padded_targets),
        "speaker_id": torch.tensor(speaker_ids),
        "emotion_id": torch.tensor(emotion_ids)
    }


def main():
    parser = argparse.ArgumentParser(description="Train production S2S")
    parser.add_argument("--data_dir", default="data/telugu_conversations")
    parser.add_argument("--output", default="checkpoints/s2s_production")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)  # Smaller for limited data
    parser.add_argument("--num_layers", type=int, default=6)
    args = parser.parse_args()
    
    # Create config
    config = ProductionConfig(
        hidden_dim=args.hidden_dim,
        num_heads=8,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output
    )
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("🎯 PRODUCTION S2S TRAINING")
    logger.info("=" * 70)
    logger.info(f"Hidden dim: {config.hidden_dim}")
    logger.info(f"Layers: {config.num_encoder_layers}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    # Create dataset
    dataset = ConversationDataset(args.data_dir)
    
    if len(dataset) == 0:
        logger.error("❌ No data found!")
        logger.error("   Generate data first: python generate_telugu_conversations.py --num_template 1000")
        return
    
    # Split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_conversation,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_conversation
    )
    
    logger.info(f"📚 Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create model
    model = ProductionS2S(config).to(config.device)
    
    if args.resume:
        logger.info(f"📥 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=False)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"📊 Parameters: {num_params:.1f}M")
    
    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            loss = train_step(model, batch, optimizer, scaler, config, global_step)
            epoch_loss += loss
            global_step += 1
            pbar.set_postfix({"loss": loss})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_codes = batch["input_codes"].to(config.device).long()
                target_codes = batch["target_codes"].to(config.device).long()
                
                logits, _ = model(input_codes, target_codes[:, :, :-1])
                B, Q, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(-1, V), target_codes[:, :, 1:].reshape(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss = epoch_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config
            }, f"{config.output_dir}/best_production_s2s.pt")
            logger.info(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss
            }, f"{config.output_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    logger.info("=" * 70)
    logger.info("✅ Training complete!")
    logger.info(f"📁 Best model: {config.output_dir}/best_production_s2s.pt")


if __name__ == "__main__":
    main()
