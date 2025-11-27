#!/usr/bin/env python3
"""
Train S2S Transformer for Telugu Conversation
=============================================

This trains your S2S model to generate RESPONSES, not just reconstruct.

Input: User's audio codes (question)
Output: Assistant's audio codes (answer)

This is the KEY missing piece for a real voice assistant!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/telugu_conversations"
    codec_path: str = "best_codec.pt"
    
    # Model (use same config as your trained S2S)
    hidden_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_quantizers: int = 8
    vocab_size: int = 1024
    max_seq_len: int = 2048
    
    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    
    # Checkpointing
    save_every: int = 5
    output_dir: str = "checkpoints/s2s_conversation"
    
    # Device
    device: str = "cuda"


class ConversationDataset(Dataset):
    """Dataset of (question_codes, answer_codes) pairs"""
    
    def __init__(self, data_dir: str, max_len: int = 1024):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        
        # Find all conversation pairs
        self.pairs = []
        for pair_dir in sorted(self.data_dir.glob("pair_*")):
            q_codes_path = pair_dir / "question_codes.pt"
            a_codes_path = pair_dir / "answer_codes.pt"
            
            if q_codes_path.exists() and a_codes_path.exists():
                self.pairs.append((q_codes_path, a_codes_path))
        
        logger.info(f"📚 Found {len(self.pairs)} conversation pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        q_path, a_path = self.pairs[idx]
        
        # Load codes
        q_codes = torch.load(q_path)  # [1, Q, T1]
        a_codes = torch.load(a_path)  # [1, Q, T2]
        
        # Remove batch dimension
        q_codes = q_codes.squeeze(0)  # [Q, T1]
        a_codes = a_codes.squeeze(0)  # [Q, T2]
        
        # Truncate if too long
        if q_codes.shape[1] > self.max_len:
            q_codes = q_codes[:, :self.max_len]
        if a_codes.shape[1] > self.max_len:
            a_codes = a_codes[:, :self.max_len]
        
        return {
            "input_codes": q_codes,   # Question audio codes
            "target_codes": a_codes   # Answer audio codes
        }


def collate_fn(batch):
    """Pad sequences to same length within batch"""
    input_codes = [item["input_codes"] for item in batch]
    target_codes = [item["target_codes"] for item in batch]
    
    # Get max lengths
    max_input_len = max(c.shape[1] for c in input_codes)
    max_target_len = max(c.shape[1] for c in target_codes)
    
    num_quantizers = input_codes[0].shape[0]
    
    # Pad
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(input_codes, target_codes):
        # Pad input
        pad_inp = torch.zeros(num_quantizers, max_input_len, dtype=inp.dtype)
        pad_inp[:, :inp.shape[1]] = inp
        padded_inputs.append(pad_inp)
        
        # Pad target
        pad_tgt = torch.zeros(num_quantizers, max_target_len, dtype=tgt.dtype)
        pad_tgt[:, :tgt.shape[1]] = tgt
        padded_targets.append(pad_tgt)
    
    return {
        "input_codes": torch.stack(padded_inputs),   # [B, Q, T1]
        "target_codes": torch.stack(padded_targets)  # [B, Q, T2]
    }


class ConversationS2S(nn.Module):
    """
    S2S Transformer for Conversation
    
    Difference from reconstruction:
    - Encoder processes question audio codes
    - Decoder generates answer audio codes (DIFFERENT from input!)
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings for each quantizer
        embed_dim = config.hidden_dim // config.num_quantizers
        self.token_embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, embed_dim)
            for _ in range(config.num_quantizers)
        ])
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        
        # Output heads for each quantizer
        self.output_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim // config.num_quantizers, config.vocab_size)
            for _ in range(config.num_quantizers)
        ])
        
    def embed_codes(self, codes):
        """Embed audio codes: [B, Q, T] -> [B, T, D]"""
        B, Q, T = codes.shape
        
        embeddings = []
        for q in range(Q):
            emb = self.token_embed[q](codes[:, q])  # [B, T, D/Q]
            embeddings.append(emb)
        
        return torch.cat(embeddings, dim=-1)  # [B, T, D]
    
    def forward(self, input_codes, target_codes):
        """
        Forward pass for training
        
        Args:
            input_codes: Question codes [B, Q, T1]
            target_codes: Answer codes [B, Q, T2] (for teacher forcing)
        
        Returns:
            logits: [B, Q, T2, vocab_size]
        """
        B, Q, T1 = input_codes.shape
        _, _, T2 = target_codes.shape
        
        # Embed inputs
        enc_input = self.embed_codes(input_codes)  # [B, T1, D]
        dec_input = self.embed_codes(target_codes)  # [B, T2, D]
        
        # Add positional encoding
        enc_input = enc_input + self.pos_embed[:, :T1]
        dec_input = dec_input + self.pos_embed[:, :T2]
        
        # Create causal mask for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T2).to(input_codes.device)
        
        # Encode question
        memory = self.encoder(enc_input)  # [B, T1, D]
        
        # Decode answer
        dec_output = self.decoder(dec_input, memory, tgt_mask=causal_mask)  # [B, T2, D]
        
        # Split and project to logits
        chunk_size = self.config.hidden_dim // Q
        logits = []
        for q in range(Q):
            chunk = dec_output[:, :, q*chunk_size:(q+1)*chunk_size]  # [B, T2, D/Q]
            logit = self.output_heads[q](chunk)  # [B, T2, vocab_size]
            logits.append(logit)
        
        return torch.stack(logits, dim=1)  # [B, Q, T2, vocab_size]
    
    @torch.no_grad()
    def generate(self, input_codes, max_len: int = 500, temperature: float = 0.8):
        """
        Generate answer codes given question codes
        
        Args:
            input_codes: Question codes [1, Q, T1]
            max_len: Maximum output length
        
        Returns:
            output_codes: Answer codes [1, Q, T2]
        """
        self.eval()
        B, Q, T1 = input_codes.shape
        device = input_codes.device
        
        # Embed and encode question
        enc_input = self.embed_codes(input_codes) + self.pos_embed[:, :T1]
        memory = self.encoder(enc_input)
        
        # Start with first token (could be special BOS token, here we use 0)
        output_codes = torch.zeros(B, Q, 1, dtype=torch.long, device=device)
        
        for t in range(max_len):
            # Embed current output
            dec_input = self.embed_codes(output_codes) + self.pos_embed[:, :t+1]
            
            # Decode
            causal_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=causal_mask)
            
            # Get next token for each quantizer
            next_tokens = []
            chunk_size = self.config.hidden_dim // Q
            
            for q in range(Q):
                chunk = dec_output[:, -1, q*chunk_size:(q+1)*chunk_size]  # [B, D/Q]
                logit = self.output_heads[q](chunk)  # [B, vocab_size]
                
                # Sample
                if temperature > 0:
                    probs = F.softmax(logit / temperature, dim=-1)
                    token = torch.multinomial(probs, 1)  # [B, 1]
                else:
                    token = logit.argmax(dim=-1, keepdim=True)
                
                next_tokens.append(token)
            
            # Append to output
            next_tokens = torch.stack(next_tokens, dim=1)  # [B, Q, 1]
            output_codes = torch.cat([output_codes, next_tokens], dim=-1)
        
        return output_codes


def train_epoch(model, dataloader, optimizer, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        input_codes = batch["input_codes"].to(config.device).long()
        target_codes = batch["target_codes"].to(config.device).long()
        
        # Forward pass (teacher forcing)
        logits = model(input_codes, target_codes[:, :, :-1])  # [B, Q, T-1, vocab]
        
        # Compute loss
        B, Q, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            target_codes[:, :, 1:].reshape(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, config):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_codes = batch["input_codes"].to(config.device).long()
            target_codes = batch["target_codes"].to(config.device).long()
            
            logits = model(input_codes, target_codes[:, :, :-1])
            
            B, Q, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                target_codes[:, :, 1:].reshape(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train S2S for conversation")
    parser.add_argument("--data_dir", default="data/telugu_conversations")
    parser.add_argument("--codec", default="best_codec.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="checkpoints/s2s_conversation")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    config = TrainConfig(
        data_dir=args.data_dir,
        codec_path=args.codec,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output
    )
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🎯 Training S2S for CONVERSATION")
    logger.info("=" * 60)
    logger.info(f"Data: {config.data_dir}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    # Create dataset
    dataset = ConversationDataset(config.data_dir)
    
    if len(dataset) == 0:
        logger.error("❌ No data found! Generate data first:")
        logger.error("   python generate_telugu_conversations.py --num_template 100")
        return
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"📚 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = ConversationS2S(config).to(config.device)
    
    # Load pretrained S2S weights if available
    if args.resume:
        logger.info(f"📥 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"📊 Model parameters: {num_params:.1f}M")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config, epoch)
        val_loss = validate(model, val_loader, config)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, f"{config.output_dir}/best_conversation_s2s.pt")
            logger.info(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f"{config.output_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    logger.info("=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"📁 Best model: {config.output_dir}/best_conversation_s2s.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
