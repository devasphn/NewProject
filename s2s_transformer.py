"""
Telugu S2S Streaming Transformer
Direct speech-to-speech model with emotion control and ultra-low latency
Based on GPT architecture with Conformer encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
from einops import rearrange, repeat
from dataclasses import dataclass
import numpy as np

# Flash Attention 2 import (if available)
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Flash Attention not available, using standard attention")

@dataclass
class S2SConfig:
    """Configuration for S2S model"""
    # Model dimensions - MUST be divisible: hidden_dim % num_heads == 0
    hidden_dim: int = 512  # For POC - smaller model
    num_heads: int = 8     # 512/8 = 64 head_dim (clean division!)
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    ffn_dim: int = 2048    # 4x hidden_dim
    
    # Conformer specific
    conv_kernel_size: int = 31
    conv_expansion_factor: int = 2
    
    # Streaming config
    chunk_size: int = 20  # 100ms at 200Hz
    lookahead: int = 4    # 20ms lookahead
    
    # Vocabulary
    vocab_size: int = 1024  # Codec codebook size
    num_quantizers: int = 8
    max_seq_len: int = 4096
    
    # Special tokens
    num_speakers: int = 4
    num_emotions: int = 9  # Including Telugu accents
    
    # Training
    dropout: float = 0.1
    use_flash_attn: bool = True
    
    def __post_init__(self):
        """Validate config after initialization"""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.hidden_dim % self.num_quantizers == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_quantizers ({self.num_quantizers})"

class RotaryPositionalEmbedding(nn.Module):
    """RoPE for better position encoding in streaming"""
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

def rotate_half(x):
    """Helper for RoPE"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ConformerConvModule(nn.Module):
    """Conformer convolution module for local feature extraction"""
    def __init__(self, channels, kernel_size=31, expansion_factor=2):
        super().__init__()
        inner_dim = channels * expansion_factor
        
        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Linear(channels, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=inner_dim
        )
        self.norm = nn.BatchNorm1d(inner_dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Linear(inner_dim, channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, time, channels]
        Returns:
            x: [batch, time, channels]
        """
        residual = x
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x, gate = x.chunk(2, dim=-1)
        
        # Depthwise conv
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, T, C]
        
        x = self.activation(x)
        x = x * torch.sigmoid(gate)  # Gated activation
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return residual + x

class MultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention support"""
    def __init__(self, dim, num_heads, dropout=0.1, use_flash=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_flash = use_flash and FLASH_AVAILABLE
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout
    
    def forward(self, x, mask=None, use_cache=False, cache=None):
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        if use_cache and cache is not None:
            # Concatenate with cached K, V (if they exist)
            if cache.get('k') is not None:
                k = torch.cat([cache['k'], k], dim=1)
                v = torch.cat([cache['v'], v], dim=1)
            cache['k'] = k
            cache['v'] = v
        
        if self.use_flash and mask is None:
            # Use Flash Attention 2
            q = q.transpose(1, 2)  # [B, H, T, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            out = out.transpose(1, 2).reshape(B, T, C)
        else:
            # Standard attention
            q = q.transpose(1, 2)  # [B, H, T, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, T, C)
        
        return self.out(out)

class ConformerBlock(nn.Module):
    """Conformer encoder block with conv + attention"""
    def __init__(self, config: S2SConfig):
        super().__init__()
        
        # First half FFN
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            config.hidden_dim, config.num_heads, 
            config.dropout, config.use_flash_attn
        )
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        
        # Convolution module
        self.conv = ConformerConvModule(
            config.hidden_dim, config.conv_kernel_size, 
            config.conv_expansion_factor
        )
        
        # Second half FFN
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer scale for training stability
        self.layer_scale = nn.Parameter(torch.ones(config.hidden_dim) * 0.1)
    
    def forward(self, x, mask=None):
        # Half FFN
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        x = x + self.self_attn(self.attn_norm(x), mask)
        
        # Convolution
        x = x + self.conv(x)
        
        # Half FFN
        x = x + 0.5 * self.ffn2(x)
        
        # Layer scale
        x = x * self.layer_scale
        
        return x

class TransformerBlock(nn.Module):
    """Standard transformer decoder block with KV cache"""
    def __init__(self, config: S2SConfig):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(
            config.hidden_dim, config.num_heads,
            config.dropout, config.use_flash_attn
        )
        self.cross_attn = MultiHeadAttention(
            config.hidden_dim, config.num_heads,
            config.dropout, config.use_flash_attn
        )
        
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ln3 = nn.LayerNorm(config.hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None,
                use_cache=False, cache=None):
        # Self-attention with cache
        x = x + self.self_attn(self.ln1(x), self_mask, use_cache, cache)
        
        # Cross-attention
        x = x + self.cross_attn(self.ln2(x), cross_mask)
        
        # FFN
        x = x + self.ffn(self.ln3(x))
        
        return x

class EmotionEmbedding(nn.Module):
    """Learnable emotion and speaker embeddings"""
    def __init__(self, config: S2SConfig):
        super().__init__()
        
        # Emotion embeddings
        self.emotion_embed = nn.Embedding(config.num_emotions, config.hidden_dim)
        
        # Speaker embeddings  
        self.speaker_embed = nn.Embedding(config.num_speakers, config.hidden_dim)
        
        # Style mixing layer
        self.style_mix = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Initialize with small values for stability
        nn.init.normal_(self.emotion_embed.weight, std=0.02)
        nn.init.normal_(self.speaker_embed.weight, std=0.02)
    
    def forward(self, emotion_id, speaker_id):
        """
        Args:
            emotion_id: [batch] emotion IDs
            speaker_id: [batch] speaker IDs
        
        Returns:
            style: [batch, hidden_dim] combined style embedding
        """
        emotion = self.emotion_embed(emotion_id)
        speaker = self.speaker_embed(speaker_id)
        
        # Mix emotion and speaker
        combined = torch.cat([emotion, speaker], dim=-1)
        style = self.style_mix(combined)
        
        return style

class TeluguS2STransformer(nn.Module):
    """
    Complete S2S transformer for Telugu speech
    Streaming-capable with <150ms latency target
    """
    def __init__(self, config: S2SConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings for codec codes
        self.token_embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_dim // config.num_quantizers)
            for _ in range(config.num_quantizers)
        ])
        
        # Position encoding
        self.rope = RotaryPositionalEmbedding(config.hidden_dim // config.num_heads)
        
        # Emotion and speaker control
        self.style_embed = EmotionEmbedding(config)
        
        # Conformer encoder
        self.encoder_layers = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.num_encoder_layers)
        ])
        
        # Transformer decoder with KV cache
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_decoder_layers)
        ])
        
        # Output heads for each quantizer
        self.output_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.vocab_size)
            for _ in range(config.num_quantizers)
        ])
        
        # Layernorm
        self.encoder_norm = nn.LayerNorm(config.hidden_dim)
        self.decoder_norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(self, input_codes, speaker_id, emotion_id, mask=None):
        """
        Encode input speech codes
        
        Args:
            input_codes: [B, Q, T] codec codes  
            speaker_id: [B] speaker IDs
            emotion_id: [B] emotion IDs
            mask: [B, T] attention mask
        
        Returns:
            encoder_output: [B, T, D] encoded representation
        """
        B, Q, T = input_codes.shape
        
        # Embed codes from each quantizer
        embeddings = []
        for q in range(Q):
            emb = self.token_embed[q](input_codes[:, q])  # [B, T, D/Q]
            embeddings.append(emb)
        
        # Combine embeddings
        x = torch.cat(embeddings, dim=-1)  # [B, T, D]
        
        # Add style embedding
        style = self.style_embed(emotion_id, speaker_id)  # [B, D]
        x = x + style.unsqueeze(1)  # Broadcast to all positions
        
        # Apply RoPE
        cos, sin = self.rope(T, x.device)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        x = self.encoder_norm(x)
        return x
    
    def decode(self, target_codes, encoder_output, speaker_id, emotion_id,
               self_mask=None, cross_mask=None, use_cache=False, cache=None):
        """
        Decode to target speech codes
        
        Args:
            target_codes: [B, Q, T'] target codec codes
            encoder_output: [B, T, D] encoder output
            speaker_id: [B] speaker IDs
            emotion_id: [B] emotion IDs
            self_mask: [B, T', T'] causal mask
            cross_mask: [B, T', T] encoder mask
            use_cache: Whether to use KV cache
            cache: KV cache dict
        
        Returns:
            logits: List of [B, T', V] logits for each quantizer
        """
        B, Q, T = target_codes.shape
        
        # Embed target codes
        embeddings = []
        for q in range(Q):
            emb = self.token_embed[q](target_codes[:, q])
            embeddings.append(emb)
        
        x = torch.cat(embeddings, dim=-1)
        
        # Add style
        style = self.style_embed(emotion_id, speaker_id)
        x = x + style.unsqueeze(1)
        
        # Initialize cache if needed
        if use_cache and cache is None:
            cache = [{'k': None, 'v': None} for _ in range(self.config.num_decoder_layers)]
        
        # Decoder layers
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = cache[i] if use_cache else None
            x = layer(x, encoder_output, self_mask, cross_mask, use_cache, layer_cache)
        
        x = self.decoder_norm(x)
        
        # Output predictions for each quantizer
        logits = []
        for q in range(Q):
            logits.append(self.output_heads[q](x))
        
        return logits, cache
    
    @torch.no_grad()
    def generate_streaming(self, input_codes, speaker_id, emotion_id,
                          max_new_tokens=200, temperature=0.8, top_p=0.95):
        """
        Streaming generation with KV cache
        
        Args:
            input_codes: [1, Q, T] input codec codes
            speaker_id: [1] speaker ID
            emotion_id: [1] emotion ID
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Yields:
            codes: [1, Q, 1] generated code chunks
        """
        # Encode input once
        encoder_output = self.encode(input_codes, speaker_id, emotion_id)
        
        # Initialize with start token
        output_codes = torch.zeros(1, self.config.num_quantizers, 1,
                                  dtype=torch.long, device=input_codes.device)
        cache = None
        
        for _ in range(max_new_tokens):
            # Decode with cache
            logits, cache = self.decode(
                output_codes[:, :, -1:],  # Only last token
                encoder_output,
                speaker_id,
                emotion_id,
                use_cache=True,
                cache=cache
            )
            
            # Sample next tokens
            next_codes = []
            for q in range(self.config.num_quantizers):
                logits_q = logits[q][:, -1, :] / temperature
                
                # Nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_q, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits_q[indices_to_remove] = -float('inf')
                
                probs = F.softmax(logits_q, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_codes.append(next_token)
            
            next_codes = torch.stack(next_codes, dim=1)  # [1, Q, 1]
            output_codes = torch.cat([output_codes, next_codes], dim=-1)
            
            # Yield generated codes
            yield next_codes
    
    def forward(self, input_codes, target_codes, speaker_id, emotion_id):
        """
        Training forward pass
        
        Args:
            input_codes: [B, Q, T_in] input codec codes
            target_codes: [B, Q, T_out] target codec codes
            speaker_id: [B] speaker IDs
            emotion_id: [B] emotion IDs
        
        Returns:
            loss: Scalar loss value
        """
        # Encode
        encoder_output = self.encode(input_codes, speaker_id, emotion_id)
        
        # Prepare decoder input (shift right)
        decoder_input = F.pad(target_codes[:, :, :-1], (1, 0))
        
        # Create causal mask
        T_out = decoder_input.size(-1)
        self_mask = torch.triu(torch.ones(T_out, T_out, device=input_codes.device), diagonal=1) == 0
        
        # Decode
        logits, _ = self.decode(decoder_input, encoder_output, speaker_id, emotion_id, self_mask)
        
        # Compute loss for each quantizer
        loss = 0
        for q in range(self.config.num_quantizers):
            loss += F.cross_entropy(
                logits[q].reshape(-1, self.config.vocab_size),
                target_codes[:, q].reshape(-1)
            )
        
        return loss / self.config.num_quantizers


# Emotion IDs mapping
EMOTION_IDS = {
    "neutral": 0,
    "happy": 1,
    "laugh": 2,
    "excited": 3,
    "empathy": 4,
    "surprise": 5,
    "thinking": 6,
    "telugu_heavy": 7,  # Heavy Telugu accent
    "telugu_mild": 8,   # Mild Telugu accent
}

# Speaker IDs
SPEAKER_IDS = {
    "male_young": 0,
    "male_mature": 1,
    "female_young": 2,
    "female_professional": 3,
}

if __name__ == "__main__":
    # Test model
    config = S2SConfig()
    model = TeluguS2STransformer(config)
    
    print(f"Model initialized")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    input_len = 50  # 250ms at 200Hz
    output_len = 60  # 300ms at 200Hz
    
    input_codes = torch.randint(0, config.vocab_size, (batch_size, config.num_quantizers, input_len))
    target_codes = torch.randint(0, config.vocab_size, (batch_size, config.num_quantizers, output_len))
    speaker_id = torch.tensor([0, 1])  # Different speakers
    emotion_id = torch.tensor([EMOTION_IDS["happy"], EMOTION_IDS["laugh"]])
    
    loss = model(input_codes, target_codes, speaker_id, emotion_id)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test streaming generation
    print("\nTesting streaming generation:")
    input_codes = torch.randint(0, config.vocab_size, (1, config.num_quantizers, 20))
    speaker_id = torch.tensor([SPEAKER_IDS["female_young"]])
    emotion_id = torch.tensor([EMOTION_IDS["excited"]])
    
    generated_chunks = []
    for i, chunk in enumerate(model.generate_streaming(
        input_codes, speaker_id, emotion_id, max_new_tokens=10
    )):
        generated_chunks.append(chunk)
        print(f"  Chunk {i}: {chunk.shape}")
        if i >= 5:  # Generate 6 chunks for demo
            break
    
    print(f"Generated {len(generated_chunks)} chunks")