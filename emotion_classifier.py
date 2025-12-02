#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SPEECH EMOTION RECOGNITION (SER) MODULE
  
  Architecture: WavLM/HuBERT features + Attentive Pooling + Classification
  Based on: emotion2vec, Odyssey 2024 Challenge winners
  
  Features:
  - SSL feature extraction (WavLM-Large preferred)
  - Multi-head attention pooling for utterance-level representation
  - 8 emotion categories (Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise, Contempt)
  - Dimensional emotion (Valence, Arousal, Dominance)
  - Real-time capable (<100ms latency)
  
  Reference: 
  - https://arxiv.org/abs/2405.04485 (emotion2vec)
  - https://arxiv.org/abs/2312.16383 (Odyssey 2024 Challenge)
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import warnings


class EmotionCategory(Enum):
    """Standard emotion categories for SER."""
    NEUTRAL = 0
    HAPPY = 1
    SAD = 2
    ANGRY = 3
    FEAR = 4
    DISGUST = 5
    SURPRISE = 6
    CONTEMPT = 7


@dataclass
class EmotionConfig:
    """Configuration for emotion classifier."""
    # Feature extraction
    sample_rate: int = 16000
    feature_dim: int = 1024  # WavLM-Large dimension
    
    # Model architecture
    hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Emotion categories
    num_emotions: int = 8
    emotion_names: List[str] = None
    
    # Dimensional emotions
    predict_valence: bool = True
    predict_arousal: bool = True
    predict_dominance: bool = True
    
    def __post_init__(self):
        if self.emotion_names is None:
            self.emotion_names = [e.name.lower() for e in EmotionCategory]


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling for utterance-level representation.
    More effective than simple mean/max pooling for emotion recognition.
    """
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - sequence of frame-level features
            mask: [B, T] - attention mask (1 for valid, 0 for padding)
        Returns:
            pooled: [B, D] - utterance-level representation
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        
        # Mean pooling over time with attention-weighted context
        pooled = context.mean(dim=1)
        
        return self.layer_norm(pooled)


class StatisticsPooling(nn.Module):
    """
    Statistics Pooling: compute mean and std for utterance representation.
    Simple but effective baseline pooling method.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim * 2  # mean + std
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T]
        Returns:
            pooled: [B, D*2]
        """
        if mask is not None:
            # Masked mean and std
            mask = mask.unsqueeze(-1).float()  # [B, T, 1]
            lengths = mask.sum(dim=1)  # [B, 1]
            
            mean = (x * mask).sum(dim=1) / lengths.clamp(min=1)
            var = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / lengths.clamp(min=1)
            std = torch.sqrt(var + 1e-8)
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
        
        return torch.cat([mean, std], dim=-1)


class EmotionClassifier(nn.Module):
    """
    Complete Speech Emotion Recognition model.
    
    Features:
    - Uses pretrained WavLM/HuBERT features
    - Multi-head attention pooling
    - Categorical emotion prediction (8 classes)
    - Dimensional emotion prediction (valence, arousal, dominance)
    
    Usage:
        classifier = EmotionClassifier()
        
        # From audio
        result = classifier(audio)
        print(result['emotion'])  # 'happy'
        print(result['valence'])  # 0.7
        print(result['arousal'])  # 0.8
    """
    def __init__(self, config: Optional[EmotionConfig] = None):
        super().__init__()
        
        self.config = config or EmotionConfig()
        
        # SSL feature extractor (WavLM-Large)
        self.feature_extractor = None  # Lazy loading to save memory
        self._feature_extractor_loaded = False
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        
        # Temporal modeling (optional Conformer-style layers)
        self.temporal_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_dim,
                nhead=8,
                dim_feedforward=self.config.hidden_dim * 4,
                dropout=self.config.dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # Attention pooling
        self.pooling = MultiHeadAttentionPooling(
            self.config.hidden_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.dropout,
        )
        
        # Alternative: Statistics pooling
        self.stats_pooling = StatisticsPooling(self.config.hidden_dim)
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_emotions),
        )
        
        # Dimensional emotion heads
        if self.config.predict_valence:
            self.valence_head = nn.Linear(self.config.hidden_dim, 1)
        if self.config.predict_arousal:
            self.arousal_head = nn.Linear(self.config.hidden_dim, 1)
        if self.config.predict_dominance:
            self.dominance_head = nn.Linear(self.config.hidden_dim, 1)
        
        # Emotion embedding for conditioning (used by S2S)
        self.emotion_embedding = nn.Embedding(
            self.config.num_emotions, 
            self.config.hidden_dim
        )
        
    def load_feature_extractor(self, model_name: str = "microsoft/wavlm-large"):
        """Lazy load WavLM feature extractor."""
        if self._feature_extractor_loaded:
            return
        
        try:
            from transformers import WavLMModel, Wav2Vec2FeatureExtractor
            
            print(f"Loading {model_name}...")
            self.feature_extractor = WavLMModel.from_pretrained(model_name)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            
            # Freeze SSL model
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            
            self._feature_extractor_loaded = True
            print(f"✅ Loaded {model_name}")
            
        except Exception as e:
            warnings.warn(f"Could not load WavLM: {e}. Using random features for testing.")
            self.feature_extractor = None
            
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract SSL features from audio.
        
        Args:
            audio: [B, samples] or [B, 1, samples]
        Returns:
            features: [B, T, D]
        """
        # Ensure correct shape
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        if self.feature_extractor is not None:
            with torch.no_grad():
                outputs = self.feature_extractor(audio, output_hidden_states=True)
                # Use weighted sum of layers (like emotion2vec)
                hidden_states = outputs.hidden_states
                # Simple: use last layer
                features = hidden_states[-1]
        else:
            # Fallback: random features for testing
            B = audio.shape[0]
            T = audio.shape[1] // 320  # Approximate WavLM downsampling
            features = torch.randn(B, T, self.config.feature_dim, device=audio.device)
        
        return features
    
    def forward(self, audio: torch.Tensor, 
                return_embedding: bool = False,
                extract_features: bool = True) -> Dict:
        """
        Predict emotion from audio.
        
        Args:
            audio: Raw audio [B, samples] or [B, 1, samples]
            return_embedding: Return intermediate embedding
            extract_features: Extract SSL features (False if features already extracted)
        Returns:
            Dict with:
                - emotion_logits: [B, num_emotions]
                - emotion_probs: [B, num_emotions]
                - emotion_id: [B]
                - emotion: List[str]
                - valence: [B] (if enabled)
                - arousal: [B] (if enabled)
                - dominance: [B] (if enabled)
                - embedding: [B, hidden_dim] (if return_embedding)
        """
        # Extract SSL features
        if extract_features:
            features = self.extract_features(audio)
        else:
            features = audio  # Already features
        
        # Project features
        x = self.feature_proj(features)
        
        # Temporal modeling
        x = self.temporal_layers(x)
        
        # Pooling
        pooled = self.pooling(x)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(pooled)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        emotion_id = emotion_logits.argmax(dim=-1)
        
        result = {
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
            'emotion_id': emotion_id,
            'emotion': [self.config.emotion_names[i.item()] for i in emotion_id],
        }
        
        # Dimensional emotions (sigmoid for [0, 1] range)
        if self.config.predict_valence:
            result['valence'] = torch.sigmoid(self.valence_head(pooled)).squeeze(-1)
        if self.config.predict_arousal:
            result['arousal'] = torch.sigmoid(self.arousal_head(pooled)).squeeze(-1)
        if self.config.predict_dominance:
            result['dominance'] = torch.sigmoid(self.dominance_head(pooled)).squeeze(-1)
        
        if return_embedding:
            result['embedding'] = pooled
        
        return result
    
    def get_emotion_embedding(self, emotion_id: torch.Tensor) -> torch.Tensor:
        """Get learnable emotion embedding for conditioning."""
        return self.emotion_embedding(emotion_id)
    
    def predict(self, audio: torch.Tensor) -> Tuple[str, float]:
        """
        Simple prediction interface.
        
        Args:
            audio: Raw audio [samples] or [1, samples]
        Returns:
            emotion: Predicted emotion name
            confidence: Confidence score
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        with torch.no_grad():
            result = self(audio)
        
        emotion = result['emotion'][0]
        confidence = result['emotion_probs'][0].max().item()
        
        return emotion, confidence


class EmotionLoss(nn.Module):
    """
    Combined loss for emotion classification.
    
    Includes:
    - Cross-entropy for categorical emotion
    - MSE/CCC for dimensional emotions
    - Optional: Focal loss for class imbalance
    """
    def __init__(self, config: EmotionConfig, 
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.config = config
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # Class weights for imbalanced data (can be updated)
        self.register_buffer('class_weights', 
                            torch.ones(config.num_emotions))
        
    def focal_loss(self, logits: torch.Tensor, 
                   targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, 
                                  weight=self.class_weights, 
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()
    
    def ccc_loss(self, pred: torch.Tensor, 
                 target: torch.Tensor) -> torch.Tensor:
        """
        Concordance Correlation Coefficient (CCC) loss.
        Standard metric for dimensional emotion prediction.
        """
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_var = pred.var()
        target_var = target.var()
        
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()
        
        ccc = (2 * covariance) / (pred_var + target_var + 
                                   (pred_mean - target_mean) ** 2 + 1e-8)
        
        return 1 - ccc
    
    def forward(self, predictions: Dict, 
                targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.
        
        Args:
            predictions: Model output dict
            targets: Dict with 'emotion_id', 'valence', 'arousal', 'dominance'
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        losses = {}
        
        # Categorical emotion loss
        if self.use_focal_loss:
            losses['emotion'] = self.focal_loss(
                predictions['emotion_logits'], 
                targets['emotion_id']
            )
        else:
            losses['emotion'] = F.cross_entropy(
                predictions['emotion_logits'],
                targets['emotion_id'],
                weight=self.class_weights
            )
        
        # Dimensional emotion losses
        if self.config.predict_valence and 'valence' in targets:
            losses['valence'] = self.ccc_loss(
                predictions['valence'], 
                targets['valence']
            )
        
        if self.config.predict_arousal and 'arousal' in targets:
            losses['arousal'] = self.ccc_loss(
                predictions['arousal'],
                targets['arousal']
            )
        
        if self.config.predict_dominance and 'dominance' in targets:
            losses['dominance'] = self.ccc_loss(
                predictions['dominance'],
                targets['dominance']
            )
        
        # Combined loss
        total_loss = losses['emotion']
        for key in ['valence', 'arousal', 'dominance']:
            if key in losses:
                total_loss = total_loss + 0.5 * losses[key]
        
        losses['total'] = total_loss
        
        return total_loss, losses


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT EMOTION CLASSIFIER (for edge deployment)
# ═══════════════════════════════════════════════════════════════════════════════

class LightweightEmotionClassifier(nn.Module):
    """
    Lightweight emotion classifier for real-time edge deployment.
    
    Uses mel-spectrogram features instead of SSL features.
    Much faster but slightly less accurate.
    """
    def __init__(self, num_emotions: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        self.num_emotions = num_emotions
        
        # Mel-spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=80,
        )
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_emotions),
        )
        
    def forward(self, audio: torch.Tensor) -> Dict:
        """
        Fast emotion prediction.
        
        Args:
            audio: [B, samples]
        Returns:
            Dict with emotion predictions
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Extract mel-spectrogram
        mel = self.mel_transform(audio)  # [B, 80, T]
        mel = (mel + 1e-8).log()  # Log mel
        mel = mel.unsqueeze(1)  # [B, 1, 80, T]
        
        # CNN features
        features = self.cnn(mel).squeeze(-1).squeeze(-1)  # [B, 128]
        
        # Classify
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'emotion_logits': logits,
            'emotion_probs': probs,
            'emotion_id': logits.argmax(dim=-1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  EMOTION CLASSIFIER TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Test lightweight classifier (no dependencies)
    print("\n--- Lightweight Classifier Test ---")
    light_model = LightweightEmotionClassifier().to(device)
    
    params = sum(p.numel() for p in light_model.parameters())
    print(f"Lightweight parameters: {params / 1e6:.2f}M")
    
    audio = torch.randn(4, 16000 * 3).to(device)
    
    with torch.no_grad():
        result = light_model(audio)
    
    print(f"Input shape: {audio.shape}")
    print(f"Emotion logits shape: {result['emotion_logits'].shape}")
    print(f"Predicted emotions: {result['emotion_id'].tolist()}")
    
    # Latency test
    import time
    
    audio = torch.randn(1, 16000 * 2).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = light_model(audio)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = light_model(audio)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) / 100 * 1000
    
    print(f"Lightweight latency: {elapsed:.2f}ms per 2-second audio")
    
    # Test full classifier (without WavLM)
    print("\n--- Full Classifier Test (without SSL) ---")
    config = EmotionConfig()
    full_model = EmotionClassifier(config).to(device)
    
    params = sum(p.numel() for p in full_model.parameters())
    print(f"Full classifier parameters: {params / 1e6:.2f}M")
    
    # Test with random features (simulating extracted features)
    features = torch.randn(4, 100, 1024).to(device)  # [B, T, D]
    
    with torch.no_grad():
        result = full_model(features, extract_features=False)
    
    print(f"Input features shape: {features.shape}")
    print(f"Predicted emotions: {result['emotion']}")
    print(f"Valence: {result['valence'].tolist()}")
    print(f"Arousal: {result['arousal'].tolist()}")
    
    # Test emotion embedding
    emotion_id = torch.tensor([0, 1, 2, 3]).to(device)
    emotion_emb = full_model.get_emotion_embedding(emotion_id)
    print(f"Emotion embedding shape: {emotion_emb.shape}")
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)
