#!/usr/bin/env python3
"""
Conversation Context Manager for Telugu S2S
Manages conversation history, user preferences, and context-aware responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
import time
import pickle
from pathlib import Path

@dataclass
class ConversationTurn:
    """Single conversation turn with metadata"""
    user_audio_codes: torch.Tensor
    bot_audio_codes: torch.Tensor
    emotion: str
    speaker: str
    timestamp: float
    user_text: Optional[str] = None  # From ASR if available
    bot_text: Optional[str] = None   # From ASR if available
    sentiment: Optional[float] = None  # -1 to 1
    topic: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "emotion": self.emotion,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "user_text": self.user_text,
            "bot_text": self.bot_text,
            "sentiment": self.sentiment,
            "topic": self.topic
        }

class ContextMemory(nn.Module):
    """
    Neural context memory with attention mechanism
    Maintains and retrieves relevant conversation context
    """
    
    def __init__(self, hidden_dim: int = 768, memory_size: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Memory encoder
        self.turn_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism for memory retrieval
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Topic classifier
        self.topic_classifier = nn.Linear(hidden_dim, 10)  # 10 common topics
        self.topics = [
            "greeting", "weather", "news", "sports", "entertainment",
            "technology", "health", "food", "travel", "general"
        ]
        
        # Sentiment analyzer
        self.sentiment_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output -1 to 1
        )
    
    def encode_turn(self, turn_codes: torch.Tensor) -> torch.Tensor:
        """Encode a conversation turn into memory representation"""
        # Average pool if needed
        if turn_codes.dim() == 3:
            turn_codes = turn_codes.mean(dim=1)
        
        # Encode through LSTM
        output, (hidden, _) = self.turn_encoder(turn_codes.unsqueeze(1))
        return output.squeeze(1)
    
    def retrieve_context(
        self,
        query: torch.Tensor,
        memory: List[torch.Tensor],
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant context from memory using attention
        
        Args:
            query: Current input representation
            memory: List of past turn representations
            top_k: Number of most relevant turns to retrieve
        
        Returns:
            context: Weighted context representation
            attention_weights: Attention scores over memory
        """
        if not memory:
            return torch.zeros_like(query), torch.zeros(1, 1)
        
        # Stack memory
        memory_tensor = torch.stack(memory[-self.memory_size:])  # [mem_size, hidden_dim]
        
        # Compute attention
        Q = self.query_projection(query.unsqueeze(0))  # [1, hidden_dim]
        K = self.key_projection(memory_tensor)  # [mem_size, hidden_dim]
        V = self.value_projection(memory_tensor)  # [mem_size, hidden_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Get top-k relevant memories
        if top_k < len(memory):
            _, top_indices = torch.topk(attention_weights.squeeze(), top_k)
            masked_weights = torch.zeros_like(attention_weights)
            masked_weights[0, top_indices] = attention_weights[0, top_indices]
            attention_weights = masked_weights / masked_weights.sum()
        
        # Weighted context
        context = torch.matmul(attention_weights, V).squeeze(0)
        
        return context, attention_weights
    
    def fuse_with_context(
        self,
        current: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Fuse current input with retrieved context"""
        combined = torch.cat([current, context], dim=-1)
        return self.context_fusion(combined)
    
    def analyze_sentiment(self, representation: torch.Tensor) -> float:
        """Analyze sentiment of the representation"""
        sentiment = self.sentiment_analyzer(representation)
        return sentiment.item()
    
    def classify_topic(self, representation: torch.Tensor) -> str:
        """Classify the topic of conversation"""
        logits = self.topic_classifier(representation)
        topic_idx = torch.argmax(logits, dim=-1).item()
        return self.topics[topic_idx]

class ConversationContextManager:
    """
    Manages conversation context, history, and user preferences
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        model_hidden_dim: int = 768,
        save_dir: str = "context_data/"
    ):
        self.max_turns = max_turns
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Conversation storage per session
        self.conversations: Dict[str, deque] = {}
        
        # User preferences per session
        self.user_preferences: Dict[str, Dict] = {}
        
        # Context memory model
        self.memory_model = ContextMemory(
            hidden_dim=model_hidden_dim,
            memory_size=max_turns
        )
        
        # Memory representations cache
        self.memory_cache: Dict[str, List[torch.Tensor]] = {}
        
        # Statistics
        self.stats = {
            "total_turns": 0,
            "topics": {},
            "average_sentiment": 0,
            "emotion_distribution": {}
        }
    
    def add_turn(
        self,
        session_id: str,
        user_codes: torch.Tensor,
        bot_codes: torch.Tensor,
        emotion: str,
        speaker: str,
        user_text: Optional[str] = None,
        bot_text: Optional[str] = None
    ) -> ConversationTurn:
        """Add a new conversation turn"""
        
        # Initialize session if new
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_turns)
            self.memory_cache[session_id] = []
            self.user_preferences[session_id] = {
                "preferred_emotion": "neutral",
                "preferred_speaker": speaker,
                "interaction_count": 0,
                "session_start": time.time()
            }
        
        # Encode turn for memory
        turn_representation = self.memory_model.encode_turn(user_codes)
        
        # Analyze turn
        sentiment = self.memory_model.analyze_sentiment(turn_representation)
        topic = self.memory_model.classify_topic(turn_representation)
        
        # Create turn object
        turn = ConversationTurn(
            user_audio_codes=user_codes,
            bot_audio_codes=bot_codes,
            emotion=emotion,
            speaker=speaker,
            timestamp=time.time(),
            user_text=user_text,
            bot_text=bot_text,
            sentiment=sentiment,
            topic=topic
        )
        
        # Store turn
        self.conversations[session_id].append(turn)
        self.memory_cache[session_id].append(turn_representation)
        
        # Update statistics
        self._update_statistics(turn)
        
        # Update preferences
        self.user_preferences[session_id]["interaction_count"] += 1
        
        return turn
    
    def get_context(
        self,
        session_id: str,
        current_input: torch.Tensor,
        include_last_n: int = 3
    ) -> Dict[str, Any]:
        """
        Get relevant context for current interaction
        
        Args:
            session_id: Session identifier
            current_input: Current input representation
            include_last_n: Number of recent turns to include
        
        Returns:
            Dictionary with context information
        """
        if session_id not in self.conversations:
            return {
                "has_context": False,
                "context_embedding": torch.zeros_like(current_input),
                "recent_turns": [],
                "preferences": {}
            }
        
        # Get memory representations
        memory = self.memory_cache.get(session_id, [])
        
        # Retrieve relevant context using attention
        context_embedding, attention_weights = self.memory_model.retrieve_context(
            current_input,
            memory,
            top_k=min(3, len(memory))
        )
        
        # Fuse with current input
        enhanced_input = self.memory_model.fuse_with_context(
            current_input,
            context_embedding
        )
        
        # Get recent turns
        recent_turns = list(self.conversations[session_id])[-include_last_n:]
        
        # Analyze conversation flow
        topics = [turn.topic for turn in recent_turns if turn.topic]
        emotions = [turn.emotion for turn in recent_turns]
        sentiments = [turn.sentiment for turn in recent_turns if turn.sentiment]
        
        return {
            "has_context": True,
            "context_embedding": enhanced_input,
            "attention_weights": attention_weights,
            "recent_turns": [turn.to_dict() for turn in recent_turns],
            "recent_topics": topics,
            "recent_emotions": emotions,
            "average_sentiment": np.mean(sentiments) if sentiments else 0,
            "preferences": self.user_preferences.get(session_id, {}),
            "conversation_length": len(self.conversations[session_id])
        }
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        if session_id not in self.conversations:
            return {"error": "Session not found"}
        
        turns = list(self.conversations[session_id])
        preferences = self.user_preferences.get(session_id, {})
        
        # Analyze conversation
        topics = {}
        emotions = {}
        sentiments = []
        
        for turn in turns:
            # Count topics
            if turn.topic:
                topics[turn.topic] = topics.get(turn.topic, 0) + 1
            
            # Count emotions
            emotions[turn.emotion] = emotions.get(turn.emotion, 0) + 1
            
            # Collect sentiments
            if turn.sentiment:
                sentiments.append(turn.sentiment)
        
        # Calculate duration
        if turns:
            duration = turns[-1].timestamp - turns[0].timestamp
        else:
            duration = 0
        
        return {
            "session_id": session_id,
            "total_turns": len(turns),
            "duration_seconds": duration,
            "dominant_topic": max(topics, key=topics.get) if topics else None,
            "topic_distribution": topics,
            "emotion_distribution": emotions,
            "average_sentiment": np.mean(sentiments) if sentiments else 0,
            "sentiment_trend": sentiments,
            "preferences": preferences,
            "session_start": preferences.get("session_start"),
            "interaction_count": preferences.get("interaction_count", 0)
        }
    
    def suggest_response_style(
        self,
        session_id: str,
        current_sentiment: float
    ) -> Dict[str, Any]:
        """
        Suggest response style based on context
        
        Args:
            session_id: Session identifier
            current_sentiment: Current detected sentiment
        
        Returns:
            Suggested emotion and response style
        """
        context = self.get_conversation_summary(session_id)
        
        # Sentiment-based suggestion
        if current_sentiment < -0.5:
            # User seems upset
            suggested_emotion = "empathy"
            response_style = "supportive"
        elif current_sentiment > 0.5:
            # User seems happy
            suggested_emotion = "happy"
            response_style = "enthusiastic"
        else:
            # Neutral or mixed
            suggested_emotion = "neutral"
            response_style = "conversational"
        
        # Adjust based on conversation history
        if context.get("average_sentiment", 0) < -0.3:
            # Overall negative conversation
            suggested_emotion = "empathy"
            response_style = "gentle"
        
        # Check for repeated topics
        topic_dist = context.get("topic_distribution", {})
        if topic_dist:
            dominant_topic = max(topic_dist, key=topic_dist.get)
            if topic_dist[dominant_topic] > 3:
                # User is focused on specific topic
                response_style = "focused"
        
        return {
            "suggested_emotion": suggested_emotion,
            "response_style": response_style,
            "confidence": 0.8,
            "reasoning": f"Based on sentiment ({current_sentiment:.2f}) and history"
        }
    
    def save_session(self, session_id: str):
        """Save session data to disk"""
        if session_id not in self.conversations:
            return
        
        save_path = self.save_dir / f"session_{session_id}.pkl"
        
        data = {
            "turns": list(self.conversations[session_id]),
            "preferences": self.user_preferences.get(session_id, {}),
            "summary": self.get_conversation_summary(session_id)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Session {session_id} saved to {save_path}")
    
    def load_session(self, session_id: str) -> bool:
        """Load session data from disk"""
        save_path = self.save_dir / f"session_{session_id}.pkl"
        
        if not save_path.exists():
            return False
        
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore conversation
        self.conversations[session_id] = deque(data["turns"], maxlen=self.max_turns)
        self.user_preferences[session_id] = data["preferences"]
        
        # Rebuild memory cache
        self.memory_cache[session_id] = []
        for turn in data["turns"]:
            if hasattr(turn, 'user_audio_codes'):
                representation = self.memory_model.encode_turn(turn.user_audio_codes)
                self.memory_cache[session_id].append(representation)
        
        print(f"Session {session_id} loaded from {save_path}")
        return True
    
    def _update_statistics(self, turn: ConversationTurn):
        """Update global statistics"""
        self.stats["total_turns"] += 1
        
        # Update topic distribution
        if turn.topic:
            self.stats["topics"][turn.topic] = \
                self.stats["topics"].get(turn.topic, 0) + 1
        
        # Update emotion distribution
        self.stats["emotion_distribution"][turn.emotion] = \
            self.stats["emotion_distribution"].get(turn.emotion, 0) + 1
        
        # Update average sentiment
        if turn.sentiment:
            alpha = 0.1  # Exponential moving average
            self.stats["average_sentiment"] = \
                (1 - alpha) * self.stats["average_sentiment"] + alpha * turn.sentiment
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get global conversation statistics"""
        return {
            **self.stats,
            "active_sessions": len(self.conversations),
            "total_sessions": len(self.conversations) + len(list(self.save_dir.glob("session_*.pkl")))
        }


if __name__ == "__main__":
    # Test context manager
    manager = ConversationContextManager(max_turns=10)
    
    # Simulate conversation
    session_id = "test_session"
    
    # Add some turns
    for i in range(5):
        user_codes = torch.randn(1, 10, 768)
        bot_codes = torch.randn(1, 10, 768)
        
        emotion = ["neutral", "happy", "excited", "laugh", "thinking"][i % 5]
        
        turn = manager.add_turn(
            session_id=session_id,
            user_codes=user_codes,
            bot_codes=bot_codes,
            emotion=emotion,
            speaker="female_young",
            user_text=f"User message {i}",
            bot_text=f"Bot response {i}"
        )
        
        print(f"Turn {i}: Topic={turn.topic}, Sentiment={turn.sentiment:.2f}")
    
    # Get context
    current_input = torch.randn(768)
    context = manager.get_context(session_id, current_input)
    print(f"\nContext retrieved: {context['has_context']}")
    print(f"Recent emotions: {context['recent_emotions']}")
    print(f"Average sentiment: {context['average_sentiment']:.2f}")
    
    # Get summary
    summary = manager.get_conversation_summary(session_id)
    print(f"\nConversation summary:")
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Topic distribution: {summary['topic_distribution']}")
    print(f"  Emotion distribution: {summary['emotion_distribution']}")
    
    # Suggest response style
    suggestion = manager.suggest_response_style(session_id, current_sentiment=0.3)
    print(f"\nSuggested response:")
    print(f"  Emotion: {suggestion['suggested_emotion']}")
    print(f"  Style: {suggestion['response_style']}")
    
    # Save session
    manager.save_session(session_id)
    print(f"\nSession saved successfully")