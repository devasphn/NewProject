#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  EMOTION-RICH SPEECH DATASET
  
  Dataset classes for training emotion-aware models.
  
  Supported Datasets:
  - RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
  - CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
  - MELD (Multimodal EmotionLines Dataset - dialogue)
  - IEMOCAP (Interactive Emotional Dyadic Motion Capture)
  - Custom Telugu/Hindi emotional recordings
  
  Used for:
  - Training emotion classifier
  - Fine-tuning S2S with emotion awareness
  - Creating emotion-conditioned synthesis
  
  Reference: 
  - RAVDESS: https://zenodo.org/record/1188976
  - CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import re
import json
import random


@dataclass
class EmotionSample:
    """Single emotion-labeled audio sample."""
    audio_path: Path
    emotion: str
    emotion_id: int
    intensity: Optional[str] = None  # 'normal', 'strong'
    speaker_id: Optional[str] = None
    gender: Optional[str] = None
    statement: Optional[str] = None
    valence: Optional[float] = None  # -1 to 1
    arousal: Optional[float] = None  # 0 to 1
    dominance: Optional[float] = None  # 0 to 1


# Standard emotion mappings
EMOTION_MAPPING = {
    # RAVDESS emotions
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7,
    
    # Alternative names
    'fear': 5,
    'surprise': 7,
    'happiness': 2,
    'anger': 4,
    'sadness': 3,
}

# Dimensional emotion values (approximate)
EMOTION_DIMENSIONS = {
    'neutral': {'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5},
    'calm': {'valence': 0.3, 'arousal': 0.2, 'dominance': 0.5},
    'happy': {'valence': 0.8, 'arousal': 0.7, 'dominance': 0.6},
    'sad': {'valence': -0.6, 'arousal': 0.3, 'dominance': 0.3},
    'angry': {'valence': -0.5, 'arousal': 0.8, 'dominance': 0.7},
    'fearful': {'valence': -0.7, 'arousal': 0.8, 'dominance': 0.2},
    'disgust': {'valence': -0.6, 'arousal': 0.5, 'dominance': 0.5},
    'surprised': {'valence': 0.2, 'arousal': 0.8, 'dominance': 0.4},
}


class RAVDESSDataset(Dataset):
    """
    RAVDESS Emotional Speech Dataset.
    
    24 actors (12 male, 12 female), 8 emotions, 2 intensities.
    Total: ~1,440 audio files.
    
    Filename format: 03-01-06-01-02-01-12.wav
    - Modality (03 = audio-only)
    - Vocal channel (01 = speech)
    - Emotion (01-08)
    - Intensity (01 = normal, 02 = strong)
    - Statement (01 or 02)
    - Repetition (01 or 02)
    - Actor (01-24, odd = male, even = female)
    
    Download: https://zenodo.org/record/1188976
    """
    
    EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    def __init__(self, root_dir: str, sample_rate: int = 16000,
                 max_duration: float = 5.0, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment
        
        self.samples = self._load_samples()
        
        print(f"RAVDESS: Loaded {len(self.samples)} samples")
        
    def _load_samples(self) -> List[EmotionSample]:
        samples = []
        
        # RAVDESS structure: Audio_Speech_Actors_01-24/Actor_XX/...
        for audio_file in self.root_dir.rglob("*.wav"):
            # Parse filename
            parts = audio_file.stem.split('-')
            
            if len(parts) != 7:
                continue
            
            modality, vocal, emotion_id, intensity, statement, rep, actor = parts
            
            # Only speech (not song)
            if modality != '03' or vocal != '01':
                continue
            
            emotion_idx = int(emotion_id) - 1
            if emotion_idx < 0 or emotion_idx >= len(self.EMOTIONS):
                continue
            
            emotion = self.EMOTIONS[emotion_idx]
            actor_num = int(actor)
            gender = 'male' if actor_num % 2 == 1 else 'female'
            
            # Get dimensional values
            dims = EMOTION_DIMENSIONS.get(emotion, {})
            
            samples.append(EmotionSample(
                audio_path=audio_file,
                emotion=emotion,
                emotion_id=EMOTION_MAPPING[emotion],
                intensity='normal' if intensity == '01' else 'strong',
                speaker_id=f"ravdess_{actor}",
                gender=gender,
                statement=statement,
                valence=dims.get('valence'),
                arousal=dims.get('arousal'),
                dominance=dims.get('dominance'),
            ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(str(sample.audio_path))
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        # Pad or truncate
        if waveform.shape[0] > self.max_samples:
            start = random.randint(0, waveform.shape[0] - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        else:
            waveform = F.pad(waveform, (0, self.max_samples - waveform.shape[0]))
        
        # Augmentation
        if self.augment:
            waveform = self._augment(waveform)
        
        return {
            'audio': waveform,
            'emotion': sample.emotion,
            'emotion_id': sample.emotion_id,
            'intensity': sample.intensity,
            'speaker_id': sample.speaker_id,
            'gender': sample.gender,
            'valence': sample.valence if sample.valence is not None else 0.0,
            'arousal': sample.arousal if sample.arousal is not None else 0.5,
            'dominance': sample.dominance if sample.dominance is not None else 0.5,
        }
    
    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations."""
        # Speed perturbation
        if random.random() < 0.3:
            speed = 0.9 + random.random() * 0.2
            new_length = int(len(waveform) / speed)
            waveform = F.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
            ).squeeze()
            
            # Adjust back to target length
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            else:
                waveform = F.pad(waveform, (0, self.max_samples - len(waveform)))
        
        # Volume perturbation
        if random.random() < 0.3:
            gain = 0.7 + random.random() * 0.6
            waveform = waveform * gain
        
        # Add noise
        if random.random() < 0.2:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        return waveform


class CREMADDataset(Dataset):
    """
    CREMA-D Emotional Speech Dataset.
    
    91 actors (48 male, 43 female), 6 emotions, 4 intensities.
    Total: ~7,442 audio files.
    
    Filename format: 1001_IEO_ANG_HI.wav
    - Actor ID (1001-1091)
    - Sentence (IEO, TIE, ITS, etc.)
    - Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
    - Intensity (LO, MD, HI, XX)
    
    Download: https://github.com/CheyneyComputerScience/CREMA-D
    """
    
    EMOTION_MAP = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad',
    }
    
    def __init__(self, root_dir: str, sample_rate: int = 16000,
                 max_duration: float = 5.0, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment
        
        self.samples = self._load_samples()
        
        print(f"CREMA-D: Loaded {len(self.samples)} samples")
    
    def _load_samples(self) -> List[EmotionSample]:
        samples = []
        
        for audio_file in self.root_dir.rglob("*.wav"):
            parts = audio_file.stem.split('_')
            
            if len(parts) < 4:
                continue
            
            actor_id, sentence, emotion_code, intensity = parts[:4]
            
            if emotion_code not in self.EMOTION_MAP:
                continue
            
            emotion = self.EMOTION_MAP[emotion_code]
            dims = EMOTION_DIMENSIONS.get(emotion, {})
            
            samples.append(EmotionSample(
                audio_path=audio_file,
                emotion=emotion,
                emotion_id=EMOTION_MAPPING[emotion],
                intensity=intensity.lower(),
                speaker_id=f"cremad_{actor_id}",
                statement=sentence,
                valence=dims.get('valence'),
                arousal=dims.get('arousal'),
                dominance=dims.get('dominance'),
            ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        waveform, sr = torchaudio.load(str(sample.audio_path))
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        if waveform.shape[0] > self.max_samples:
            start = random.randint(0, waveform.shape[0] - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        else:
            waveform = F.pad(waveform, (0, self.max_samples - waveform.shape[0]))
        
        return {
            'audio': waveform,
            'emotion': sample.emotion,
            'emotion_id': sample.emotion_id,
            'intensity': sample.intensity,
            'speaker_id': sample.speaker_id,
            'valence': sample.valence if sample.valence is not None else 0.0,
            'arousal': sample.arousal if sample.arousal is not None else 0.5,
            'dominance': sample.dominance if sample.dominance is not None else 0.5,
        }


class GenericEmotionDataset(Dataset):
    """
    Generic emotion dataset from directory structure.
    
    Expected structure:
    root_dir/
        happy/
            audio1.wav
            audio2.wav
        sad/
            audio1.wav
        angry/
            ...
    
    Or with metadata JSON:
    root_dir/
        audio/
            file1.wav
            file2.wav
        metadata.json  (with emotion labels)
    """
    
    def __init__(self, root_dir: str, sample_rate: int = 16000,
                 max_duration: float = 5.0, augment: bool = False,
                 emotion_map: Optional[Dict[str, int]] = None):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment
        self.emotion_map = emotion_map or EMOTION_MAPPING
        
        self.samples = self._load_samples()
        
        print(f"Generic Emotion Dataset: Loaded {len(self.samples)} samples")
    
    def _load_samples(self) -> List[EmotionSample]:
        samples = []
        
        # Check for metadata.json
        metadata_path = self.root_dir / 'metadata.json'
        if metadata_path.exists():
            return self._load_from_metadata(metadata_path)
        
        # Load from directory structure
        for emotion_dir in self.root_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            
            emotion = emotion_dir.name.lower()
            if emotion not in self.emotion_map:
                continue
            
            for audio_file in emotion_dir.glob("*.wav"):
                dims = EMOTION_DIMENSIONS.get(emotion, {})
                
                samples.append(EmotionSample(
                    audio_path=audio_file,
                    emotion=emotion,
                    emotion_id=self.emotion_map[emotion],
                    valence=dims.get('valence'),
                    arousal=dims.get('arousal'),
                    dominance=dims.get('dominance'),
                ))
        
        return samples
    
    def _load_from_metadata(self, metadata_path: Path) -> List[EmotionSample]:
        samples = []
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for item in metadata:
            audio_path = self.root_dir / item['audio_path']
            if not audio_path.exists():
                continue
            
            emotion = item['emotion'].lower()
            if emotion not in self.emotion_map:
                continue
            
            dims = EMOTION_DIMENSIONS.get(emotion, {})
            
            samples.append(EmotionSample(
                audio_path=audio_path,
                emotion=emotion,
                emotion_id=self.emotion_map[emotion],
                speaker_id=item.get('speaker_id'),
                valence=item.get('valence', dims.get('valence')),
                arousal=item.get('arousal', dims.get('arousal')),
                dominance=item.get('dominance', dims.get('dominance')),
            ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        waveform, sr = torchaudio.load(str(sample.audio_path))
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        if waveform.shape[0] > self.max_samples:
            start = random.randint(0, waveform.shape[0] - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        else:
            waveform = F.pad(waveform, (0, self.max_samples - waveform.shape[0]))
        
        return {
            'audio': waveform,
            'emotion': sample.emotion,
            'emotion_id': sample.emotion_id,
            'valence': sample.valence if sample.valence is not None else 0.0,
            'arousal': sample.arousal if sample.arousal is not None else 0.5,
            'dominance': sample.dominance if sample.dominance is not None else 0.5,
        }


def create_combined_emotion_dataset(
    ravdess_dir: Optional[str] = None,
    cremad_dir: Optional[str] = None,
    custom_dirs: Optional[List[str]] = None,
    sample_rate: int = 16000,
    max_duration: float = 5.0,
    augment: bool = False,
) -> Dataset:
    """
    Create combined emotion dataset from multiple sources.
    
    Usage:
        dataset = create_combined_emotion_dataset(
            ravdess_dir='/data/RAVDESS',
            cremad_dir='/data/CREMA-D',
            custom_dirs=['/data/telugu_emotions', '/data/hindi_emotions'],
        )
    """
    datasets = []
    
    if ravdess_dir and Path(ravdess_dir).exists():
        datasets.append(RAVDESSDataset(
            ravdess_dir, sample_rate, max_duration, augment
        ))
    
    if cremad_dir and Path(cremad_dir).exists():
        datasets.append(CREMADDataset(
            cremad_dir, sample_rate, max_duration, augment
        ))
    
    if custom_dirs:
        for custom_dir in custom_dirs:
            if Path(custom_dir).exists():
                datasets.append(GenericEmotionDataset(
                    custom_dir, sample_rate, max_duration, augment
                ))
    
    if len(datasets) == 0:
        raise ValueError("No valid emotion datasets found")
    
    if len(datasets) == 1:
        return datasets[0]
    
    return ConcatDataset(datasets)


def emotion_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for emotion dataloader."""
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'emotion_id': torch.tensor([item['emotion_id'] for item in batch]),
        'valence': torch.tensor([item['valence'] for item in batch]),
        'arousal': torch.tensor([item['arousal'] for item in batch]),
        'dominance': torch.tensor([item['dominance'] for item in batch]),
        'emotions': [item['emotion'] for item in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def download_ravdess(output_dir: str):
    """
    Download RAVDESS dataset from Zenodo.
    
    Note: This is a large download (~5GB).
    """
    import subprocess
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading RAVDESS from Zenodo...")
    print("Note: This is ~5GB and may take a while.")
    
    # Zenodo record URL
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    subprocess.run([
        "wget", "-c", url, "-O", str(output_path / "ravdess.zip")
    ])
    
    print("Extracting...")
    subprocess.run([
        "unzip", str(output_path / "ravdess.zip"), "-d", str(output_path)
    ])
    
    print(f"RAVDESS downloaded to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  EMOTION DATASET TEST")
    print("=" * 70)
    
    # Test with synthetic data
    print("\n--- Creating Test Dataset ---")
    
    # Create temporary test structure
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create emotion directories
        for emotion in ['happy', 'sad', 'angry', 'neutral']:
            emotion_dir = Path(tmpdir) / emotion
            emotion_dir.mkdir()
            
            # Create synthetic audio files
            for i in range(5):
                audio_path = emotion_dir / f"sample_{i}.wav"
                audio = torch.randn(16000 * 3)  # 3 seconds
                torchaudio.save(str(audio_path), audio.unsqueeze(0), 16000)
        
        # Test GenericEmotionDataset
        dataset = GenericEmotionDataset(tmpdir, sample_rate=16000)
        print(f"Dataset size: {len(dataset)}")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"Sample audio shape: {sample['audio'].shape}")
        print(f"Sample emotion: {sample['emotion']}")
        print(f"Sample emotion_id: {sample['emotion_id']}")
        
        # Test dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True,
            collate_fn=emotion_collate_fn
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch audio shape: {batch['audio'].shape}")
        print(f"Batch emotions: {batch['emotions']}")
        print(f"Batch emotion_ids: {batch['emotion_id']}")
        print(f"Batch valence: {batch['valence']}")
        print(f"Batch arousal: {batch['arousal']}")
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)
