#!/usr/bin/env python3
"""
Prepare speaker-labeled dataset from Telugu data
Maps data sources to 4 distinct speaker identities
"""

import os
import json
import shutil
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Speaker mapping based on data sources
SPEAKER_MAPPING = {
    # Male speakers
    "raw_talks": {
        "male_keywords": ["vk", "male", "host"],
        "speaker_id": 0,  # male_young
        "name": "Arjun"
    },
    "10tv": {
        "male_keywords": ["anchor", "male", "news"],
        "speaker_id": 1,  # male_mature
        "name": "Ravi"
    },
    # Female speakers
    "raw_talks_female": {
        "female_keywords": ["female", "guest", "interview"],
        "speaker_id": 2,  # female_young
        "name": "Priya"
    },
    "sakshi": {
        "female_keywords": ["anchor", "female", "news"],
        "speaker_id": 3,  # female_professional
        "name": "Lakshmi"
    }
}

def classify_audio_file(file_path: Path, metadata: Dict = None) -> int:
    """
    Classify audio file to speaker ID based on source and metadata
    
    Args:
        file_path: Path to audio file
        metadata: Optional metadata dictionary
    
    Returns:
        Speaker ID (0-3)
    """
    file_str = str(file_path).lower()
    
    # Check source directory patterns
    if "raw_talks" in file_str or "rawtalks" in file_str:
        # Check if it's male or female speaker
        if any(kw in file_str for kw in ["female", "woman", "lady"]):
            return 2  # female_young
        else:
            return 0  # male_young (default for Raw Talks)
    
    elif "10tv" in file_str or "ntv" in file_str:
        # News channels - check for gender
        if any(kw in file_str for kw in ["female", "woman", "anchor_f"]):
            return 3  # female_professional
        else:
            return 1  # male_mature
    
    elif "sakshi" in file_str or "tv9" in file_str:
        # More news channels
        if any(kw in file_str for kw in ["male", "man", "anchor_m"]):
            return 1  # male_mature
        else:
            return 3  # female_professional (default for Sakshi)
    
    # Check metadata if available
    if metadata:
        gender = metadata.get("gender", "").lower()
        age = metadata.get("age", "unknown")
        source = metadata.get("source", "").lower()
        
        if gender == "male":
            if age == "young" or "podcast" in source:
                return 0  # male_young
            else:
                return 1  # male_mature
        elif gender == "female":
            if age == "young" or "podcast" in source:
                return 2  # female_young
            else:
                return 3  # female_professional
    
    # Default: randomly assign based on distribution
    return random.choice([0, 1, 2, 3])

def prepare_speaker_dataset(
    data_dir: str,
    output_dir: str,
    copy_files: bool = False
) -> Dict:
    """
    Prepare speaker-labeled dataset
    
    Args:
        data_dir: Source data directory
        output_dir: Output directory for organized data
        copy_files: Whether to copy files or just create mapping
    
    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create speaker directories
    speaker_dirs = {
        0: output_path / "speaker_0_male_young",
        1: output_path / "speaker_1_male_mature",
        2: output_path / "speaker_2_female_young",
        3: output_path / "speaker_3_female_professional"
    }
    
    for speaker_dir in speaker_dirs.values():
        speaker_dir.mkdir(exist_ok=True)
    
    # Process audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    speaker_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    file_mapping = []
    
    logger.info("Processing audio files...")
    
    # Load metadata if exists
    metadata_file = data_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Process all audio files
    for audio_file in data_path.rglob("*"):
        if audio_file.suffix.lower() in audio_extensions:
            # Get metadata for this file
            file_metadata = metadata.get(str(audio_file), {})
            
            # Classify to speaker
            speaker_id = classify_audio_file(audio_file, file_metadata)
            speaker_counts[speaker_id] += 1
            
            # Create mapping entry
            relative_path = audio_file.relative_to(data_path)
            mapping_entry = {
                "source": str(relative_path),
                "speaker_id": speaker_id,
                "speaker_name": ["Arjun", "Ravi", "Priya", "Lakshmi"][speaker_id],
                "original_path": str(audio_file)
            }
            file_mapping.append(mapping_entry)
            
            # Copy or link file if requested
            if copy_files:
                dest_file = speaker_dirs[speaker_id] / f"audio_{speaker_counts[speaker_id]:05d}{audio_file.suffix}"
                
                if speaker_counts[speaker_id] <= 1000:  # Limit files per speaker
                    try:
                        # Create symbolic link to save space
                        if not dest_file.exists():
                            dest_file.symlink_to(audio_file)
                    except:
                        # Fall back to copy if symlink fails
                        shutil.copy2(audio_file, dest_file)
    
    # Balance dataset
    min_count = min(speaker_counts.values())
    logger.info(f"Speaker distribution: {speaker_counts}")
    logger.info(f"Minimum samples per speaker: {min_count}")
    
    # Create balanced splits
    balanced_mapping = balance_speakers(file_mapping, min_count)
    
    # Save mapping
    mapping_file = output_path / "speaker_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            "file_mapping": balanced_mapping,
            "speaker_counts": speaker_counts,
            "speakers": {
                0: "male_young (Arjun)",
                1: "male_mature (Ravi)",
                2: "female_young (Priya)",
                3: "female_professional (Lakshmi)"
            }
        }, f, indent=2)
    
    logger.info(f"Saved speaker mapping to {mapping_file}")
    
    # Create train/val/test splits
    create_splits(balanced_mapping, output_path)
    
    return {
        "total_files": len(file_mapping),
        "balanced_files": len(balanced_mapping),
        "speaker_counts": speaker_counts,
        "output_dir": str(output_path)
    }

def balance_speakers(
    file_mapping: List[Dict],
    target_count: int
) -> List[Dict]:
    """
    Balance dataset to have equal samples per speaker
    
    Args:
        file_mapping: List of file mappings
        target_count: Target number of samples per speaker
    
    Returns:
        Balanced file mapping
    """
    balanced = []
    speaker_files = {0: [], 1: [], 2: [], 3: []}
    
    # Group by speaker
    for entry in file_mapping:
        speaker_files[entry["speaker_id"]].append(entry)
    
    # Sample equally from each speaker
    for speaker_id in range(4):
        files = speaker_files[speaker_id]
        
        if len(files) > target_count:
            # Randomly sample
            sampled = random.sample(files, target_count)
        else:
            # Use all available
            sampled = files
        
        balanced.extend(sampled)
    
    random.shuffle(balanced)
    return balanced

def create_splits(
    file_mapping: List[Dict],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Create train/val/test splits with speaker balance
    
    Args:
        file_mapping: List of file mappings
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    # Group by speaker
    speaker_files = {0: [], 1: [], 2: [], 3: []}
    for entry in file_mapping:
        speaker_files[entry["speaker_id"]].append(entry)
    
    splits = {"train": [], "val": [], "test": []}
    
    # Split each speaker's data
    for speaker_id in range(4):
        files = speaker_files[speaker_id]
        n = len(files)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle files
        random.shuffle(files)
        
        # Assign to splits
        splits["train"].extend(files[:n_train])
        splits["val"].extend(files[n_train:n_train + n_val])
        splits["test"].extend(files[n_train + n_val:])
    
    # Shuffle each split
    for split in splits.values():
        random.shuffle(split)
    
    # Save splits
    for split_name, split_data in splits.items():
        split_file = output_dir / f"{split_name}_split.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        logger.info(f"{split_name}: {len(split_data)} samples")
    
    # Save metadata for training
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "num_speakers": 4,
            "speaker_names": {
                "0": "Arjun (male_young)",
                "1": "Ravi (male_mature)",
                "2": "Priya (female_young)",
                "3": "Lakshmi (female_professional)"
            },
            "splits": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"])
            },
            "total_samples": len(file_mapping)
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Prepare speaker-labeled dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Source Telugu data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for speaker data")
    parser.add_argument("--copy_files", action="store_true",
                        help="Copy files to speaker directories")
    args = parser.parse_args()
    
    # Prepare dataset
    stats = prepare_speaker_dataset(
        args.data_dir,
        args.output_dir,
        args.copy_files
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("Speaker Dataset Preparation Complete!")
    print("="*50)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Balanced dataset size: {stats['balanced_files']}")
    print(f"Output directory: {stats['output_dir']}")
    print("\nSpeaker distribution:")
    for sid, count in stats['speaker_counts'].items():
        names = ["Arjun (male_young)", "Ravi (male_mature)", 
                 "Priya (female_young)", "Lakshmi (female_professional)"]
        print(f"  Speaker {sid} ({names[sid]}): {count} samples")
    print("="*50)

if __name__ == "__main__":
    main()