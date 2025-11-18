#!/usr/bin/env python3
"""
Telugu Data Collection Pipeline
Collects, segments, and prepares Telugu speech data for S2S training
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TeluguDataCollector:
    """
    Comprehensive data collection pipeline for Telugu S2S training
    """
    
    def __init__(self, config_path: str = "data_sources.yaml", output_dir: str = "telugu_data"):
        """Initialize the data collector"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.segments_dir = self.output_dir / "segments"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir in [self.raw_dir, self.processed_dir, self.segments_dir, self.metadata_dir]:
            dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Statistics
        self.stats = {
            "total_hours_collected": 0,
            "total_segments": 0,
            "sources_processed": [],
            "errors": []
        }
    
    def download_youtube_content(self, url: str, output_name: str, 
                                max_duration: int = 7200) -> Optional[Path]:
        """Download audio from YouTube using yt-dlp"""
        try:
            output_path = self.raw_dir / f"{output_name}.wav"
            
            if output_path.exists():
                logger.info(f"Already downloaded: {output_name}")
                return output_path
            
            logger.info(f"Downloading: {url}")
            
            cmd = [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", str(output_path),
                "--quiet",
                "--no-warnings",
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {output_name}")
                return output_path
            else:
                logger.error(f"Failed to download {url}")
                return None
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    def extract_audio_segment(self, audio_path: Path, start: float, end: float, 
                            output_path: Path) -> bool:
        """Extract audio segment using ffmpeg"""
        try:
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-ss", str(start),
                "-to", str(end),
                "-ar", "16000",
                "-ac", "1",
                "-y",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return False
    
    def process_source(self, source_name: str, source_info: Dict) -> Dict:
        """Process a single data source"""
        logger.info(f"Processing: {source_name}")
        
        stats = {
            "name": source_name,
            "total_hours": 0,
            "total_segments": 0,
            "status": "processing"
        }
        
        # Get URL from source info
        url = source_info.get("url") or source_info.get("channel_url") or source_info.get("playlist_url")
        
        if not url:
            stats["status"] = "no_url"
            return stats
        
        # Download content
        audio_path = self.download_youtube_content(url, source_name)
        if not audio_path:
            stats["status"] = "download_failed"
            return stats
        
        stats["status"] = "completed"
        return stats
    
    def collect_priority_sources(self):
        """Collect high-priority sources"""
        logger.info("Starting priority data collection")
        
        # Process Raw Talks first (highest quality)
        if "raw_talks_vk" in self.config["primary_sources"]:
            stats = self.process_source("raw_talks_vk", 
                                       self.config["primary_sources"]["raw_talks_vk"])
            self.stats["sources_processed"].append(stats)
        
        # Process news channels
        if "news_channels" in self.config["primary_sources"]:
            for channel in self.config["primary_sources"]["news_channels"][:3]:
                stats = self.process_source(channel["name"], channel)
                self.stats["sources_processed"].append(stats)
    
    def generate_report(self):
        """Generate collection report"""
        report_path = self.output_dir / "collection_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        logger.info(f"Total hours collected: {self.stats['total_hours_collected']:.2f}")
        logger.info(f"Total segments: {self.stats['total_segments']}")

if __name__ == "__main__":
    collector = TeluguDataCollector()
    collector.collect_priority_sources()
    collector.generate_report()