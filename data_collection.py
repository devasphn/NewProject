#!/usr/bin/env python3
"""
Telugu Data Collection Pipeline
Collects, segments, and prepares Telugu speech data for S2S training
"""

import os
import json
import subprocess
import logging
import argparse
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
        # Verify Node.js is installed
        self._verify_nodejs()
        
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
    
    def _verify_nodejs(self):
        """Verify Node.js is installed for yt-dlp"""
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"✓ Node.js detected: {version}")
            else:
                logger.warning("⚠ Node.js not found - downloads may fail")
                logger.warning("Install with: curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs")
        except FileNotFoundError:
            logger.warning("⚠ Node.js not found - downloads may fail")
            logger.warning("Install with: curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs")
    
    def download_youtube_content(self, url: str, output_name: str, 
                                max_duration: int = 7200) -> Optional[Path]:
        """Download audio from YouTube using yt-dlp with Node.js runtime"""
        try:
            # Create output directory for this source
            source_dir = self.raw_dir / output_name
            source_dir.mkdir(exist_ok=True)
            
            logger.info(f"Downloading from: {url}")
            
            # For channels, get videos tab URL
            if "@" in url and "/videos" not in url:
                url = url.rstrip("/") + "/videos"
            
            # Download latest videos from channel (max 10 videos, max 2 hours each)
            # With Node.js installed, we can use default extractor for better quality
            cmd = [
                "yt-dlp",
                "-f", "bestaudio",  # Best audio quality
                "-x",  # Extract audio
                "--audio-format", "wav",
                "--audio-quality", "0",  # Best quality
                "-o", str(source_dir / "%(title)s.%(ext)s"),
                "--max-downloads", "10",
                "--playlist-end", "10",
                "--match-filter", f"duration < {max_duration} & duration > 60",  # 1min to 2hr
                "--no-check-certificate",
                "--ignore-errors",  # Continue on errors
                "--no-warnings",
                "--progress",  # Show progress
                url
            ]
            
            logger.info(f"Running: yt-dlp (with Node.js runtime)...")
            logger.info(f"Target: {url}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Check if any files were downloaded
            downloaded_files = list(source_dir.glob("*.wav"))
            
            if downloaded_files:
                total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**3)  # GB
                logger.info(f"✓ Downloaded {len(downloaded_files)} files for {output_name} ({total_size:.2f} GB)")
                return source_dir
            else:
                logger.warning(f"No files downloaded for {output_name}")
                if result.stderr:
                    logger.warning(f"stderr: {result.stderr[:500]}")
                if result.stdout:
                    logger.warning(f"stdout: {result.stdout[:500]}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for {url} (>10 minutes)")
            return None
        except Exception as e:
            logger.error(f"Download error for {output_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    parser = argparse.ArgumentParser(description='Telugu Data Collection Pipeline')
    parser.add_argument('--data_dir', type=str, default='telugu_data',
                       help='Output directory for collected data')
    parser.add_argument('--config', type=str, default='data_sources.yaml',
                       help='Path to data sources configuration file')
    parser.add_argument('--max_hours', type=int, default=100,
                       help='Maximum hours to collect')
    parser.add_argument('--quality', type=str, default='high',
                       choices=['low', 'medium', 'high'],
                       help='Audio quality setting')
    
    args = parser.parse_args()
    
    logger.info(f"Starting data collection with:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Max hours: {args.max_hours}")
    logger.info(f"  Quality: {args.quality}")
    
    collector = TeluguDataCollector(config_path=args.config, output_dir=args.data_dir)
    collector.collect_priority_sources()
    collector.generate_report()