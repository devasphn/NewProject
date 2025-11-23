"""
PRODUCTION Telugu Data Collection Script
Downloads 200-500 hours of Telugu speech from YouTube
Uses data_sources_PRODUCTION.yaml configuration
"""

import os
import sys
import yaml
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime
import shutil

class TeluguDataCollector:
    def __init__(self, config_file="data_sources_PRODUCTION.yaml", output_dir="/workspace/telugu_data_production"):
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.raw_video_dir = self.output_dir / "raw_videos"
        self.audio_dir = self.output_dir / "raw_audio"
        self.processed_dir = self.output_dir / "processed"
        
        # Create directories
        for d in [self.raw_video_dir, self.audio_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        # Statistics
        self.stats = {
            "total_videos_downloaded": 0,
            "total_size_gb": 0,
            "total_duration_hours": 0,
            "speakers": {},
            "errors": []
        }
        
        # Log file
        self.log_file = self.output_dir / f"collection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log(self, message):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def check_disk_space(self):
        """Check available disk space"""
        stat = shutil.disk_usage(self.output_dir)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        
        self.log(f"\n{'='*60}")
        self.log(f"DISK SPACE CHECK")
        self.log(f"{'='*60}")
        self.log(f"Total: {total_gb:.2f} GB")
        self.log(f"Used: {used_gb:.2f} GB")
        self.log(f"Free: {free_gb:.2f} GB")
        self.log(f"{'='*60}\n")
        
        if free_gb < 20:
            self.log("⚠️  WARNING: Less than 20GB free space!")
            return False
        return True
    
    def check_dependencies(self):
        """Check if required tools are installed"""
        self.log("Checking dependencies...")
        
        required = {
            "yt-dlp": "yt-dlp --version",
            "ffmpeg": "ffmpeg -version"
        }
        
        missing = []
        for tool, cmd in required.items():
            try:
                subprocess.run(cmd.split(), capture_output=True, check=True)
                self.log(f"  ✓ {tool} installed")
            except:
                self.log(f"  ✗ {tool} NOT installed")
                missing.append(tool)
        
        if missing:
            self.log(f"\n❌ Missing dependencies: {', '.join(missing)}")
            self.log("Install with:")
            self.log("  pip install yt-dlp")
            self.log("  apt-get install -y ffmpeg")
            return False
        
        self.log("✅ All dependencies installed\n")
        return True
    
    def get_channel_info(self, channel_url):
        """Get channel video count and info"""
        try:
            cmd = [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                channel_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                video_count = len([l for l in lines if l.strip()])
                return {"video_count": video_count, "success": True}
            else:
                return {"video_count": 0, "success": False, "error": result.stderr}
        except Exception as e:
            return {"video_count": 0, "success": False, "error": str(e)}
    
    def download_channel(self, channel_config, tier, download_mode):
        """Download videos from a channel"""
        channel_name = channel_config.get("channel_name", "Unknown")
        channel_url = channel_config.get("channel_url")
        
        if not channel_url:
            self.log(f"⚠️  Skipping {channel_name}: No URL provided")
            return
        
        self.log(f"\n{'='*60}")
        self.log(f"DOWNLOADING: {channel_name}")
        self.log(f"Tier: {tier}")
        self.log(f"Mode: {download_mode}")
        self.log(f"URL: {channel_url}")
        self.log(f"{'='*60}\n")
        
        # Get channel info
        self.log("Getting channel info...")
        info = self.get_channel_info(channel_url)
        
        if info["success"]:
            self.log(f"  Total videos in channel: {info['video_count']}")
        else:
            self.log(f"  ⚠️  Could not get channel info: {info.get('error', 'Unknown')}")
        
        # Determine playlist end based on download mode
        playlist_items = None
        if download_mode == "RECENT_500_VIDEOS":
            playlist_items = "1-500"
        elif download_mode == "RECENT_200_VIDEOS":
            playlist_items = "1-200"
        elif download_mode == "RECENT_150_VIDEOS":
            playlist_items = "1-150"
        elif download_mode == "RECENT_100_VIDEOS":
            playlist_items = "1-100"
        # elif download_mode == "ALL_VIDEOS": playlist_items remains None (all)
        
        # Create output directory for this channel
        channel_slug = channel_name.replace(" ", "_").replace("/", "_")
        channel_output = self.raw_video_dir / tier / channel_slug
        channel_output.mkdir(parents=True, exist_ok=True)
        
        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "--format", "best[height<=720]",  # 720p max to save space
            "--output", str(channel_output / "%(title)s.%(ext)s"),
            "--extractor-args", "youtube:player_client=android,web",  # Bypass JS requirement
            "--write-info-json",
            "--write-description",
            "--write-thumbnail",
            "--ignore-errors",
            "--no-overwrites",
            "--download-archive", str(channel_output / "downloaded.txt"),
            "--sleep-interval", "10",  # Rate limiting: 10 sec between videos
            "--max-sleep-interval", "30",  # Up to 30 sec random delay
            "--sleep-requests", "1",  # 1 sec between requests
            "--max-downloads", "50",  # Download 50 at a time, then pause
            "--concurrent-fragments", "2",  # Reduce concurrent connections
        ]
        
        # Add cookies to bypass YouTube bot detection
        # CRITICAL: Export cookies from signed-in YouTube browser session
        # Save as: /workspace/cookies/youtube_cookies.txt
        cookies_path = Path("/workspace/cookies/youtube_cookies.txt")
        if cookies_path.exists():
            cmd.extend(["--cookies", str(cookies_path)])
            self.log("  ✓ Using cookies for authentication")
        else:
            self.log("  ⚠️  WARNING: No cookies found at /workspace/cookies/youtube_cookies.txt")
            self.log("  ⚠️  Downloads may fail due to bot detection!")
            self.log("  ⚠️  Export cookies from browser: See FIX_YOUTUBE_BOT_DETECTION.md")
        
        if playlist_items:
            cmd.extend(["--playlist-items", playlist_items])
        
        # Add filters
        cmd.extend([
            "--match-filter", "duration > 60 & duration < 7200",  # 1 min to 2 hours
            "--no-playlist" if "/watch?" in channel_url else "--yes-playlist"
        ])
        
        cmd.append(channel_url)
        
        # Execute download
        self.log(f"Starting download...")
        self.log(f"Command: {' '.join(cmd[:5])}...")  # Log first few args
        
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                if "[download]" in line or "[ExtractAudio]" in line or "ERROR" in line:
                    print(line.strip())
            
            process.wait()
            
            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                self.log(f"✅ Download completed in {elapsed/60:.1f} minutes")
                
                # Count downloaded files
                video_files = list(channel_output.glob("*.mp4")) + list(channel_output.glob("*.webm")) + list(channel_output.glob("*.mkv"))
                self.log(f"  Downloaded: {len(video_files)} videos")
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in video_files) / (1024**3)
                self.log(f"  Total size: {total_size:.2f} GB")
                
                # Update stats
                self.stats["total_videos_downloaded"] += len(video_files)
                self.stats["total_size_gb"] += total_size
                
                # Track speaker
                speaker_id = channel_config.get("speaker_profile", {}).get("id", channel_name)
                if speaker_id not in self.stats["speakers"]:
                    self.stats["speakers"][speaker_id] = {
                        "channel": channel_name,
                        "videos": 0,
                        "size_gb": 0
                    }
                self.stats["speakers"][speaker_id]["videos"] += len(video_files)
                self.stats["speakers"][speaker_id]["size_gb"] += total_size
                
            else:
                self.log(f"❌ Download failed with return code {process.returncode}")
                self.stats["errors"].append({
                    "channel": channel_name,
                    "error": f"Download failed (code {process.returncode})"
                })
        
        except Exception as e:
            self.log(f"❌ Exception during download: {e}")
            self.stats["errors"].append({
                "channel": channel_name,
                "error": str(e)
            })
        
        # Show current stats
        self.print_stats()
    
    def extract_audio_from_videos(self):
        """Extract 16kHz mono audio from all downloaded videos"""
        self.log(f"\n{'='*60}")
        self.log("EXTRACTING AUDIO FROM VIDEOS")
        self.log(f"{'='*60}\n")
        
        # Find all video files
        video_extensions = [".mp4", ".webm", ".mkv", ".avi", ".flv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(self.raw_video_dir.rglob(f"*{ext}"))
        
        self.log(f"Found {len(video_files)} video files")
        
        processed = 0
        errors = 0
        
        for video_file in video_files:
            try:
                # Create parallel directory structure in audio_dir
                rel_path = video_file.relative_to(self.raw_video_dir)
                audio_file = self.audio_dir / rel_path.with_suffix(".wav")
                audio_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if already extracted
                if audio_file.exists():
                    continue
                
                # Extract audio using ffmpeg
                cmd = [
                    "ffmpeg",
                    "-i", str(video_file),
                    "-ar", "16000",  # 16kHz
                    "-ac", "1",       # Mono
                    "-c:a", "pcm_s16le",  # 16-bit PCM
                    "-y",  # Overwrite
                    str(audio_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=300  # 5 minute timeout per file
                )
                
                if result.returncode == 0:
                    processed += 1
                    if processed % 10 == 0:
                        self.log(f"  Processed: {processed}/{len(video_files)}")
                else:
                    errors += 1
                    self.log(f"  ⚠️  Error extracting {video_file.name}")
            
            except Exception as e:
                errors += 1
                self.log(f"  ❌ Exception extracting {video_file.name}: {e}")
        
        self.log(f"\n✅ Audio extraction complete")
        self.log(f"  Processed: {processed}")
        self.log(f"  Errors: {errors}")
        
        # Calculate audio size
        audio_files = list(self.audio_dir.rglob("*.wav"))
        total_audio_size = sum(f.stat().st_size for f in audio_files) / (1024**3)
        self.log(f"  Total audio size: {total_audio_size:.2f} GB")
        
        # Estimate hours (16kHz mono WAV ≈ 30MB per hour)
        estimated_hours = (total_audio_size * 1024) / 30
        self.log(f"  Estimated hours: {estimated_hours:.1f} hours")
        self.stats["total_duration_hours"] = estimated_hours
    
    def print_stats(self):
        """Print current statistics"""
        self.log(f"\n{'='*60}")
        self.log("COLLECTION STATISTICS")
        self.log(f"{'='*60}")
        self.log(f"Total videos: {self.stats['total_videos_downloaded']}")
        self.log(f"Total size: {self.stats['total_size_gb']:.2f} GB")
        self.log(f"Estimated hours: {self.stats['total_duration_hours']:.1f}")
        self.log(f"Unique speakers: {len(self.stats['speakers'])}")
        
        if self.stats['speakers']:
            self.log(f"\nSpeaker breakdown:")
            for speaker_id, data in self.stats['speakers'].items():
                self.log(f"  {speaker_id}: {data['videos']} videos, {data['size_gb']:.2f} GB")
        
        if self.stats['errors']:
            self.log(f"\nErrors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5
                self.log(f"  {error['channel']}: {error['error']}")
        
        self.log(f"{'='*60}\n")
    
    def save_stats(self):
        """Save statistics to JSON"""
        stats_file = self.output_dir / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        self.log(f"Statistics saved to {stats_file}")
    
    def run_tier(self, tier_name, tier_config):
        """Download all channels in a tier"""
        self.log(f"\n{'#'*60}")
        self.log(f"STARTING TIER: {tier_name.upper()}")
        self.log(f"{'#'*60}\n")
        
        if isinstance(tier_config, dict):
            # Single channel
            for channel_key, channel_config in tier_config.items():
                if isinstance(channel_config, dict) and "channel_url" in channel_config:
                    download_mode = channel_config.get("download_mode", "ALL_VIDEOS")
                    self.download_channel(channel_config, tier_name, download_mode)
        
        elif isinstance(tier_config, list):
            # List of channels
            for channel_config in tier_config:
                if isinstance(channel_config, dict) and "channel_url" in channel_config:
                    download_mode = channel_config.get("download_mode", "ALL_VIDEOS")
                    self.download_channel(channel_config, tier_name, download_mode)
    
    def collect_all(self):
        """Main collection pipeline"""
        self.log(f"\n{'#'*60}")
        self.log("TELUGU DATA COLLECTION - PRODUCTION")
        self.log(f"{'#'*60}\n")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Configuration: {self.config_file}")
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Pre-flight checks
        if not self.check_dependencies():
            self.log("❌ Dependency check failed. Exiting.")
            return False
        
        if not self.check_disk_space():
            self.log("⚠️  Low disk space warning")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != "yes":
                self.log("Exiting.")
                return False
        
        try:
            # Phase 1: Tier 1 channels (High priority)
            if "tier1_podcasts" in self.config:
                self.run_tier("tier1_podcasts", self.config["tier1_podcasts"])
            
            if "tier1_news" in self.config:
                self.run_tier("tier1_news", self.config["tier1_news"])
            
            # Phase 2: Tier 2 channels (Speaker diversity)
            if "tier2_speaker_diversity" in self.config:
                for key, value in self.config["tier2_speaker_diversity"].items():
                    if isinstance(value, dict) and "channel_url" in value:
                        download_mode = value.get("download_mode", "ALL_VIDEOS")
                        self.download_channel(value, "tier2_speaker_diversity", download_mode)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "channel_url" in item:
                                download_mode = item.get("download_mode", "ALL_VIDEOS")
                                self.download_channel(item, "tier2_speaker_diversity", download_mode)
            
            # Phase 3: Tier 3 channels (Additional diversity)
            if "tier3_additional_speakers" in self.config:
                for key, value in self.config["tier3_additional_speakers"].items():
                    if isinstance(value, dict) and "channel_url" in value:
                        download_mode = value.get("download_mode", "ALL_VIDEOS")
                        self.download_channel(value, "tier3_additional_speakers", download_mode)
            
            # Phase 4: Accent diversity
            if "accent_diversity_channels" in self.config:
                for key, value in self.config["accent_diversity_channels"].items():
                    if isinstance(value, dict) and "channel_url" in value:
                        download_mode = value.get("download_mode", "ALL_VIDEOS")
                        self.download_channel(value, "accent_diversity_channels", download_mode)
            
            # Phase 5: Extract audio
            self.log("\n" + "="*60)
            self.log("VIDEO DOWNLOAD COMPLETE - STARTING AUDIO EXTRACTION")
            self.log("="*60 + "\n")
            
            self.extract_audio_from_videos()
            
            # Final stats
            self.print_stats()
            self.save_stats()
            
            self.log(f"\n{'#'*60}")
            self.log("COLLECTION COMPLETE!")
            self.log(f"{'#'*60}\n")
            self.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"Total videos: {self.stats['total_videos_downloaded']}")
            self.log(f"Total size: {self.stats['total_size_gb']:.2f} GB")
            self.log(f"Estimated hours: {self.stats['total_duration_hours']:.1f}")
            self.log(f"Unique speakers: {len(self.stats['speakers'])}")
            self.log(f"\nNext steps:")
            self.log(f"  1. Run silence removal: python process_audio.py")
            self.log(f"  2. Prepare speaker dataset: python prepare_speaker_data.py")
            self.log(f"  3. Start training: python train_codec_dac.py")
            
            return True
        
        except KeyboardInterrupt:
            self.log("\n\n⚠️  Collection interrupted by user")
            self.print_stats()
            self.save_stats()
            return False
        
        except Exception as e:
            self.log(f"\n\n❌ Collection failed with exception: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.print_stats()
            self.save_stats()
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Telugu data for codec training")
    parser.add_argument(
        "--config",
        default="data_sources_PRODUCTION.yaml",
        help="Path to configuration YAML"
    )
    parser.add_argument(
        "--output",
        default="/workspace/telugu_data_production",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Download only Tier 1 channels (high priority)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip audio extraction (download only)"
    )
    
    args = parser.parse_args()
    
    collector = TeluguDataCollector(
        config_file=args.config,
        output_dir=args.output
    )
    
    success = collector.collect_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
