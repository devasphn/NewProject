#!/usr/bin/env python3
"""
Project Cleanup Script - Remove unnecessary files and organize codebase
Run with --dry-run to preview changes, or without to execute
"""

import os
import shutil
import argparse
from pathlib import Path

# Files to DELETE (unnecessary/duplicate)
FILES_TO_DELETE = [
    # Duplicate download scripts (7 scripts when we only need 1)
    "download_all_channels.sh",
    "download_single_channel.sh", 
    "download_tier1_only.sh",
    "download_tier1_optimized.sh",
    "download_tier1_SAFE.sh",
    "QUICK_START_AFTER_COOKIES.sh",
    "COMPLETE_FIX_COMMANDS.sh",
    
    # Duplicate/older versions
    "telugu_codec.py",           # Keep telugu_codec_fixed.py
    "train_codec.py",            # Keep train_codec_dac.py
    "streaming_server.py",       # Keep streaming_server_advanced.py
    
    # Excessive documentation (YouTube-focused, no longer needed)
    "FIX_YOUTUBE_BOT_DETECTION.md",
    "FIX_RATE_LIMIT_CHECKLIST.md",
    "START_DATA_COLLECTION.md",
    "PRODUCTION_DOWNLOAD_GUIDE.md",
    "STORAGE_CALCULATOR.md",
    "COMPLETE_COMMAND_REFERENCE.md",
    "COMPLETE_SETUP_COMMANDS.md",
    
    # Utility scripts no longer needed
    "verify_setup.sh",
    "debug_validation_data.py",
    
    # Duplicate cookie file at root
    "youtube_cookies.txt",
]

# Directories to DELETE
DIRS_TO_DELETE = [
    "cookies",  # No longer using YouTube scraping as primary
]

# Files to KEEP (Essential)
ESSENTIAL_FILES = [
    # Core Architecture
    "telugu_codec_fixed.py",
    "discriminator_dac.py",
    "s2s_transformer.py",
    "speaker_embeddings.py",
    "context_manager.py",
    "streaming_server_advanced.py",
    "config.py",
    
    # Training Scripts
    "train_codec_dac.py",
    "train_speakers.py",
    "train_s2s.py",
    
    # Utilities
    "benchmark_latency.py",
    "system_test.py",
    "prepare_speaker_data.py",
    
    # Data & Config
    "data_sources_PRODUCTION.yaml",  # Backup reference
    "requirements_new.txt",
    "data_collection.py",  # Keep for backup YouTube option
    "download_telugu_data_PRODUCTION.py",  # Backup
    
    # New Files
    "download_free_datasets.sh",
    "RECOVERY_PLAN_V1.md",
    "cleanup_and_organize.py",
    
    # Documentation (keep minimal)
    "FROM_SCRATCH_SETUP_GUIDE.md",
    "QUICK_START_RUNPOD.md",
    "RUNPOD_TEMPLATE_SETUP.md",
    "RUNPOD_ENV_VARS.txt",
    
    # Git & Config
    ".gitignore",
]

# Directories to KEEP
ESSENTIAL_DIRS = [
    "static",
    ".git",
]


def cleanup_project(project_dir: str, dry_run: bool = True):
    """Clean up the project directory"""
    project_path = Path(project_dir)
    
    print("=" * 70)
    print("PROJECT CLEANUP SCRIPT")
    print("=" * 70)
    print(f"Project directory: {project_path}")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'EXECUTE (will delete files)'}")
    print("=" * 70)
    print()
    
    deleted_files = []
    deleted_dirs = []
    kept_files = []
    errors = []
    
    # Delete specified files
    print("FILES TO DELETE:")
    print("-" * 50)
    for filename in FILES_TO_DELETE:
        filepath = project_path / filename
        if filepath.exists():
            print(f"  ❌ {filename}")
            if not dry_run:
                try:
                    os.remove(filepath)
                    deleted_files.append(filename)
                except Exception as e:
                    errors.append(f"Error deleting {filename}: {e}")
            else:
                deleted_files.append(filename)
        else:
            print(f"  ⚪ {filename} (not found)")
    
    print()
    
    # Delete specified directories
    print("DIRECTORIES TO DELETE:")
    print("-" * 50)
    for dirname in DIRS_TO_DELETE:
        dirpath = project_path / dirname
        if dirpath.exists():
            print(f"  ❌ {dirname}/")
            if not dry_run:
                try:
                    shutil.rmtree(dirpath)
                    deleted_dirs.append(dirname)
                except Exception as e:
                    errors.append(f"Error deleting {dirname}: {e}")
            else:
                deleted_dirs.append(dirname)
        else:
            print(f"  ⚪ {dirname}/ (not found)")
    
    print()
    
    # List files being kept
    print("FILES TO KEEP:")
    print("-" * 50)
    for filename in ESSENTIAL_FILES:
        filepath = project_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"  ✅ {filename} ({size:.1f} KB)")
            kept_files.append(filename)
        else:
            print(f"  ⚠️  {filename} (MISSING - needs to be created)")
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files to delete: {len(deleted_files)}")
    print(f"Directories to delete: {len(deleted_dirs)}")
    print(f"Files to keep: {len(kept_files)}")
    
    if errors:
        print()
        print("ERRORS:")
        for error in errors:
            print(f"  ⚠️  {error}")
    
    print()
    
    if dry_run:
        print("This was a DRY RUN. No files were actually deleted.")
        print("Run with --execute to perform the cleanup.")
    else:
        print("✅ Cleanup completed!")
    
    print()
    
    # Print final directory structure
    print("=" * 70)
    print("FINAL PROJECT STRUCTURE (after cleanup):")
    print("=" * 70)
    print("""
NewProject/
├── Core Architecture (7 files)
│   ├── telugu_codec_fixed.py      # Neural audio codec
│   ├── discriminator_dac.py       # GAN discriminator  
│   ├── s2s_transformer.py         # Speech-to-Speech model
│   ├── speaker_embeddings.py      # Speaker system
│   ├── context_manager.py         # Conversation context
│   ├── streaming_server_advanced.py # WebSocket server
│   └── config.py                  # Configuration
│
├── Training Scripts (3 files)
│   ├── train_codec_dac.py         # Codec training
│   ├── train_speakers.py          # Speaker training
│   └── train_s2s.py               # S2S model training
│
├── Utilities (3 files)
│   ├── benchmark_latency.py       # Latency testing
│   ├── system_test.py             # System validation
│   └── prepare_speaker_data.py    # Data preparation
│
├── Data Collection (3 files - backup)
│   ├── download_free_datasets.sh  # FREE datasets (PRIMARY!)
│   ├── data_collection.py         # Backup YouTube option
│   └── download_telugu_data_PRODUCTION.py
│
├── Documentation (4 files)
│   ├── RECOVERY_PLAN_V1.md        # Main execution plan
│   ├── FROM_SCRATCH_SETUP_GUIDE.md
│   ├── QUICK_START_RUNPOD.md
│   └── RUNPOD_TEMPLATE_SETUP.md
│
├── Config Files (3 files)
│   ├── requirements_new.txt
│   ├── data_sources_PRODUCTION.yaml
│   └── RUNPOD_ENV_VARS.txt
│
├── Web Interface
│   └── static/index.html
│
└── .gitignore

TOTAL: ~20 essential files (down from 42)
""")

    return {
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "kept_files": kept_files,
        "errors": errors
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up Telugu S2S project")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually delete files (default is dry run)")
    parser.add_argument("--project-dir", default=".", 
                       help="Project directory path")
    
    args = parser.parse_args()
    
    cleanup_project(args.project_dir, dry_run=not args.execute)
