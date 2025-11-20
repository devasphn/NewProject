#!/usr/bin/env python3
"""
Fix data path issue: Copy audio files to correct train/val/test directories
"""

import shutil
from pathlib import Path
import json

def fix_data_paths():
    """Copy audio files from source to train/val/test directories"""
    
    print("="*70)
    print("FIXING DATA PATHS")
    print("="*70)
    print()
    
    # Load the split info
    poc_data_dir = Path('/workspace/telugu_poc_data')
    
    print("Loading split files...")
    with open(poc_data_dir / 'train_split.json') as f:
        train_files = json.load(f)
    with open(poc_data_dir / 'val_split.json') as f:
        val_files = json.load(f)
    with open(poc_data_dir / 'test_split.json') as f:
        test_files = json.load(f)
    
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")
    print()
    
    # Source directory
    src_dir = Path('/workspace/telugu_data_production/raw_audio')
    print(f"Source directory: {src_dir}")
    print()
    
    # Create output directories
    train_out = Path('/workspace/telugu_poc_data/train')
    val_out = Path('/workspace/telugu_poc_data/val')
    test_out = Path('/workspace/telugu_poc_data/test')
    
    train_out.mkdir(exist_ok=True, parents=True)
    val_out.mkdir(exist_ok=True, parents=True)
    test_out.mkdir(exist_ok=True, parents=True)
    
    # Copy train files
    print("Copying train files...")
    train_copied = 0
    train_missing = 0
    for item in train_files:
        # Handle both string and dict formats
        if isinstance(item, dict):
            filename = item.get('file', '')
        else:
            filename = item
        
        if not filename:
            continue
            
        # Try to find the file
        src_file = None
        for f in src_dir.rglob('*.wav'):
            if f.name == Path(filename).name or str(f).endswith(filename):
                src_file = f
                break
        
        if src_file and src_file.exists():
            dest = train_out / src_file.name
            if not dest.exists():
                shutil.copy2(src_file, dest)
                train_copied += 1
                if train_copied % 20 == 0:
                    print(f"  Copied {train_copied} files...")
        else:
            train_missing += 1
    
    print(f"✅ Train: {train_copied} copied, {train_missing} missing")
    print()
    
    # Copy val files
    print("Copying val files...")
    val_copied = 0
    val_missing = 0
    for item in val_files:
        if isinstance(item, dict):
            filename = item.get('file', '')
        else:
            filename = item
        
        if not filename:
            continue
            
        src_file = None
        for f in src_dir.rglob('*.wav'):
            if f.name == Path(filename).name or str(f).endswith(filename):
                src_file = f
                break
        
        if src_file and src_file.exists():
            dest = val_out / src_file.name
            if not dest.exists():
                shutil.copy2(src_file, dest)
                val_copied += 1
        else:
            val_missing += 1
    
    print(f"✅ Val: {val_copied} copied, {val_missing} missing")
    print()
    
    # Copy test files
    print("Copying test files...")
    test_copied = 0
    test_missing = 0
    for item in test_files:
        if isinstance(item, dict):
            filename = item.get('file', '')
        else:
            filename = item
        
        if not filename:
            continue
            
        src_file = None
        for f in src_dir.rglob('*.wav'):
            if f.name == Path(filename).name or str(f).endswith(filename):
                src_file = f
                break
        
        if src_file and src_file.exists():
            dest = test_out / src_file.name
            if not dest.exists():
                shutil.copy2(src_file, dest)
                test_copied += 1
        else:
            test_missing += 1
    
    print(f"✅ Test: {test_copied} copied, {test_missing} missing")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Train: {train_copied} files in {train_out}")
    print(f"Val: {val_copied} files in {val_out}")
    print(f"Test: {test_copied} files in {test_out}")
    print(f"Total: {train_copied + val_copied + test_copied} files copied")
    print()
    
    # Verify
    print("Verifying...")
    train_count = len(list(train_out.glob('*.wav')))
    val_count = len(list(val_out.glob('*.wav')))
    test_count = len(list(test_out.glob('*.wav')))
    
    print(f"  Train directory: {train_count} WAV files")
    print(f"  Val directory: {val_count} WAV files")
    print(f"  Test directory: {test_count} WAV files")
    print()
    
    if train_count > 0 and val_count > 0 and test_count > 0:
        print("✅ SUCCESS! All directories have audio files")
        print()
        print("Next step:")
        print("  python finetune_encodec_telugu.py \\")
        print("      --train_dir /workspace/telugu_poc_data/train \\")
        print("      --val_dir /workspace/telugu_poc_data/val \\")
        print("      --output_dir /workspace/models \\")
        print("      --epochs 10 \\")
        print("      --batch_size 4")
    else:
        print("⚠️ WARNING: Some directories still empty!")
        print("   Check source directory has files")
    
    print()

if __name__ == "__main__":
    fix_data_paths()
