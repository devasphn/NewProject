#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    VERIFY AND TRAIN - Complete Production Pipeline
    
    This script:
    1. Verifies all dependencies are installed
    2. Checks all data directories
    3. Tests the codec model
    4. Starts training with optimal settings
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path
import subprocess

print("="*70)
print("   PRODUCTION CODEC - VERIFY AND TRAIN")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Check Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📦 Step 1: Checking dependencies...")

required_packages = [
    "torch", "torchaudio", "numpy", "tqdm", 
    "einops", "tensorboard"
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} - MISSING")
        missing.append(pkg)

if missing:
    print(f"\n⚠️ Installing missing packages: {missing}")
    subprocess.run([sys.executable, "-m", "pip", "install"] + missing + ["-q"])
    print("   ✅ Packages installed")

# Now import everything
import torch
import torchaudio
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Check GPU
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🖥️ Step 2: Checking GPU...")

if not torch.cuda.is_available():
    print("   ❌ CUDA not available!")
    print("   Training will be very slow on CPU.")
    device = "cpu"
    batch_size = 4
else:
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   ✅ GPU: {gpu_name}")
    print(f"   ✅ Memory: {gpu_mem:.1f} GB")
    
    # Set batch size based on GPU memory
    if gpu_mem >= 140:  # H200
        batch_size = 64
    elif gpu_mem >= 80:  # A100-80GB, H100
        batch_size = 48
    elif gpu_mem >= 48:  # A40, A6000
        batch_size = 32
    elif gpu_mem >= 24:  # RTX 3090/4090
        batch_size = 24
    else:
        batch_size = 16
    
    print(f"   ✅ Batch size: {batch_size}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Check Data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📁 Step 3: Checking data directories...")

data_root = Path("data")
if not data_root.exists():
    print("   ❌ data/ directory not found!")
    print("   Please download data first.")
    sys.exit(1)

audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
data_dirs = []
total_files = 0
total_hours_estimate = 0

for subdir in sorted(data_root.iterdir()):
    if subdir.is_dir():
        files = []
        for ext in audio_extensions:
            files.extend(list(subdir.rglob(f"*{ext}")))
        
        if files:
            data_dirs.append(str(subdir))
            total_files += len(files)
            # Rough estimate: 1000 files ≈ 10 hours
            hours_est = len(files) / 100
            total_hours_estimate += hours_est
            print(f"   ✅ {subdir.name}: {len(files):,} files (~{hours_est:.0f}h)")

print(f"\n   📊 TOTAL: {total_files:,} files (~{total_hours_estimate:.0f} hours)")

if total_files < 100:
    print("\n   ⚠️ Not enough data! Please download more.")
    print("   Run: bash download_remaining_data.sh")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Test Codec Model
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔧 Step 4: Testing codec model...")

try:
    from codec_production import ProductionCodec, CodecConfig
    from discriminator_dac import DACDiscriminator
    
    # Create codec
    config = CodecConfig()
    codec = ProductionCodec(config)
    
    # Test forward pass
    test_audio = torch.randn(2, 1, 16000)  # 1 second batch
    codec.eval()
    with torch.no_grad():
        output = codec(test_audio)
    
    print(f"   ✅ Codec forward pass successful")
    print(f"   ✅ Input: {test_audio.shape}")
    print(f"   ✅ Output: {output['audio'].shape}")
    print(f"   ✅ Codes: {output['codes'].shape}")
    
    # Test discriminator
    disc = DACDiscriminator()
    with torch.no_grad():
        logits, features = disc(test_audio)
    print(f"   ✅ Discriminator test successful")
    
    # Calculate model size
    codec_params = sum(p.numel() for p in codec.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f"   ✅ Codec params: {codec_params:.1f}M")
    print(f"   ✅ Discriminator params: {disc_params:.1f}M")
    
except Exception as e:
    print(f"   ❌ Error testing model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Show Training Command
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("✅ ALL CHECKS PASSED!")
print("="*70)

data_dirs_str = " ".join(data_dirs)
training_cmd = f"""
python train_codec_production.py \\
    --data_dirs {data_dirs_str} \\
    --batch_size {batch_size} \\
    --num_epochs 100 \\
    --checkpoint_dir checkpoints_production
"""

print("\n📋 Training command:")
print(training_cmd)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Ask to Start Training
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
response = input("🚀 Start training now? (y/n): ").strip().lower()

if response == 'y':
    print("\n🚀 Starting training...")
    print("="*70 + "\n")
    
    # Import and run training
    from train_codec_production import TrainConfig, ProductionCodecTrainer
    
    train_config = TrainConfig(
        data_dirs=data_dirs,
        batch_size=batch_size,
        num_epochs=100,
        gen_lr=1e-4,
        disc_lr=1e-4,
        checkpoint_dir="checkpoints_production",
        num_workers=4,
        use_fp16=True,
    )
    
    trainer = ProductionCodecTrainer(train_config)
    trainer.train()
else:
    print("\n👋 Training cancelled. Run the command above when ready.")
    print("   Or run: bash start_training.sh")
