#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    PRE-FLIGHT CHECK - Run this before training!
    
    Verifies:
    ✅ All imports work
    ✅ GPU is available
    ✅ Data directories exist and have files
    ✅ Model can be created
    ✅ Forward pass works
    ✅ Backward pass works
    ✅ No CUDA OOM errors
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os

def check_imports():
    """Check all required imports"""
    print("\n📦 STEP 1: Checking imports...")
    
    required = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM"),
        ("einops", "Einops"),
    ]
    
    failed = []
    for module, name in required:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️ Missing packages: {failed}")
        print("   Installing...")
        os.system(f"pip install {' '.join(failed)} -q")
        return False
    
    return True


def check_gpu():
    """Check GPU availability"""
    print("\n🖥️ STEP 2: Checking GPU...")
    
    import torch
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available!")
        return False, None, 16
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"   ✅ GPU: {gpu_name}")
    print(f"   ✅ Memory: {gpu_mem:.1f} GB")
    
    # Recommend batch size
    if gpu_mem >= 140:
        batch_size = 64
    elif gpu_mem >= 80:
        batch_size = 48
    elif gpu_mem >= 48:
        batch_size = 32
    elif gpu_mem >= 24:
        batch_size = 24
    else:
        batch_size = 16
    
    print(f"   ✅ Recommended batch size: {batch_size}")
    
    return True, gpu_name, batch_size


def check_data():
    """Check data directories"""
    print("\n📁 STEP 3: Checking data directories...")
    
    from pathlib import Path
    
    data_root = Path("data")
    if not data_root.exists():
        print("   ❌ data/ directory not found!")
        return False, [], 0
    
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    data_dirs = []
    total_files = 0
    
    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            files = []
            for ext in audio_extensions:
                files.extend(list(subdir.rglob(f"*{ext}")))
            
            if files:
                data_dirs.append(str(subdir))
                total_files += len(files)
                hours_est = len(files) / 100  # Rough estimate
                print(f"   ✅ {subdir.name}: {len(files):,} files (~{hours_est:.0f}h)")
    
    if total_files < 100:
        print(f"\n   ⚠️ Only {total_files} files found - may need more data")
    else:
        print(f"\n   📊 TOTAL: {total_files:,} audio files")
    
    return total_files >= 100, data_dirs, total_files


def check_model():
    """Check model creation and forward pass"""
    print("\n🔧 STEP 4: Testing model...")
    
    import torch
    
    try:
        from codec_production import ProductionCodec, CodecConfig
        from discriminator_dac import DACDiscriminator
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create codec
        config = CodecConfig()
        codec = ProductionCodec(config).to(device)
        print(f"   ✅ Codec created")
        
        # Count parameters
        codec_params = sum(p.numel() for p in codec.parameters()) / 1e6
        print(f"   ✅ Codec params: {codec_params:.1f}M")
        
        # Create discriminator
        disc = DACDiscriminator().to(device)
        disc_params = sum(p.numel() for p in disc.parameters()) / 1e6
        print(f"   ✅ Discriminator params: {disc_params:.1f}M")
        
        # Test forward pass
        test_audio = torch.randn(2, 1, 32000).to(device)  # 2 second batch
        
        codec.eval()
        with torch.no_grad():
            output = codec(test_audio)
        
        print(f"   ✅ Forward pass successful")
        print(f"      Input: {test_audio.shape}")
        print(f"      Output: {output['audio'].shape}")
        print(f"      Codes: {output['codes'].shape}")
        
        # Test discriminator
        with torch.no_grad():
            logits, features = disc(test_audio)
        print(f"   ✅ Discriminator test successful")
        
        # Test backward pass (training mode)
        codec.train()
        disc.train()
        
        output = codec(test_audio)
        loss = output['loss']
        loss.backward()
        print(f"   ✅ Backward pass successful")
        
        # Clear GPU memory
        del codec, disc, output, test_audio
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_step():
    """Test a full training step"""
    print("\n⚡ STEP 5: Testing full training step...")
    
    import torch
    from torch.cuda.amp import autocast, GradScaler
    
    try:
        from codec_production import ProductionCodec, CodecConfig
        from discriminator_dac import DACDiscriminator, discriminator_loss, generator_adversarial_loss, feature_matching_loss
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models
        codec = ProductionCodec(CodecConfig()).to(device)
        disc = DACDiscriminator().to(device)
        
        # Create optimizers
        gen_opt = torch.optim.AdamW(codec.parameters(), lr=1e-4)
        disc_opt = torch.optim.AdamW(disc.parameters(), lr=1e-4)
        
        # Create scaler
        scaler = GradScaler()
        
        # Test data
        audio = torch.randn(4, 1, 32000).to(device)
        
        # ============ DISCRIMINATOR STEP ============
        disc_opt.zero_grad()
        
        with autocast(enabled=True):
            with torch.no_grad():
                output = codec(audio)
                fake_audio = output['audio']
            
            real_logits, real_features = disc(audio)
            fake_logits, fake_features = disc(fake_audio.detach())
            disc_loss, _, _ = discriminator_loss(real_logits, fake_logits)
        
        scaler.scale(disc_loss).backward()
        scaler.unscale_(disc_opt)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        scaler.step(disc_opt)
        
        print(f"   ✅ Discriminator step: loss={disc_loss.item():.4f}")
        
        # ============ GENERATOR STEP ============
        gen_opt.zero_grad()
        
        with autocast(enabled=True):
            output = codec(audio)
            fake_audio = output['audio']
            gen_loss = output['loss']
            
            fake_logits, fake_features = disc(fake_audio)
            real_logits, real_features = disc(audio)
            
            adv_loss = generator_adversarial_loss(fake_logits)
            feat_loss = feature_matching_loss(real_features, fake_features)
            
            total_loss = gen_loss + adv_loss + 10.0 * feat_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(gen_opt)
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        scaler.step(gen_opt)
        scaler.update()
        
        print(f"   ✅ Generator step: loss={total_loss.item():.4f}")
        print(f"      Recon: {output['recon_loss'].item():.4f}")
        print(f"      VQ: {output['vq_loss'].item():.4f}")
        print(f"      Spectral: {output['spectral_loss'].item():.4f}")
        
        # Check GPU memory
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"   ✅ GPU memory used: {mem_used:.2f} GB")
        
        # Cleanup
        del codec, disc, gen_opt, disc_opt, scaler
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("   PRE-FLIGHT CHECK - Production Codec Training")
    print("="*70)
    
    results = {}
    
    # Run all checks
    results['imports'] = check_imports()
    results['gpu'], gpu_name, batch_size = check_gpu()
    results['data'], data_dirs, total_files = check_data()
    results['model'] = check_model()
    results['training'] = check_training_step()
    
    # Summary
    print("\n" + "="*70)
    print("   SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check.upper()}: {status}")
    
    if all_passed:
        print("\n" + "="*70)
        print("   ✅ ALL CHECKS PASSED! Ready to train.")
        print("="*70)
        
        # Generate training command
        data_dirs_str = " ".join(data_dirs)
        print(f"\n📋 RECOMMENDED TRAINING COMMAND:\n")
        print(f"python train_codec_production.py \\")
        print(f"    --data_dirs {data_dirs_str} \\")
        print(f"    --batch_size {batch_size} \\")
        print(f"    --num_epochs 100 \\")
        print(f"    --checkpoint_dir checkpoints_production")
        
        # Estimate training time
        steps_per_epoch = total_files // batch_size
        print(f"\n⏱️ TRAINING ESTIMATES:")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total steps (100 epochs): {steps_per_epoch * 100:,}")
        if gpu_name and "A40" in gpu_name:
            time_per_epoch = steps_per_epoch * 0.5 / 60  # ~0.5s per step
            print(f"   Estimated time per epoch: ~{time_per_epoch:.0f} minutes")
            print(f"   Estimated total time: ~{time_per_epoch * 100 / 60:.0f} hours")
        
        return 0
    else:
        print("\n" + "="*70)
        print("   ❌ SOME CHECKS FAILED! Fix issues before training.")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
