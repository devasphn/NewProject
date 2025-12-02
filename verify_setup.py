#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  PRE-TRAINING VERIFICATION SCRIPT
  
  Run this BEFORE starting training to verify:
  1. All dependencies are installed
  2. GPU is available and has enough VRAM
  3. Data directories have audio files
  4. Codec model can be instantiated
  5. Discriminator can be instantiated
  
  Usage: python verify_setup.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)

def print_check(name, passed, details=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")
    if details:
        print(f"      {details}")

def main():
    print_header("PRE-TRAINING VERIFICATION")
    
    all_passed = True
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 1: Python & Dependencies
    # ═══════════════════════════════════════════════════════════════════════
    print("\n📦 DEPENDENCIES")
    print("-" * 50)
    
    # Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 9)
    print_check(f"Python {py_version.major}.{py_version.minor}", py_ok,
               "Requires Python 3.9+")
    all_passed &= py_ok
    
    # PyTorch
    try:
        import torch
        torch_ok = True
        print_check(f"PyTorch {torch.__version__}", True)
    except ImportError:
        print_check("PyTorch", False, "Run: pip install torch")
        torch_ok = False
        all_passed = False
    
    # TorchAudio
    try:
        import torchaudio
        print_check(f"TorchAudio {torchaudio.__version__}", True)
    except ImportError:
        print_check("TorchAudio", False, "Run: pip install torchaudio")
        all_passed = False
    
    # Transformers (for WavLM)
    try:
        from transformers import WavLMModel
        print_check("Transformers (WavLM)", True)
    except ImportError:
        print_check("Transformers", False, "Run: pip install transformers")
        all_passed = False
    
    # einops
    try:
        import einops
        print_check(f"einops {einops.__version__}", True)
    except ImportError:
        print_check("einops", False, "Run: pip install einops")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 2: GPU
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🖥️  GPU")
    print("-" * 50)
    
    if torch_ok:
        cuda_available = torch.cuda.is_available()
        print_check("CUDA Available", cuda_available)
        all_passed &= cuda_available
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print_check(f"GPU: {gpu_name}", True)
            
            # Check VRAM (need at least 24GB for production training)
            vram_ok = gpu_mem >= 20
            print_check(f"VRAM: {gpu_mem:.1f} GB", vram_ok,
                       "Recommended: 24GB+ for batch_size=32")
            
            # Recommend batch size
            if gpu_mem >= 80:
                rec_batch = 64
            elif gpu_mem >= 40:
                rec_batch = 32
            elif gpu_mem >= 24:
                rec_batch = 16
            else:
                rec_batch = 8
            print(f"      Recommended batch size: {rec_batch}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 3: Data Directories
    # ═══════════════════════════════════════════════════════════════════════
    print("\n📁 DATA DIRECTORIES")
    print("-" * 50)
    
    # Check common data locations
    data_dirs = [
        "/workspace/data/english",
        "/workspace/data/hindi", 
        "/workspace/data/telugu",
        "data/english",
        "data/hindi",
        "data/telugu",
    ]
    
    total_files = 0
    for data_dir in data_dirs:
        path = Path(data_dir)
        if path.exists():
            # Count audio files
            wav_count = len(list(path.rglob("*.wav")))
            flac_count = len(list(path.rglob("*.flac")))
            mp3_count = len(list(path.rglob("*.mp3")))
            total = wav_count + flac_count + mp3_count
            total_files += total
            
            if total > 0:
                print_check(f"{data_dir}", True, f"{total} audio files")
            else:
                print_check(f"{data_dir}", False, "No audio files found")
    
    if total_files == 0:
        print("  ⚠️  No audio data found!")
        print("      Run: bash download_6000h_data.sh")
        print("      Then: python download_huggingface_data.py")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 4: Model Instantiation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🧠 MODEL INSTANTIATION")
    print("-" * 50)
    
    # Try to import and instantiate codec
    try:
        from codec_production import ProductionCodec, CodecConfig
        config = CodecConfig()
        codec = ProductionCodec(config)
        param_count = sum(p.numel() for p in codec.parameters())
        print_check(f"ProductionCodec ({param_count/1e6:.1f}M params)", True)
        del codec
    except Exception as e:
        print_check("ProductionCodec", False, str(e)[:50])
        all_passed = False
    
    # Try to import discriminator
    try:
        from discriminator_dac import DACDiscriminator
        disc = DACDiscriminator()
        param_count = sum(p.numel() for p in disc.parameters())
        print_check(f"DACDiscriminator ({param_count/1e6:.1f}M params)", True)
        del disc
    except Exception as e:
        print_check("DACDiscriminator", False, str(e)[:50])
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 5: Quick Forward Pass
    # ═══════════════════════════════════════════════════════════════════════
    if torch_ok and cuda_available:
        print("\n⚡ QUICK FORWARD PASS TEST")
        print("-" * 50)
        
        try:
            from codec_production import ProductionCodec, CodecConfig
            
            device = torch.device("cuda")
            config = CodecConfig()
            codec = ProductionCodec(config).to(device)
            
            # Test forward pass
            x = torch.randn(2, 1, 16000).to(device)  # 1 second of audio
            
            with torch.no_grad():
                output = codec(x)
            
            print_check("Forward pass (batch=2, 1sec)", True)
            print(f"      Input shape: {x.shape}")
            print(f"      Output reconstructed shape: {output['reconstructed'].shape}")
            
            # Estimate latency
            import time
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                with torch.no_grad():
                    output = codec(x)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 10 * 1000
            
            print_check(f"Inference latency: {elapsed:.1f}ms per batch", elapsed < 100,
                       "Target: <100ms for <400ms E2E")
            
            del codec
            torch.cuda.empty_cache()
            
        except Exception as e:
            print_check("Forward pass test", False, str(e)[:60])
            all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print_header("SUMMARY")
    
    if all_passed:
        print("  ✅ ALL CHECKS PASSED!")
        print("\n  Ready to train. Recommended command:")
        print("  " + "-" * 60)
        print("  python train_codec_production.py \\")
        print("      --data_dirs /workspace/data/english /workspace/data/hindi /workspace/data/telugu \\")
        print("      --batch_size 32 \\")
        print("      --num_epochs 100 \\")
        print("      --checkpoint_dir /workspace/checkpoints_codec")
        print("  " + "-" * 60)
    else:
        print("  ❌ SOME CHECKS FAILED")
        print("\n  Please fix the issues above before training.")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
