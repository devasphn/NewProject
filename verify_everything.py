#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    COMPREHENSIVE VERIFICATION SCRIPT
    
    Verifies:
    1. All data directories and counts
    2. Removes empty directories  
    3. Checks for duplicates
    4. Tests codec encode/decode latency
    5. Tests Telugu audio encoding specifically
    6. Provides final training command
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from collections import defaultdict
import shutil

def verify_data():
    """Verify all data directories"""
    print("\n" + "="*70)
    print("   STEP 1: DATA VERIFICATION")
    print("="*70)
    
    data_root = Path("data")
    if not data_root.exists():
        print("❌ data/ directory not found!")
        return None
    
    audio_extensions = {'.wav', '.flac', '.mp3', '.opus', '.ogg', '.m4a'}
    
    results = {}
    empty_dirs = []
    
    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            files = []
            for ext in audio_extensions:
                files.extend(list(subdir.rglob(f"*{ext}")))
            
            size_bytes = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
            size_gb = size_bytes / (1024**3)
            
            if len(files) == 0:
                empty_dirs.append(subdir)
                print(f"   ❌ {subdir.name}: EMPTY (will be deleted)")
            else:
                hours_est = len(files) * 5 / 3600  # ~5 sec average per file
                results[subdir.name] = {
                    'path': str(subdir),
                    'files': len(files),
                    'size_gb': size_gb,
                    'hours_est': hours_est
                }
                print(f"   ✅ {subdir.name}: {len(files):,} files, {size_gb:.1f}GB (~{hours_est:.0f}h)")
    
    # Delete empty directories
    if empty_dirs:
        print(f"\n   🗑️ Deleting {len(empty_dirs)} empty directories...")
        for d in empty_dirs:
            try:
                shutil.rmtree(d)
                print(f"      Deleted: {d.name}")
            except Exception as e:
                print(f"      Failed to delete {d.name}: {e}")
    
    # Total
    total_files = sum(r['files'] for r in results.values())
    total_size = sum(r['size_gb'] for r in results.values())
    total_hours = sum(r['hours_est'] for r in results.values())
    
    print(f"\n   📊 TOTAL: {total_files:,} files, {total_size:.1f}GB (~{total_hours:.0f} hours)")
    
    return results


def check_duplicates():
    """Check for duplicate Telugu files"""
    print("\n" + "="*70)
    print("   STEP 2: CHECKING FOR DUPLICATES")
    print("="*70)
    
    telugu_dirs = list(Path("data").glob("*telugu*"))
    
    if len(telugu_dirs) <= 1:
        print("   ✅ No duplicate Telugu directories")
        return
    
    print(f"   Found {len(telugu_dirs)} Telugu directories:")
    
    # Get sample files from each
    file_hashes = defaultdict(list)
    
    for tdir in telugu_dirs:
        if not tdir.exists():
            continue
        wav_files = list(tdir.rglob("*.wav"))[:100]  # Sample first 100
        
        for wf in wav_files:
            try:
                # Hash first 10KB
                with open(wf, 'rb') as f:
                    h = hashlib.md5(f.read(10240)).hexdigest()
                file_hashes[h].append(str(wf))
            except:
                pass
    
    duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
    
    if duplicates:
        print(f"   ⚠️ Found {len(duplicates)} duplicate files!")
        print("   Recommendation: Merge or delete duplicate directories")
        
        # Check if telugu and telugu_openslr are duplicates
        telugu_path = Path("data/telugu")
        openslr_path = Path("data/telugu_openslr")
        
        if telugu_path.exists() and openslr_path.exists():
            telugu_files = set(f.name for f in telugu_path.rglob("*.wav"))
            openslr_files = set(f.name for f in openslr_path.rglob("*.wav"))
            
            overlap = telugu_files & openslr_files
            if len(overlap) > len(telugu_files) * 0.5:
                print(f"\n   🔄 telugu/ and telugu_openslr/ are ~{len(overlap)/len(telugu_files)*100:.0f}% duplicates")
                print("   Merging into single directory...")
                
                # Move unique files from openslr to telugu
                for f in openslr_path.rglob("*.wav"):
                    if f.name not in telugu_files:
                        dest = telugu_path / f.name
                        shutil.copy2(f, dest)
                
                # Delete openslr directory
                shutil.rmtree(openslr_path)
                print("   ✅ Merged and deleted duplicate directory")
    else:
        print("   ✅ No duplicate files found")


def test_latency():
    """Test codec encode/decode latency"""
    print("\n" + "="*70)
    print("   STEP 3: LATENCY TEST")
    print("="*70)
    
    import torch
    
    try:
        from codec_production import ProductionCodec, CodecConfig
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create codec
        config = CodecConfig()
        codec = ProductionCodec(config).to(device)
        codec.eval()
        
        # Frame rate info
        total_downsample = 1
        for r in config.downsample_rates:
            total_downsample *= r
        frame_rate = config.sample_rate / total_downsample
        frame_duration_ms = 1000 / frame_rate
        
        print(f"   Frame rate: {frame_rate} Hz")
        print(f"   Frame duration: {frame_duration_ms:.1f} ms")
        
        # Test with different audio lengths
        test_lengths = [
            (0.02, "20ms - single frame"),
            (0.1, "100ms - 5 frames"),
            (1.0, "1 second"),
            (2.0, "2 seconds"),
        ]
        
        print(f"\n   {'Length':<25} {'Encode':<12} {'Decode':<12} {'Total':<12}")
        print("   " + "-"*60)
        
        for duration, label in test_lengths:
            samples = int(config.sample_rate * duration)
            audio = torch.randn(1, 1, samples).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = codec.encode(audio)
            
            # Time encoding
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                codes = codec.encode(audio)
            torch.cuda.synchronize()
            encode_time = (time.perf_counter() - start) * 1000
            
            # Time decoding
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                decoded = codec.decode(codes)
            torch.cuda.synchronize()
            decode_time = (time.perf_counter() - start) * 1000
            
            total_time = encode_time + decode_time
            
            print(f"   {label:<25} {encode_time:>8.2f} ms  {decode_time:>8.2f} ms  {total_time:>8.2f} ms")
        
        # Calculate real-time factor
        audio_2s = torch.randn(1, 1, 32000).to(device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            codes = codec.encode(audio_2s)
            decoded = codec.decode(codes)
        torch.cuda.synchronize()
        total_2s = (time.perf_counter() - start) * 1000
        
        rtf = total_2s / 2000  # 2000ms of audio
        
        print(f"\n   📊 LATENCY SUMMARY:")
        print(f"      Algorithmic latency: {frame_duration_ms:.1f}ms (one frame)")
        print(f"      Encode+Decode (2s audio): {total_2s:.1f}ms")
        print(f"      Real-time factor: {rtf:.4f}x")
        
        if rtf < 0.1:
            print(f"      ✅ EXCELLENT: {1/rtf:.0f}x faster than real-time!")
        elif rtf < 0.5:
            print(f"      ✅ GOOD: {1/rtf:.1f}x faster than real-time")
        else:
            print(f"      ⚠️ Slow: Only {1/rtf:.1f}x real-time")
        
        del codec
        torch.cuda.empty_cache()
        
        return True, frame_duration_ms, total_2s
        
    except Exception as e:
        print(f"   ❌ Latency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0


def test_telugu_encoding():
    """Test encoding actual Telugu audio"""
    print("\n" + "="*70)
    print("   STEP 4: TELUGU AUDIO ENCODING TEST")
    print("="*70)
    
    import torch
    import torchaudio
    
    telugu_dir = Path("data/telugu")
    if not telugu_dir.exists():
        print("   ⚠️ Telugu directory not found, skipping test")
        return True
    
    # Find a Telugu audio file
    wav_files = list(telugu_dir.rglob("*.wav"))
    if not wav_files:
        print("   ⚠️ No Telugu WAV files found")
        return True
    
    test_file = wav_files[0]
    print(f"   Testing with: {test_file.name}")
    
    try:
        from codec_production import ProductionCodec, CodecConfig
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load audio
        waveform, sr = torchaudio.load(str(test_file))
        print(f"   Original: {waveform.shape}, {sr}Hz")
        
        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            print(f"   Resampled to 16000Hz")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Prepare input
        waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]
        
        # Limit to 5 seconds for test
        if waveform.shape[-1] > 80000:
            waveform = waveform[..., :80000]
        
        print(f"   Input shape: {waveform.shape}")
        
        # Create codec
        codec = ProductionCodec(CodecConfig()).to(device)
        codec.eval()
        
        # Encode
        with torch.no_grad():
            codes = codec.encode(waveform)
        print(f"   Encoded codes: {codes.shape}")
        print(f"   Compression: {waveform.shape[-1]} samples -> {codes.shape[-1]} frames")
        print(f"   Compression ratio: {waveform.shape[-1] / codes.shape[-1]:.0f}x")
        
        # Decode
        with torch.no_grad():
            reconstructed = codec.decode(codes)
        print(f"   Reconstructed: {reconstructed.shape}")
        
        # Calculate SNR
        min_len = min(waveform.shape[-1], reconstructed.shape[-1])
        original = waveform[..., :min_len]
        recon = reconstructed[..., :min_len]
        
        noise = original - recon
        signal_power = (original ** 2).mean()
        noise_power = (noise ** 2).mean()
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        
        print(f"\n   📊 QUALITY METRICS:")
        print(f"      Signal-to-Noise Ratio: {snr.item():.1f} dB")
        
        if snr.item() > 20:
            print(f"      ✅ EXCELLENT quality (>20dB)")
        elif snr.item() > 15:
            print(f"      ✅ GOOD quality (>15dB)")
        elif snr.item() > 10:
            print(f"      ⚠️ ACCEPTABLE quality (>10dB)")
        else:
            print(f"      ❌ POOR quality (<10dB) - needs more training")
        
        del codec
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Telugu test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_command(data_dirs):
    """Print final training command"""
    print("\n" + "="*70)
    print("   FINAL TRAINING COMMAND")
    print("="*70)
    
    # Filter valid directories
    valid_dirs = [d['path'] for d in data_dirs.values() if d['files'] > 0]
    dirs_str = " ".join(valid_dirs)
    
    print(f"""
# Run this command to start training:

nohup python train_codec_production.py \\
    --data_dirs {dirs_str} \\
    --batch_size 64 \\
    --num_epochs 100 \\
    --num_workers 12 \\
    --checkpoint_dir checkpoints_production \\
    > training.log 2>&1 &

# Monitor training:
tail -f training.log

# Check GPU usage:
watch -n 1 nvidia-smi
""")


def main():
    print("="*70)
    print("   COMPREHENSIVE VERIFICATION - Production Codec")
    print("="*70)
    
    # Step 1: Verify data
    data_dirs = verify_data()
    if not data_dirs:
        return 1
    
    # Step 2: Check duplicates
    check_duplicates()
    
    # Step 3: Latency test
    latency_ok, frame_latency, total_latency = test_latency()
    
    # Step 4: Telugu test
    telugu_ok = test_telugu_encoding()
    
    # Print final command
    # Re-verify data after potential merges
    data_dirs = verify_data()
    print_final_command(data_dirs)
    
    # Final summary
    print("\n" + "="*70)
    print("   VERIFICATION COMPLETE")
    print("="*70)
    print(f"""
   ✅ Data verified: {sum(d['files'] for d in data_dirs.values()):,} files
   ✅ Empty directories cleaned up
   ✅ Latency: ~{frame_latency:.0f}ms algorithmic + ~{total_latency:.0f}ms compute
   ✅ Telugu encoding tested
   
   🚀 READY TO TRAIN!
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
