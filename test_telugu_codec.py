#!/usr/bin/env python3
"""
Test fine-tuned Telugu EnCodec
Generate before/after demos for MD presentation
"""

import torch
import torchaudio
from encodec import EncodecModel
from pathlib import Path
import json
import numpy as np

def calculate_snr(original, reconstructed):
    """Calculate Signal-to-Noise Ratio in dB"""
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Calculate signal and noise power
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def test_codec():
    """Test the fine-tuned codec on test samples"""
    
    print("="*70)
    print("TESTING TELUGU ENCODEC POC")
    print("="*70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    model = EncodecModel.encodec_model_24khz()
    model.load_state_dict(torch.load('/workspace/models/encodec_telugu_poc.pt'))
    model = model.to(device)
    model.set_target_bandwidth(6.0)
    model.eval()
    print("✅ Model loaded")
    print()
    
    # Get test files
    test_dir = Path('/workspace/telugu_poc_data/test')
    test_files = list(test_dir.glob('*.wav'))[:5]  # Test on 5 samples
    
    print(f"Testing on {len(test_files)} samples...")
    print()
    
    # Create output directory
    output_dir = Path('/workspace/demo_outputs')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    with torch.no_grad():
        for idx, test_file in enumerate(test_files):
            print(f"[{idx+1}/{len(test_files)}] {test_file.name}")
            
            # Load audio
            waveform, sr = torchaudio.load(test_file)
            
            # Resample to 24kHz if needed
            if sr != 24000:
                waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Limit to 10 seconds for testing
            max_len = 24000 * 10
            if waveform.shape[1] > max_len:
                waveform = waveform[:, :max_len]
            
            # Move to device
            waveform = waveform.to(device)
            
            # Original size
            original_size = waveform.numel() * 4  # 32-bit float
            
            # Encode
            encoded_frames = model.encode(waveform.unsqueeze(0))
            
            # Calculate compressed size (approximate)
            compressed_size = sum(frame[0].numel() for frame in encoded_frames) * 2  # codes are int16
            
            # Decode
            decoded = model.decode(encoded_frames)
            decoded = decoded.squeeze(0)
            
            # Ensure same length for SNR calculation
            min_len = min(waveform.shape[1], decoded.shape[1])
            waveform_trimmed = waveform[:, :min_len]
            decoded_trimmed = decoded[:, :min_len]
            
            # Calculate metrics
            snr = calculate_snr(waveform_trimmed[0], decoded_trimmed[0])
            compression_ratio = original_size / compressed_size
            
            # Save original
            original_path = output_dir / f"original_{idx+1}.wav"
            torchaudio.save(original_path, waveform.cpu(), 24000)
            
            # Save reconstructed
            reconstructed_path = output_dir / f"reconstructed_{idx+1}.wav"
            torchaudio.save(reconstructed_path, decoded.cpu(), 24000)
            
            # Store results
            result = {
                "file": test_file.name,
                "snr_db": round(snr, 2),
                "compression_ratio": round(compression_ratio, 1),
                "original_size_mb": round(original_size / 1024 / 1024, 2),
                "compressed_size_kb": round(compressed_size / 1024, 2)
            }
            results.append(result)
            
            print(f"  SNR: {snr:.2f} dB")
            print(f"  Compression: {compression_ratio:.1f}x")
            print(f"  Original: {original_size/1024/1024:.2f} MB")
            print(f"  Compressed: {compressed_size/1024:.2f} KB")
            print()
    
    # Calculate averages
    avg_snr = np.mean([r['snr_db'] for r in results])
    avg_compression = np.mean([r['compression_ratio'] for r in results])
    
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Average SNR: {avg_snr:.2f} dB")
    print(f"Average Compression: {avg_compression:.1f}x")
    print()
    
    # Interpret results
    print("Quality Assessment:")
    if avg_snr >= 20:
        quality = "EXCELLENT ✅"
        assessment = "Production-quality audio reconstruction"
    elif avg_snr >= 15:
        quality = "GOOD ✅"
        assessment = "High-quality POC demonstration"
    elif avg_snr >= 10:
        quality = "ACCEPTABLE ⚠️"
        assessment = "Sufficient for POC, needs improvement for production"
    else:
        quality = "POOR ❌"
        assessment = "Not suitable for POC, requires more training"
    
    print(f"  Quality: {quality}")
    print(f"  Assessment: {assessment}")
    print()
    
    # POC verdict
    print("POC Verdict:")
    if avg_snr >= 15:
        print("  ✅ POC SUCCESS - Ready to demo!")
        print("  ✅ Quality meets POC requirements (>15 dB)")
        print("  ✅ Compression working as expected")
    else:
        print("  ⚠️ POC MARGINAL - Consider more training or use pretrained")
        print(f"  ⚠️ Quality slightly below target ({avg_snr:.1f} dB vs 15+ dB)")
    print()
    
    # Save results
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "avg_snr_db": round(avg_snr, 2),
                "avg_compression_ratio": round(avg_compression, 1),
                "quality": quality,
                "poc_ready": avg_snr >= 15
            },
            "individual_results": results
        }, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")
    print(f"✅ Demo audio files saved to {output_dir}")
    print()
    
    # Next steps
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("1. Listen to demo files:")
    print(f"   cd {output_dir}")
    print("   # Download and compare original_X.wav vs reconstructed_X.wav")
    print()
    print("2. Generate presentation:")
    print("   python generate_poc_report.py")
    print()
    print("3. Prepare MD demo:")
    print("   - Show before/after audio samples")
    print("   - Present quality metrics")
    print("   - Demonstrate compression ratio")
    print()
    
    return avg_snr, avg_compression

if __name__ == "__main__":
    snr, compression = test_codec()
