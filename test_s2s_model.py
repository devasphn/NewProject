#!/usr/bin/env python3
"""
Quick test script to verify S2S model works correctly
Run this BEFORE full training to catch issues early
"""

import torch
import sys

# Add path
sys.path.insert(0, '/workspace/NewProject')

def test_s2s_model():
    """Test S2S model with dummy data"""
    print("=" * 60)
    print("TESTING S2S TRANSFORMER MODEL")
    print("=" * 60)
    
    from s2s_transformer import TeluguS2STransformer, S2SConfig
    
    # Create config with compatible dimensions
    config = S2SConfig(
        hidden_dim=512,
        num_heads=8,  # 512/8 = 64 (clean!)
        num_encoder_layers=6,
        num_decoder_layers=6,
        use_flash_attn=False  # Disable for testing
    )
    
    print(f"Config: hidden_dim={config.hidden_dim}, num_heads={config.num_heads}")
    print(f"Head dim: {config.hidden_dim // config.num_heads}")
    print(f"Quantizer embed dim: {config.hidden_dim // config.num_quantizers}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = TeluguS2STransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 50  # Time steps
    num_quantizers = 8
    
    # Input codes: [B, Q, T]
    input_codes = torch.randint(0, config.vocab_size, (batch_size, num_quantizers, seq_len)).to(device)
    target_codes = torch.randint(0, config.vocab_size, (batch_size, num_quantizers, seq_len)).to(device)
    speaker_ids = torch.randint(0, config.num_speakers, (batch_size,)).to(device)
    emotion_ids = torch.randint(0, config.num_emotions, (batch_size,)).to(device)
    
    print(f"\nInput shapes:")
    print(f"  input_codes: {input_codes.shape}")
    print(f"  target_codes: {target_codes.shape}")
    print(f"  speaker_ids: {speaker_ids.shape}")
    print(f"  emotion_ids: {emotion_ids.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.train()
    
    try:
        loss = model(input_codes, target_codes, speaker_ids, emotion_ids)
        print(f"✅ Forward pass SUCCESS!")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print("\nRunning backward pass...")
    try:
        loss.backward()
        print(f"✅ Backward pass SUCCESS!")
    except Exception as e:
        print(f"❌ Backward pass FAILED: {e}")
        return False
    
    # Test generation
    print("\nTesting streaming generation...")
    model.eval()
    try:
        test_input = input_codes[:1]  # Single sample
        test_speaker = speaker_ids[:1]
        test_emotion = emotion_ids[:1]
        
        generated = []
        for i, chunk in enumerate(model.generate_streaming(
            test_input, test_speaker, test_emotion, max_new_tokens=5
        )):
            generated.append(chunk)
            if i >= 2:  # Just test a few
                break
        
        print(f"✅ Generation SUCCESS! Generated {len(generated)} chunks")
    except Exception as e:
        print(f"❌ Generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Model is ready for training.")
    print("=" * 60)
    return True


def test_with_codec():
    """Test S2S model with actual codec"""
    print("\n" + "=" * 60)
    print("TESTING WITH ACTUAL CODEC")
    print("=" * 60)
    
    import torchaudio
    from pathlib import Path
    from telugu_codec_fixed import TeluCodec
    from s2s_transformer import TeluguS2STransformer, S2SConfig
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load codec
    codec_path = "/workspace/models/codec/best_codec.pt"
    if not Path(codec_path).exists():
        print(f"Codec not found at {codec_path}, skipping codec test")
        return True
    
    codec = TeluCodec().to(device)
    checkpoint = torch.load(codec_path, map_location=device)
    if 'codec_state_dict' in checkpoint:
        codec.load_state_dict(checkpoint['codec_state_dict'])
    codec.eval()
    print("✅ Codec loaded")
    
    # Find a sample audio
    sample_dirs = [
        "/workspace/telugu_data/openslr",
        "/workspace/telugu_data/indictts/audio"
    ]
    sample_file = None
    for d in sample_dirs:
        files = list(Path(d).glob("*.wav"))[:1] if Path(d).exists() else []
        if files:
            sample_file = str(files[0])
            break
    
    if not sample_file:
        print("No sample audio found, skipping")
        return True
    
    # Load audio
    waveform, sr = torchaudio.load(sample_file)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Trim to 2 seconds for testing
    max_samples = 32000
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    print(f"Audio shape: {waveform.shape}")
    
    # Encode with codec
    with torch.no_grad():
        codes = codec.encode(waveform)
    print(f"Codec output codes: {codes.shape}")
    print(f"Codes dtype: {codes.dtype}")
    
    # Convert to long
    codes = codes.long()
    
    # Create S2S model
    config = S2SConfig(
        hidden_dim=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        use_flash_attn=False
    )
    model = TeluguS2STransformer(config).to(device)
    model.train()
    
    # Forward pass with real codec codes
    speaker_id = torch.tensor([0], device=device)
    emotion_id = torch.tensor([0], device=device)
    
    try:
        loss = model(codes, codes, speaker_id, emotion_id)
        print(f"✅ Forward pass with REAL CODEC CODES: Loss = {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Codec integration test PASSED!")
    return True


if __name__ == "__main__":
    success1 = test_s2s_model()
    if success1:
        success2 = test_with_codec()
    
    if success1 and success2:
        print("\n🚀 Ready for S2S training!")
    else:
        print("\n❌ Tests failed. Please fix issues before training.")
        sys.exit(1)
