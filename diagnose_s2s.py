#!/usr/bin/env python3
"""
Deep Diagnostic for S2S Model
=============================

This script shows EXACTLY what's happening at each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time

print("=" * 70)
print("🔬 DEEP S2S DIAGNOSTIC")
print("=" * 70)

# ============================================================
# 1. CHECK TRAINING DATA
# ============================================================
print("\n" + "=" * 70)
print("📊 STEP 1: Checking Training Data")
print("=" * 70)

data_dir = Path("data/telugu_conversations")
pairs = list(data_dir.glob("pair_*"))
print(f"Total pairs found: {len(pairs)}")

if len(pairs) > 0:
    # Check first 3 pairs
    for i, pair_dir in enumerate(pairs[:3]):
        print(f"\n--- Pair {i}: {pair_dir.name} ---")
        
        # Check files
        q_wav = pair_dir / "question.wav"
        a_wav = pair_dir / "answer.wav"
        q_codes = pair_dir / "question_codes.pt"
        a_codes = pair_dir / "answer_codes.pt"
        meta = pair_dir / "metadata.json"
        
        if meta.exists():
            with open(meta) as f:
                m = json.load(f)
            print(f"  Question: {m.get('question_text', 'N/A')[:50]}...")
            print(f"  Answer: {m.get('answer_text', 'N/A')[:50]}...")
        
        if q_codes.exists():
            qc = torch.load(q_codes)
            print(f"  Q codes shape: {qc.shape}, dtype: {qc.dtype}")
            print(f"  Q codes range: [{qc.min().item()}, {qc.max().item()}]")
            print(f"  Q codes mean: {qc.float().mean().item():.2f}, std: {qc.float().std().item():.2f}")
        
        if a_codes.exists():
            ac = torch.load(a_codes)
            print(f"  A codes shape: {ac.shape}, dtype: {ac.dtype}")
            print(f"  A codes range: [{ac.min().item()}, {ac.max().item()}]")
            print(f"  A codes mean: {ac.float().mean().item():.2f}, std: {ac.float().std().item():.2f}")
        
        if q_wav.exists():
            import soundfile as sf
            audio, sr = sf.read(q_wav)
            print(f"  Q audio: {len(audio)} samples, {len(audio)/sr:.2f}s")
        
        if a_wav.exists():
            audio, sr = sf.read(a_wav)
            print(f"  A audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

# ============================================================
# 2. CHECK CODEC
# ============================================================
print("\n" + "=" * 70)
print("📦 STEP 2: Testing Codec (Encode → Decode)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load codec
from telugu_codec_fixed import TeluCodec
codec = TeluCodec().to(device)
checkpoint = torch.load("best_codec.pt", map_location=device, weights_only=False)
if 'codec_state_dict' in checkpoint:
    codec.load_state_dict(checkpoint['codec_state_dict'])
else:
    codec.load_state_dict(checkpoint)
codec.eval()
print("✅ Codec loaded")

# Test with a real audio file
test_audio_path = None
for pair_dir in pairs[:1]:
    q_wav = pair_dir / "question.wav"
    if q_wav.exists():
        test_audio_path = q_wav
        break

if test_audio_path:
    print(f"\nTesting with: {test_audio_path}")
    audio, sr = sf.read(test_audio_path)
    print(f"  Original audio: {len(audio)} samples, range [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Encode
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = codec.encode(audio_t)
    print(f"  Encoded codes: shape {codes.shape}")
    print(f"  Codes range: [{codes.min().item()}, {codes.max().item()}]")
    print(f"  Codes distribution per quantizer:")
    for q in range(codes.shape[1]):
        q_codes = codes[0, q].cpu()
        print(f"    Q{q}: min={q_codes.min().item()}, max={q_codes.max().item()}, unique={len(q_codes.unique())}")
    
    # Decode
    with torch.no_grad():
        reconstructed = codec.decode(codes)
    recon_audio = reconstructed.squeeze().cpu().numpy()
    print(f"  Reconstructed: {len(recon_audio)} samples, range [{recon_audio.min():.3f}, {recon_audio.max():.3f}]")
    
    # Save for listening
    sf.write("diagnostic_original.wav", audio, 16000)
    sf.write("diagnostic_reconstructed.wav", recon_audio, 16000)
    print("  💾 Saved: diagnostic_original.wav, diagnostic_reconstructed.wav")
    
    # Calculate reconstruction error
    min_len = min(len(audio), len(recon_audio))
    mse = np.mean((audio[:min_len] - recon_audio[:min_len])**2)
    print(f"  MSE: {mse:.6f}")

# ============================================================
# 3. CHECK S2S MODEL
# ============================================================
print("\n" + "=" * 70)
print("🧠 STEP 3: Testing S2S Model")
print("=" * 70)

@dataclass
class TrainConfig:
    data_dir: str = "data/telugu_conversations"
    codec_path: str = "best_codec.pt"
    hidden_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_quantizers: int = 8
    vocab_size: int = 1024
    max_seq_len: int = 2048
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    save_every: int = 5
    output_dir: str = "checkpoints/s2s_conversation"
    device: str = "cuda"

# Load S2S
from train_s2s_conversation import ConversationS2S
config = TrainConfig()
s2s = ConversationS2S(config).to(device)

s2s_path = "checkpoints/s2s_conversation/best_conversation_s2s.pt"
checkpoint = torch.load(s2s_path, map_location=device, weights_only=False)
if 'model_state' in checkpoint:
    s2s.load_state_dict(checkpoint['model_state'])
else:
    s2s.load_state_dict(checkpoint)
s2s.eval()

print(f"✅ S2S loaded from {s2s_path}")
print(f"  Epoch trained: {checkpoint.get('epoch', 'unknown')}")
print(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")

# Get a real input from training data
test_pair = pairs[0]
q_codes_path = test_pair / "question_codes.pt"
a_codes_path = test_pair / "answer_codes.pt"

input_codes = torch.load(q_codes_path).to(device).long()
target_codes = torch.load(a_codes_path).to(device).long()

print(f"\nTest input from training data:")
print(f"  Input codes shape: {input_codes.shape}")
print(f"  Target codes shape: {target_codes.shape}")

# ============================================================
# 4. TEST S2S GENERATION (SHORT)
# ============================================================
print("\n" + "=" * 70)
print("🔄 STEP 4: S2S Generation Test (SHORT - 50 steps)")
print("=" * 70)

# Generate with SHORT max_len to test
print("Generating 50 tokens...")
start = time.perf_counter()

with torch.no_grad():
    # Short generation for testing
    output_codes = s2s.generate(input_codes, max_len=50, temperature=0.8)

gen_time = time.perf_counter() - start
print(f"Generation time: {gen_time*1000:.1f}ms for 50 tokens ({gen_time*1000/50:.1f}ms per token)")
print(f"Output codes shape: {output_codes.shape}")
print(f"Output codes range: [{output_codes.min().item()}, {output_codes.max().item()}]")

# Check code distribution
print("\nOutput codes distribution per quantizer:")
for q in range(output_codes.shape[1]):
    q_out = output_codes[0, q].cpu()
    print(f"  Q{q}: min={q_out.min().item()}, max={q_out.max().item()}, unique={len(q_out.unique())}")

# Compare with target distribution
print("\nTarget codes distribution per quantizer:")
for q in range(target_codes.shape[1]):
    q_tgt = target_codes[0, q].cpu()
    print(f"  Q{q}: min={q_tgt.min().item()}, max={q_tgt.max().item()}, unique={len(q_tgt.unique())}")

# ============================================================
# 5. DECODE AND LISTEN
# ============================================================
print("\n" + "=" * 70)
print("🔊 STEP 5: Decode S2S Output")
print("=" * 70)

# Decode the generated codes
with torch.no_grad():
    generated_audio = codec.decode(output_codes)
gen_audio = generated_audio.squeeze().cpu().numpy()

print(f"Generated audio: {len(gen_audio)} samples, {len(gen_audio)/16000:.2f}s")
print(f"Audio range: [{gen_audio.min():.3f}, {gen_audio.max():.3f}]")
print(f"Audio mean: {gen_audio.mean():.6f}")
print(f"Audio std: {gen_audio.std():.6f}")

# Check for problems
if gen_audio.std() < 0.001:
    print("⚠️  WARNING: Audio has very low variance - likely silence or DC!")
if gen_audio.max() > 10 or gen_audio.min() < -10:
    print("⚠️  WARNING: Audio has extreme values - likely unstable generation!")
if np.isnan(gen_audio).any():
    print("❌ ERROR: Audio contains NaN values!")

# Save
sf.write("diagnostic_s2s_output.wav", gen_audio, 16000)
print("💾 Saved: diagnostic_s2s_output.wav")

# Also decode the target for comparison
with torch.no_grad():
    target_audio = codec.decode(target_codes)
tgt_audio = target_audio.squeeze().cpu().numpy()
sf.write("diagnostic_target.wav", tgt_audio, 16000)
print("💾 Saved: diagnostic_target.wav (what S2S should produce)")

# ============================================================
# 6. PROBLEM ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("🔍 STEP 6: Problem Analysis")
print("=" * 70)

problems = []

# Check 1: Loss still high
val_loss = checkpoint.get('val_loss', 999)
if val_loss > 2.0:
    problems.append(f"High validation loss ({val_loss:.2f}) - model not well trained")
    
# Check 2: Not enough data
if len(pairs) < 500:
    problems.append(f"Only {len(pairs)} training pairs - need 500+ for good results")

# Check 3: Code distribution mismatch
out_codes_flat = output_codes[0].cpu().flatten()
tgt_codes_flat = target_codes[0].cpu().flatten()

out_mean = out_codes_flat.float().mean().item()
tgt_mean = tgt_codes_flat.float().mean().item()
if abs(out_mean - tgt_mean) > 200:
    problems.append(f"Code distribution mismatch: output mean={out_mean:.0f}, target mean={tgt_mean:.0f}")

# Check 4: Generation is slow
if gen_time > 1.0:
    problems.append(f"Generation too slow ({gen_time:.1f}s for 50 tokens)")

if problems:
    print("❌ PROBLEMS FOUND:")
    for p in problems:
        print(f"   • {p}")
else:
    print("✅ No obvious problems detected")

# ============================================================
# 7. RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS")
print("=" * 70)

print("""
1. TRAINING DATA:
   - Current: {pairs} pairs
   - Needed: 500-1000+ pairs for basic quality
   - Generate more: python generate_telugu_conversations.py --num_template 500 --num_llm 500

2. TRAINING TIME:
   - Current: 50 epochs on 100 pairs
   - Issue: Model hasn't learned enough patterns
   - Solution: More data + more epochs (100-200)

3. GENERATION SPEED:
   - Problem: Autoregressive generation is slow (one token at a time)
   - Solution: Use parallel decoding or smaller output length

4. AUDIO QUALITY:
   - Listen to diagnostic_*.wav files to verify
   - diagnostic_original.wav - Your input
   - diagnostic_reconstructed.wav - Codec quality check
   - diagnostic_target.wav - What S2S should produce
   - diagnostic_s2s_output.wav - What S2S actually produces

5. QUICK TEST:
   aplay diagnostic_reconstructed.wav  # Should sound good
   aplay diagnostic_s2s_output.wav      # Should sound like speech
""".format(pairs=len(pairs)))

print("\n" + "=" * 70)
print("🎧 LISTEN TO THE FILES:")
print("=" * 70)
print("""
Run these commands to listen:

# Check codec works:
aplay diagnostic_original.wav
aplay diagnostic_reconstructed.wav

# Check S2S output:
aplay diagnostic_target.wav      # Expected output
aplay diagnostic_s2s_output.wav  # Actual output

Or download them to your local machine to listen.
""")
