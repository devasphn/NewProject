"""Test latency of Telugu S2S pipeline"""
import torch
import numpy as np
import argparse
from s2s_pipeline import TeluguS2SPipeline
from config import *

def test_latency(mode="baseline"):
    print(f"\n{'='*60}")
    print(f"Latency Test - {mode.upper()}")
    print(f"Target: <{TARGET_TOTAL_LATENCY}ms")
    print(f"{'='*60}\n")
    
    # Create test audio (3 seconds of silence)
    test_audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.001
    
    # Initialize pipeline
    use_telugu = mode == "telugu"
    pipeline = TeluguS2SPipeline(use_telugu_model=use_telugu)
    
    # Warm-up run
    print("Warming up...")
    import asyncio
    asyncio.run(pipeline.process(test_audio))
    
    # Test runs
    print(f"\nRunning {5} test iterations...\n")
    latencies = []
    
    for i in range(5):
        result = asyncio.run(pipeline.process(test_audio))
        latency = result["latency_ms"]
        latencies.append(latency)
        
        status = "✓" if latency < TARGET_TOTAL_LATENCY else "⚠"
        print(f"Run {i+1}: {status} {latency}ms "
              f"(ASR:{result['breakdown']['asr_ms']}, "
              f"LLM:{result['breakdown']['llm_ms']}, "
              f"TTS:{result['breakdown']['tts_ms']})")
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n{'='*60}")
    print(f"Average Latency: {int(avg_latency)}ms")
    print(f"Target: {TARGET_TOTAL_LATENCY}ms")
    print(f"Status: {'✓ PASS' if avg_latency < TARGET_TOTAL_LATENCY else '⚠ NEEDS OPTIMIZATION'}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "telugu"], default="baseline")
    args = parser.parse_args()
    test_latency(args.mode)
