#!/usr/bin/env python3
"""
Latency benchmarking for Telugu S2S system
Tests streaming and turn-based modes
"""

import torch
import time
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from telugu_codec import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig
from speaker_embeddings import SpeakerEmbeddingSystem

class LatencyBenchmark:
    """Comprehensive latency testing"""
    
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load models
        self._load_models()
        
        # Results storage
        self.results = {
            "stream_mode": {},
            "turn_mode": {},
            "component_breakdown": {},
            "statistics": {}
        }
    
    def _load_models(self):
        """Load all models for benchmarking"""
        print("Loading models...")
        
        # Codec
        self.codec = TeluCodec().to(self.device)
        codec_checkpoint = torch.load(
            self.model_dir / "best_codec.pt",
            map_location=self.device
        )
        self.codec.load_state_dict(codec_checkpoint["model_state"])
        self.codec.eval()
        
        # S2S Model
        config = S2SConfig(use_flash_attn=self.device.type == "cuda")
        self.s2s = TeluguS2STransformer(config).to(self.device)
        s2s_checkpoint = torch.load(
            self.model_dir / "s2s_best.pt",
            map_location=self.device
        )
        self.s2s.load_state_dict(s2s_checkpoint["model_state"])
        self.s2s.eval()
        
        # Speaker system
        self.speaker_system = SpeakerEmbeddingSystem().to(self.device)
        speaker_path = self.model_dir / "speaker_embeddings.json"
        if speaker_path.exists():
            self.speaker_system.load_embeddings(str(speaker_path))
        
        # Compile if available
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            print("Compiling models with torch.compile()...")
            self.codec = torch.compile(self.codec, mode='reduce-overhead')
            self.s2s = torch.compile(self.s2s, mode='reduce-overhead')
        
        print("✓ Models loaded and ready")
    
    def warmup(self, num_iterations: int = 5):
        """Warmup GPU and models"""
        print("Warming up...")
        
        for _ in range(num_iterations):
            with torch.no_grad():
                dummy_audio = torch.randn(1, 1, 1600).to(self.device)
                codes = self.codec.encode(dummy_audio)
                
                for chunk in self.s2s.generate_streaming(
                    codes,
                    torch.tensor([0], device=self.device),
                    torch.tensor([0], device=self.device),
                    max_new_tokens=5
                ):
                    break
                
                self.codec.decode(chunk)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        print("✓ Warmup complete")
    
    def benchmark_streaming_mode(
        self,
        num_tests: int = 20,
        chunk_sizes_ms: List[int] = [50, 100, 200]
    ) -> Dict:
        """Benchmark streaming mode with different chunk sizes"""
        print("\n" + "="*50)
        print("STREAMING MODE BENCHMARK")
        print("="*50)
        
        results = {}
        
        for chunk_ms in chunk_sizes_ms:
            chunk_samples = int(16000 * chunk_ms / 1000)
            latencies = []
            
            print(f"\nTesting {chunk_ms}ms chunks ({chunk_samples} samples)...")
            
            for i in range(num_tests):
                # Generate test audio
                audio = torch.randn(1, 1, chunk_samples).to(self.device)
                
                # Measure end-to-end latency
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    # Encode
                    codes = self.codec.encode(audio)
                    
                    # Generate first response chunk
                    for chunk in self.s2s.generate_streaming(
                        codes,
                        torch.tensor([0], device=self.device),
                        torch.tensor([0], device=self.device),
                        max_new_tokens=10
                    ):
                        response_codes = chunk
                        break
                    
                    # Decode
                    response_audio = self.codec.decode(response_codes)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                if (i + 1) % 5 == 0:
                    print(f"  Test {i+1}/{num_tests}: {latency:.1f}ms")
            
            # Calculate statistics
            results[f"{chunk_ms}ms"] = {
                "latencies": latencies,
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            }
            
            print(f"\nResults for {chunk_ms}ms chunks:")
            print(f"  Mean: {results[f'{chunk_ms}ms']['mean']:.1f}ms")
            print(f"  Std:  {results[f'{chunk_ms}ms']['std']:.1f}ms")
            print(f"  Min:  {results[f'{chunk_ms}ms']['min']:.1f}ms")
            print(f"  Max:  {results[f'{chunk_ms}ms']['max']:.1f}ms")
            print(f"  P50:  {results[f'{chunk_ms}ms']['p50']:.1f}ms")
            print(f"  P95:  {results[f'{chunk_ms}ms']['p95']:.1f}ms")
            print(f"  P99:  {results[f'{chunk_ms}ms']['p99']:.1f}ms")
        
        self.results["stream_mode"] = results
        return results
    
    def benchmark_turn_mode(
        self,
        num_tests: int = 20,
        utterance_lengths_sec: List[float] = [0.5, 1.0, 2.0]
    ) -> Dict:
        """Benchmark turn-based mode with different utterance lengths"""
        print("\n" + "="*50)
        print("TURN-BASED MODE BENCHMARK")
        print("="*50)
        
        results = {}
        
        for length_sec in utterance_lengths_sec:
            length_samples = int(16000 * length_sec)
            latencies = []
            
            print(f"\nTesting {length_sec}s utterances ({length_samples} samples)...")
            
            for i in range(num_tests):
                # Generate test audio
                audio = torch.randn(1, 1, length_samples).to(self.device)
                
                # Measure end-to-end latency
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    # Encode full utterance
                    codes = self.codec.encode(audio)
                    
                    # Generate complete response
                    max_tokens = int(length_sec * 50)  # Proportional to input
                    response_codes = self.s2s.generate(
                        codes,
                        torch.tensor([0], device=self.device),
                        torch.tensor([0], device=self.device),
                        max_new_tokens=max_tokens,
                        temperature=0.7
                    )
                    
                    # Decode complete response
                    response_audio = self.codec.decode(response_codes)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                if (i + 1) % 5 == 0:
                    print(f"  Test {i+1}/{num_tests}: {latency:.1f}ms")
            
            # Calculate statistics
            results[f"{length_sec}s"] = {
                "latencies": latencies,
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            }
            
            print(f"\nResults for {length_sec}s utterances:")
            print(f"  Mean: {results[f'{length_sec}s']['mean']:.1f}ms")
            print(f"  Std:  {results[f'{length_sec}s']['std']:.1f}ms")
            print(f"  Min:  {results[f'{length_sec}s']['min']:.1f}ms")
            print(f"  Max:  {results[f'{length_sec}s']['max']:.1f}ms")
            print(f"  P50:  {results[f'{length_sec}s']['p50']:.1f}ms")
            print(f"  P95:  {results[f'{length_sec}s']['p95']:.1f}ms")
        
        self.results["turn_mode"] = results
        return results
    
    def benchmark_components(self, num_tests: int = 50) -> Dict:
        """Benchmark individual components"""
        print("\n" + "="*50)
        print("COMPONENT BREAKDOWN")
        print("="*50)
        
        # Test audio (100ms)
        audio = torch.randn(1, 1, 1600).to(self.device)
        
        components = {
            "codec_encode": [],
            "codec_decode": [],
            "s2s_encode": [],
            "s2s_decode_first": [],
            "speaker_embedding": []
        }
        
        for _ in range(num_tests):
            with torch.no_grad():
                # Codec encode
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                codes = self.codec.encode(audio)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                components["codec_encode"].append((time.perf_counter() - start) * 1000)
                
                # S2S encode
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                encoder_out = self.s2s.encode(
                    codes,
                    torch.tensor([0], device=self.device),
                    torch.tensor([0], device=self.device)
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                components["s2s_encode"].append((time.perf_counter() - start) * 1000)
                
                # S2S decode first token
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for chunk in self.s2s.generate_streaming(
                    codes,
                    torch.tensor([0], device=self.device),
                    torch.tensor([0], device=self.device),
                    max_new_tokens=1
                ):
                    first_token = chunk
                    break
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                components["s2s_decode_first"].append((time.perf_counter() - start) * 1000)
                
                # Codec decode
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                decoded = self.codec.decode(first_token)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                components["codec_decode"].append((time.perf_counter() - start) * 1000)
                
                # Speaker embedding
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                speaker_embed = self.speaker_system(
                    torch.tensor([0], device=self.device)
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                components["speaker_embedding"].append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        results = {}
        total_latency = 0
        
        print("\nComponent Latencies:")
        for component, latencies in components.items():
            mean_latency = np.mean(latencies)
            results[component] = {
                "mean": mean_latency,
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies)
            }
            total_latency += mean_latency
            
            print(f"  {component:20s}: {mean_latency:6.2f}ms ± {np.std(latencies):.2f}ms")
        
        print(f"\n  {'TOTAL':20s}: {total_latency:6.2f}ms")
        
        self.results["component_breakdown"] = results
        return results
    
    def plot_results(self, save_path: str = "latency_benchmark.png"):
        """Plot benchmark results"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Stream mode comparison
            ax = axes[0, 0]
            stream_data = self.results.get("stream_mode", {})
            if stream_data:
                labels = list(stream_data.keys())
                means = [data["mean"] for data in stream_data.values()]
                stds = [data["std"] for data in stream_data.values()]
                
                ax.bar(labels, means, yerr=stds, capsize=5)
                ax.axhline(y=150, color='r', linestyle='--', label='Target (150ms)')
                ax.set_title("Streaming Mode Latency")
                ax.set_ylabel("Latency (ms)")
                ax.legend()
            
            # Turn mode comparison
            ax = axes[0, 1]
            turn_data = self.results.get("turn_mode", {})
            if turn_data:
                labels = list(turn_data.keys())
                means = [data["mean"] for data in turn_data.values()]
                stds = [data["std"] for data in turn_data.values()]
                
                ax.bar(labels, means, yerr=stds, capsize=5)
                ax.set_title("Turn Mode Latency")
                ax.set_ylabel("Latency (ms)")
            
            # Component breakdown
            ax = axes[1, 0]
            component_data = self.results.get("component_breakdown", {})
            if component_data:
                labels = list(component_data.keys())
                means = [data["mean"] for data in component_data.values()]
                
                ax.barh(labels, means)
                ax.set_title("Component Latency Breakdown")
                ax.set_xlabel("Latency (ms)")
            
            # Latency distribution
            ax = axes[1, 1]
            if "100ms" in stream_data:
                latencies = stream_data["100ms"]["latencies"]
                ax.hist(latencies, bins=20, edgecolor='black')
                ax.axvline(x=150, color='r', linestyle='--', label='Target')
                ax.set_title("Latency Distribution (100ms chunks)")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Count")
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"\n✓ Plot saved to {save_path}")
            
        except ImportError:
            print("\n⚠ Matplotlib not available, skipping plot")
    
    def save_results(self, output_file: str = "latency_results.json"):
        """Save detailed results to JSON"""
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_converted = convert_types(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")
    
    def run_full_benchmark(
        self,
        num_tests: int = 20,
        modes: List[str] = ["stream", "turn", "components"]
    ):
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("TELUGU S2S LATENCY BENCHMARK")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Number of tests per configuration: {num_tests}")
        
        # Warmup
        self.warmup()
        
        # Run benchmarks
        if "stream" in modes:
            self.benchmark_streaming_mode(num_tests)
        
        if "turn" in modes:
            self.benchmark_turn_mode(num_tests)
        
        if "components" in modes:
            self.benchmark_components(num_tests * 2)
        
        # Summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Check if target is met
        stream_100ms = self.results.get("stream_mode", {}).get("100ms", {})
        if stream_100ms:
            mean_latency = stream_100ms.get("mean", 999)
            p95_latency = stream_100ms.get("p95", 999)
            
            print(f"\nStreaming Mode (100ms chunks):")
            print(f"  Mean latency: {mean_latency:.1f}ms")
            print(f"  P95 latency:  {p95_latency:.1f}ms")
            
            if mean_latency < 150:
                print("\n✅ TARGET ACHIEVED: Mean latency < 150ms")
            else:
                print(f"\n⚠️ Target missed: {mean_latency:.1f}ms > 150ms")
            
            if p95_latency < 200:
                print("✅ P95 latency < 200ms (good consistency)")
            else:
                print(f"⚠️ P95 latency high: {p95_latency:.1f}ms")
        
        # Save results
        self.save_results()
        self.plot_results()
        
        print("\n" + "="*60)
        print("Benchmark complete!")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Telugu S2S Latency")
    parser.add_argument("--model_dir", type=str, default="/workspace/models",
                        help="Directory containing models")
    parser.add_argument("--num_tests", type=int, default=20,
                        help="Number of tests per configuration")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["stream", "turn", "components", "all"],
                        help="Benchmark mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = LatencyBenchmark(args.model_dir, args.device)
    
    # Determine modes
    if args.mode == "all":
        modes = ["stream", "turn", "components"]
    else:
        modes = [args.mode]
    
    # Run benchmark
    benchmark.run_full_benchmark(args.num_tests, modes)

if __name__ == "__main__":
    main()