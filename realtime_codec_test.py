#!/usr/bin/env python3
"""
Real-time Telugu Codec Test
Speak into microphone → Encode → Decode → Play back with latency measurement

This tests the codec's real-time performance:
- Records audio chunks
- Encodes to discrete codes
- Decodes back to audio
- Plays reconstructed audio
- Measures end-to-end latency
"""

import torch
import numpy as np
import time
import argparse
import logging
import threading
import queue
from pathlib import Path

# Audio handling
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not installed. Run: pip install sounddevice")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

from telugu_codec_fixed import TeluCodec

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class RealtimeCodecTest:
    """Real-time codec latency tester"""
    
    def __init__(
        self,
        codec_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,  # 500ms chunks
        device: str = "cuda"
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"🔧 Sample rate: {sample_rate} Hz")
        logger.info(f"🔧 Chunk duration: {chunk_duration}s ({self.chunk_size} samples)")
        
        # Load codec
        self._load_codec(codec_path)
        
        # Audio queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Latency tracking
        self.latencies = []
        self.running = False
        
    def _load_codec(self, path: str):
        """Load the Telugu codec"""
        logger.info("📥 Loading Telugu Codec...")
        self.codec = TeluCodec().to(self.device)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'codec_state_dict' in checkpoint:
            self.codec.load_state_dict(checkpoint['codec_state_dict'])
        else:
            self.codec.load_state_dict(checkpoint)
        self.codec.eval()
        logger.info("✅ Codec loaded successfully")
        
        # Warmup
        logger.info("🔥 Warming up codec...")
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.chunk_size).to(self.device)
            for _ in range(3):
                codes = self.codec.encode(dummy)
                _ = self.codec.decode(codes)
        logger.info("✅ Warmup complete")
    
    @torch.no_grad()
    def process_chunk(self, audio_chunk: np.ndarray) -> tuple:
        """
        Process a single audio chunk through codec
        Returns: (reconstructed_audio, encode_time, decode_time)
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        audio_tensor = audio_tensor.to(self.device)
        
        # Normalize
        max_val = audio_tensor.abs().max()
        if max_val > 0:
            audio_tensor = audio_tensor / max_val * 0.9
        
        # Encode
        encode_start = time.perf_counter()
        codes = self.codec.encode(audio_tensor)
        encode_time = (time.perf_counter() - encode_start) * 1000
        
        # Decode
        decode_start = time.perf_counter()
        reconstructed = self.codec.decode(codes)
        decode_time = (time.perf_counter() - decode_start) * 1000
        
        # Convert back to numpy
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        
        # Match original length
        if len(reconstructed_np) > len(audio_chunk):
            reconstructed_np = reconstructed_np[:len(audio_chunk)]
        elif len(reconstructed_np) < len(audio_chunk):
            reconstructed_np = np.pad(reconstructed_np, (0, len(audio_chunk) - len(reconstructed_np)))
        
        return reconstructed_np, encode_time, decode_time
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Sounddevice callback for real-time processing"""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Get input audio
        audio_chunk = indata[:, 0].copy()  # Mono
        
        # Process through codec
        start_time = time.perf_counter()
        reconstructed, encode_time, decode_time = self.process_chunk(audio_chunk)
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Track latency
        self.latencies.append({
            'encode': encode_time,
            'decode': decode_time,
            'total': total_time
        })
        
        # Output reconstructed audio
        outdata[:, 0] = reconstructed
    
    def run_realtime(self, duration: float = 30.0):
        """
        Run real-time codec test
        Speaks into mic, hears reconstructed audio back
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("❌ sounddevice not available!")
            return
        
        logger.info("\n" + "="*60)
        logger.info("🎤 REAL-TIME CODEC TEST")
        logger.info("="*60)
        logger.info(f"Duration: {duration}s")
        logger.info("Speak into your microphone - you'll hear the reconstructed audio")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*60 + "\n")
        
        self.latencies = []
        
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                callback=self.audio_callback
            ):
                logger.info("🎙️ Recording... Speak now!")
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    time.sleep(0.5)
                    
                    # Print live stats
                    if self.latencies:
                        recent = self.latencies[-10:]
                        avg_encode = np.mean([l['encode'] for l in recent])
                        avg_decode = np.mean([l['decode'] for l in recent])
                        avg_total = np.mean([l['total'] for l in recent])
                        
                        print(f"\r⏱️ Latency - Encode: {avg_encode:.1f}ms | "
                              f"Decode: {avg_decode:.1f}ms | "
                              f"Total: {avg_total:.1f}ms | "
                              f"Chunks: {len(self.latencies)}", end="", flush=True)
                
                print()  # New line
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Stopped by user")
        
        self._print_results()
    
    def run_file_test(self, input_file: str, output_file: str = "reconstructed.wav"):
        """
        Test codec on a file and measure latency
        """
        logger.info("\n" + "="*60)
        logger.info("📁 FILE-BASED CODEC TEST")
        logger.info("="*60)
        
        # Load audio
        import torchaudio
        waveform, sr = torchaudio.load(input_file)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        audio_np = waveform.squeeze().numpy()
        duration = len(audio_np) / self.sample_rate
        
        logger.info(f"📄 Input: {input_file}")
        logger.info(f"⏱️ Duration: {duration:.2f}s")
        
        # Process in chunks
        num_chunks = len(audio_np) // self.chunk_size
        reconstructed_chunks = []
        self.latencies = []
        
        logger.info(f"🔄 Processing {num_chunks} chunks...")
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size
            chunk = audio_np[start_idx:end_idx]
            
            start_time = time.perf_counter()
            reconstructed, encode_time, decode_time = self.process_chunk(chunk)
            total_time = (time.perf_counter() - start_time) * 1000
            
            reconstructed_chunks.append(reconstructed)
            self.latencies.append({
                'encode': encode_time,
                'decode': decode_time,
                'total': total_time
            })
        
        # Handle remaining samples
        remaining = len(audio_np) % self.chunk_size
        if remaining > 0:
            chunk = np.pad(audio_np[-remaining:], (0, self.chunk_size - remaining))
            reconstructed, _, _ = self.process_chunk(chunk)
            reconstructed_chunks.append(reconstructed[:remaining])
        
        # Combine and save
        full_reconstructed = np.concatenate(reconstructed_chunks)
        
        # Calculate SNR
        min_len = min(len(audio_np), len(full_reconstructed))
        orig = audio_np[:min_len]
        recon = full_reconstructed[:min_len]
        
        signal_power = np.mean(orig ** 2)
        noise_power = np.mean((orig - recon) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # Save output
        output_tensor = torch.from_numpy(full_reconstructed).unsqueeze(0)
        torchaudio.save(output_file, output_tensor, self.sample_rate)
        
        logger.info(f"💾 Saved: {output_file}")
        
        self._print_results()
        logger.info(f"\n📊 Reconstruction SNR: {snr:.2f} dB")
        
        return snr
    
    def _print_results(self):
        """Print latency statistics"""
        if not self.latencies:
            logger.warning("No latency data collected!")
            return
        
        encode_times = [l['encode'] for l in self.latencies]
        decode_times = [l['decode'] for l in self.latencies]
        total_times = [l['total'] for l in self.latencies]
        
        logger.info("\n" + "="*60)
        logger.info("📊 LATENCY RESULTS")
        logger.info("="*60)
        logger.info(f"Chunks processed: {len(self.latencies)}")
        logger.info(f"Chunk duration: {self.chunk_duration*1000:.0f}ms")
        logger.info("-"*60)
        logger.info(f"ENCODE:  avg={np.mean(encode_times):.2f}ms, "
                   f"min={np.min(encode_times):.2f}ms, "
                   f"max={np.max(encode_times):.2f}ms")
        logger.info(f"DECODE:  avg={np.mean(decode_times):.2f}ms, "
                   f"min={np.min(decode_times):.2f}ms, "
                   f"max={np.max(decode_times):.2f}ms")
        logger.info(f"TOTAL:   avg={np.mean(total_times):.2f}ms, "
                   f"min={np.min(total_times):.2f}ms, "
                   f"max={np.max(total_times):.2f}ms")
        logger.info("-"*60)
        
        # Real-time factor
        rtf = np.mean(total_times) / (self.chunk_duration * 1000)
        logger.info(f"Real-time factor: {rtf:.4f}x")
        
        if rtf < 1.0:
            logger.info(f"✅ REAL-TIME CAPABLE! ({(1/rtf):.1f}x faster than real-time)")
        else:
            logger.info(f"❌ NOT real-time capable (need {rtf:.1f}x speedup)")
        
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Real-time Telugu Codec Test")
    parser.add_argument("--codec_path", default="/workspace/models/codec/best_codec.pt",
                       help="Path to codec checkpoint")
    parser.add_argument("--mode", choices=["realtime", "file"], default="file",
                       help="Test mode: realtime (mic) or file")
    parser.add_argument("--input", type=str, help="Input audio file (for file mode)")
    parser.add_argument("--output", type=str, default="reconstructed.wav",
                       help="Output file (for file mode)")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Recording duration in seconds (for realtime mode)")
    parser.add_argument("--chunk_ms", type=int, default=500,
                       help="Chunk size in milliseconds")
    args = parser.parse_args()
    
    # Initialize tester
    tester = RealtimeCodecTest(
        codec_path=args.codec_path,
        chunk_duration=args.chunk_ms / 1000.0
    )
    
    if args.mode == "realtime":
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("❌ Real-time mode requires sounddevice: pip install sounddevice")
            return
        tester.run_realtime(duration=args.duration)
    else:
        # File mode
        if args.input:
            input_file = args.input
        else:
            # Find a sample file
            sample_dirs = [
                "/workspace/telugu_data/openslr",
                "/workspace/telugu_data/indictts/audio",
                "/workspace/telugu_data"
            ]
            input_file = None
            for d in sample_dirs:
                if Path(d).exists():
                    files = list(Path(d).glob("*.wav"))
                    if files:
                        input_file = str(files[0])
                        break
            
            if not input_file:
                logger.error("❌ No input file specified and no samples found!")
                return
        
        tester.run_file_test(input_file, args.output)


if __name__ == "__main__":
    main()
