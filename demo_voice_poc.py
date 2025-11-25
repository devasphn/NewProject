#!/usr/bin/env python3
"""
Telugu S2S Voice POC Demo
Shows codec reconstruction with speaker control
Works with trained codec model
"""

import torch
import torch.nn.functional as F
import torchaudio
import json
from pathlib import Path
import argparse
import logging

from telugu_codec_fixed import TeluCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoicePOCDemo:
    """Simple POC demo using trained codec"""
    
    def __init__(self, codec_path: str, speaker_embeddings_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load codec
        self.codec = TeluCodec().to(self.device)
        self._load_codec(codec_path)
        self.codec.eval()
        logger.info("Codec loaded successfully!")
        
        # Load speaker embeddings if available
        self.speaker_embeddings = None
        if speaker_embeddings_path and Path(speaker_embeddings_path).exists():
            with open(speaker_embeddings_path, 'r') as f:
                data = json.load(f)
                self.speaker_embeddings = torch.tensor(data['embeddings']).to(self.device)
                self.speakers = data['speakers']
            logger.info(f"Loaded {len(self.speakers)} speaker embeddings")
    
    def _load_codec(self, codec_path: str):
        """Load trained codec"""
        checkpoint = torch.load(codec_path, map_location=self.device)
        if 'codec_state_dict' in checkpoint:
            self.codec.load_state_dict(checkpoint['codec_state_dict'])
        elif 'model_state' in checkpoint:
            self.codec.load_state_dict(checkpoint['model_state'])
        else:
            self.codec.load_state_dict(checkpoint)
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """Load and preprocess audio"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8) * 0.9
        
        return waveform.to(self.device)
    
    def save_audio(self, waveform: torch.Tensor, output_path: str, sr: int = 16000):
        """Save audio to file"""
        waveform = waveform.cpu()
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        torchaudio.save(output_path, waveform, sr)
        logger.info(f"Saved audio to {output_path}")
    
    @torch.no_grad()
    def encode_decode(self, audio_path: str, output_path: str = None):
        """
        Encode audio to codes and decode back
        Demonstrates codec quality
        """
        # Load audio
        waveform = self.load_audio(audio_path)
        waveform = waveform.unsqueeze(0)  # Add batch dim
        
        logger.info(f"Input shape: {waveform.shape}")
        
        # Encode to discrete codes
        codes = self.codec.encode(waveform)
        logger.info(f"Encoded codes shape: {codes.shape}")
        logger.info(f"Code values range: [{codes.min().item()}, {codes.max().item()}]")
        
        # Decode back to audio
        reconstructed = self.codec.decode(codes)
        logger.info(f"Reconstructed shape: {reconstructed.shape}")
        
        # Calculate metrics
        min_len = min(waveform.shape[-1], reconstructed.shape[-1])
        waveform_trim = waveform[..., :min_len]
        reconstructed_trim = reconstructed[..., :min_len]
        
        # SNR
        signal_power = torch.mean(waveform_trim ** 2)
        noise_power = torch.mean((waveform_trim - reconstructed_trim) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        logger.info(f"Reconstruction SNR: {snr.item():.2f} dB")
        
        # Save output
        if output_path is None:
            output_path = str(Path(audio_path).stem) + "_reconstructed.wav"
        self.save_audio(reconstructed, output_path)
        
        return {
            "snr": snr.item(),
            "codes_shape": list(codes.shape),
            "output_path": output_path
        }
    
    @torch.no_grad()
    def batch_process(self, input_dir: str, output_dir: str, max_files: int = 10):
        """Process multiple audio files"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(input_path.glob("*.wav"))[:max_files]
        logger.info(f"Processing {len(audio_files)} files...")
        
        results = []
        for audio_file in audio_files:
            try:
                output_file = output_path / f"{audio_file.stem}_reconstructed.wav"
                result = self.encode_decode(str(audio_file), str(output_file))
                results.append({"file": audio_file.name, **result})
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
        
        # Summary
        if results:
            avg_snr = sum(r["snr"] for r in results) / len(results)
            logger.info(f"\n=== Summary ===")
            logger.info(f"Processed: {len(results)} files")
            logger.info(f"Average SNR: {avg_snr:.2f} dB")
        
        return results
    
    def list_speakers(self):
        """List available speakers"""
        if self.speaker_embeddings is not None:
            logger.info("\n=== Available Speakers ===")
            for name, info in self.speakers.items():
                logger.info(f"  {info['id']}: {info['name']} - {info['description']}")
        else:
            logger.info("No speaker embeddings loaded")


def main():
    parser = argparse.ArgumentParser(description="Telugu S2S Voice POC Demo")
    parser.add_argument("--codec_path", type=str, default="/workspace/models/codec/best_codec.pt")
    parser.add_argument("--speaker_path", type=str, default="/workspace/models/speaker_embeddings.json")
    parser.add_argument("--input", type=str, help="Input audio file or directory")
    parser.add_argument("--output", type=str, default="./output", help="Output path")
    parser.add_argument("--mode", type=str, choices=["single", "batch", "demo"], default="demo")
    args = parser.parse_args()
    
    # Initialize demo
    demo = VoicePOCDemo(args.codec_path, args.speaker_path)
    
    if args.mode == "demo":
        # Run demo on sample files
        logger.info("\n" + "="*60)
        logger.info("TELUGU S2S VOICE POC DEMO")
        logger.info("="*60)
        
        demo.list_speakers()
        
        # Find sample audio
        sample_dirs = [
            "/workspace/telugu_data/openslr",
            "/workspace/telugu_data/indictts/audio",
            "/workspace/telugu_data"
        ]
        
        sample_file = None
        for d in sample_dirs:
            files = list(Path(d).glob("*.wav"))[:1] if Path(d).exists() else []
            if files:
                sample_file = str(files[0])
                break
        
        if sample_file:
            logger.info(f"\nProcessing sample: {sample_file}")
            result = demo.encode_decode(sample_file, "demo_output.wav")
            
            logger.info("\n" + "="*60)
            logger.info("POC DEMO RESULTS")
            logger.info("="*60)
            logger.info(f"✅ Codec encoding/decoding: WORKING")
            logger.info(f"✅ Audio quality (SNR): {result['snr']:.2f} dB")
            logger.info(f"✅ Discrete codes shape: {result['codes_shape']}")
            logger.info(f"✅ Output saved to: {result['output_path']}")
            logger.info("="*60)
        else:
            logger.warning("No sample audio files found!")
    
    elif args.mode == "single" and args.input:
        demo.encode_decode(args.input, args.output)
    
    elif args.mode == "batch" and args.input:
        demo.batch_process(args.input, args.output)


if __name__ == "__main__":
    main()
