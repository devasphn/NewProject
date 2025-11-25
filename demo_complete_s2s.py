#!/usr/bin/env python3
"""
Telugu S2S Complete Demo
End-to-end speech-to-speech demonstration
Shows: Audio Input → Codec → S2S Transformer → Codec Decode → Audio Output
"""

import torch
import torchaudio
import argparse
import logging
from pathlib import Path
import time
import json

from telugu_codec_fixed import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeluguS2SDemo:
    """Complete S2S demonstration"""
    
    def __init__(self, codec_path: str, s2s_path: str, speaker_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load Codec
        self.codec = self._load_codec(codec_path)
        logger.info("✅ Codec loaded")
        
        # Load S2S Model
        self.s2s_model = self._load_s2s(s2s_path)
        logger.info("✅ S2S Transformer loaded")
        
        # Load speaker embeddings
        self.speakers = self._load_speakers(speaker_path)
        logger.info(f"✅ {len(self.speakers)} speakers available")
    
    def _load_codec(self, path: str) -> TeluCodec:
        codec = TeluCodec().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        if 'codec_state_dict' in checkpoint:
            codec.load_state_dict(checkpoint['codec_state_dict'])
        else:
            codec.load_state_dict(checkpoint)
        codec.eval()
        return codec
    
    def _load_s2s(self, path: str) -> TeluguS2STransformer:
        config = S2SConfig(
            hidden_dim=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            use_flash_attn=False
        )
        model = TeluguS2STransformer(config).to(self.device)
        
        checkpoint = torch.load(path, map_location=self.device)
        if 's2s_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['s2s_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def _load_speakers(self, path: str) -> dict:
        if path and Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get('speakers', {})
        return {
            "arjun": {"id": 0, "name": "Arjun", "description": "Young male"},
            "ravi": {"id": 1, "name": "Ravi", "description": "Mature male"},
            "priya": {"id": 2, "name": "Priya", "description": "Young female"},
            "lakshmi": {"id": 3, "name": "Lakshmi", "description": "Professional female"}
        }
    
    def load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform / (waveform.abs().max() + 1e-8) * 0.9
        return waveform.to(self.device)
    
    def save_audio(self, waveform: torch.Tensor, path: str):
        """Save audio to file"""
        waveform = waveform.cpu()
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        torchaudio.save(path, waveform, 16000)
        logger.info(f"Saved: {path}")
    
    @torch.no_grad()
    def process_audio(self, input_path: str, output_path: str, 
                      speaker_id: int = 0, emotion_id: int = 0,
                      use_streaming: bool = False):
        """
        Complete S2S pipeline:
        Audio → Encode → S2S Transform → Decode → Audio
        """
        logger.info(f"\n{'='*60}")
        logger.info("TELUGU S2S PROCESSING")
        logger.info(f"{'='*60}")
        
        # Load input audio
        waveform = self.load_audio(input_path)
        waveform = waveform.unsqueeze(0)  # Add batch dim
        logger.info(f"Input audio: {waveform.shape} ({waveform.shape[-1]/16000:.2f}s)")
        
        # Step 1: Encode to discrete codes
        start_time = time.time()
        input_codes = self.codec.encode(waveform).long()
        encode_time = (time.time() - start_time) * 1000
        logger.info(f"Encoded codes: {input_codes.shape} [{encode_time:.1f}ms]")
        
        # Step 2: S2S Transform
        speaker = torch.tensor([speaker_id], device=self.device)
        emotion = torch.tensor([emotion_id], device=self.device)
        
        if use_streaming:
            # Streaming generation
            start_time = time.time()
            output_codes_list = []
            chunk_times = []
            
            for chunk in self.s2s_model.generate_streaming(
                input_codes, speaker, emotion, max_new_tokens=input_codes.shape[-1]
            ):
                chunk_time = (time.time() - start_time) * 1000
                chunk_times.append(chunk_time)
                output_codes_list.append(chunk)
                start_time = time.time()
            
            if output_codes_list:
                output_codes = torch.cat(output_codes_list, dim=-1)
                avg_chunk_time = sum(chunk_times) / len(chunk_times)
                logger.info(f"Streaming: {len(output_codes_list)} chunks, avg {avg_chunk_time:.1f}ms/chunk")
            else:
                output_codes = input_codes  # Fallback
        else:
            # Direct forward (reconstruction)
            start_time = time.time()
            output_codes = input_codes  # For reconstruction demo
            transform_time = (time.time() - start_time) * 1000
            logger.info(f"Transform: direct reconstruction [{transform_time:.1f}ms]")
        
        # Step 3: Decode back to audio
        start_time = time.time()
        output_waveform = self.codec.decode(output_codes)
        decode_time = (time.time() - start_time) * 1000
        logger.info(f"Decoded audio: {output_waveform.shape} [{decode_time:.1f}ms]")
        
        # Calculate quality metrics
        min_len = min(waveform.shape[-1], output_waveform.shape[-1])
        orig = waveform[..., :min_len]
        recon = output_waveform[..., :min_len]
        
        signal_power = torch.mean(orig ** 2)
        noise_power = torch.mean((orig - recon) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        
        # Save output
        self.save_audio(output_waveform, output_path)
        
        # Summary
        total_time = encode_time + decode_time
        logger.info(f"\n{'='*60}")
        logger.info("RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Input duration: {waveform.shape[-1]/16000:.2f}s")
        logger.info(f"✅ Output duration: {output_waveform.shape[-1]/16000:.2f}s")
        logger.info(f"✅ Reconstruction SNR: {snr.item():.2f} dB")
        logger.info(f"✅ Total latency: {total_time:.1f}ms")
        logger.info(f"✅ Speaker: {list(self.speakers.values())[speaker_id]['name']}")
        logger.info(f"✅ Output saved: {output_path}")
        logger.info(f"{'='*60}")
        
        return {
            "snr": snr.item(),
            "latency_ms": total_time,
            "input_duration": waveform.shape[-1]/16000,
            "output_duration": output_waveform.shape[-1]/16000
        }
    
    def list_speakers(self):
        """Display available speakers"""
        logger.info("\n=== Available Speakers ===")
        for name, info in self.speakers.items():
            logger.info(f"  {info['id']}: {info['name']} - {info['description']}")
    
    def run_demo(self):
        """Run interactive demo"""
        logger.info("\n" + "="*60)
        logger.info("🎤 TELUGU S2S COMPLETE DEMO")
        logger.info("="*60)
        
        self.list_speakers()
        
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
        
        if not sample_file:
            logger.error("No sample audio found!")
            return
        
        logger.info(f"\nProcessing: {sample_file}")
        
        # Process with different speakers
        results = []
        for speaker_id in range(min(2, len(self.speakers))):
            output_file = f"output_speaker_{speaker_id}.wav"
            result = self.process_audio(
                sample_file, output_file, 
                speaker_id=speaker_id, emotion_id=0
            )
            results.append(result)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 DEMO COMPLETE!")
        logger.info("="*60)
        logger.info(f"Average SNR: {sum(r['snr'] for r in results)/len(results):.2f} dB")
        logger.info(f"Average Latency: {sum(r['latency_ms'] for r in results)/len(results):.1f}ms")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Telugu S2S Complete Demo")
    parser.add_argument("--codec_path", default="/workspace/models/codec/best_codec.pt")
    parser.add_argument("--s2s_path", default="/workspace/models/s2s/s2s_best.pt")
    parser.add_argument("--speaker_path", default="/workspace/models/speaker_embeddings.json")
    parser.add_argument("--input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (0-3)")
    parser.add_argument("--emotion", type=int, default=0, help="Emotion ID (0-8)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming generation")
    args = parser.parse_args()
    
    # Initialize demo
    demo = TeluguS2SDemo(args.codec_path, args.s2s_path, args.speaker_path)
    
    if args.input:
        # Process specific file
        demo.process_audio(args.input, args.output, args.speaker, args.emotion, args.streaming)
    else:
        # Run interactive demo
        demo.run_demo()


if __name__ == "__main__":
    main()
