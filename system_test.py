#!/usr/bin/env python3
"""
Comprehensive system test for Telugu S2S
Verifies all components are working correctly
"""

import torch
import time
import numpy as np
import argparse
import json
import asyncio
import websockets
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Import all components
from telugu_codec_fixed import TeluCodec
from s2s_transformer import TeluguS2STransformer, S2SConfig, EMOTION_IDS, SPEAKER_IDS
from speaker_embeddings import SpeakerEmbeddingSystem
from context_manager import ConversationContextManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system testing"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        logger.info(f"System Tester initialized on {self.device}")
    
    def test_codec(self) -> bool:
        """Test codec loading and performance"""
        try:
            logger.info("Testing Codec...")
            
            # Load codec
            codec = TeluCodec().to(self.device)
            codec_path = self.model_dir / "best_codec.pt"
            
            if not codec_path.exists():
                logger.error(f"Codec not found at {codec_path}")
                self.results["codec"] = {"status": "FAIL", "error": "File not found"}
                return False
            
            checkpoint = torch.load(codec_path, map_location=self.device)
            codec.load_state_dict(checkpoint["model_state"])
            codec.eval()
            
            # Test encoding/decoding
            test_audio = torch.randn(1, 1, 16000).to(self.device)  # 1 second
            
            # Measure encode latency
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                codes = codec.encode(test_audio)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            encode_time = (time.time() - start) * 1000
            
            # Measure decode latency
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                reconstructed = codec.decode(codes)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            decode_time = (time.time() - start) * 1000
            
            # Calculate metrics
            bitrate = codec.calculate_bitrate()
            compression_ratio = 16000 * 16 / (bitrate * 1000)  # 16-bit audio
            
            # Reconstruction quality
            mse = torch.mean((test_audio - reconstructed) ** 2).item()
            snr = -10 * np.log10(mse + 1e-8)
            
            self.results["codec"] = {
                "status": "PASS",
                "encode_latency_ms": encode_time,
                "decode_latency_ms": decode_time,
                "bitrate_kbps": bitrate,
                "compression_ratio": compression_ratio,
                "snr_db": snr
            }
            
            logger.info(f"✓ Codec: Encode={encode_time:.1f}ms, Decode={decode_time:.1f}ms, Bitrate={bitrate:.1f}kbps")
            return encode_time < 15 and decode_time < 15
            
        except Exception as e:
            logger.error(f"Codec test failed: {e}")
            self.results["codec"] = {"status": "FAIL", "error": str(e)}
            return False
    
    def test_speakers(self) -> bool:
        """Test speaker embeddings"""
        try:
            logger.info("Testing Speaker System...")
            
            # Load speaker system
            speaker_system = SpeakerEmbeddingSystem().to(self.device)
            speaker_path = self.model_dir / "speaker_embeddings.json"
            
            if speaker_path.exists():
                speaker_system.load_embeddings(str(speaker_path))
            
            # Test all 4 speakers
            results = {}
            for speaker_id in range(4):
                speaker_tensor = torch.tensor([speaker_id], device=self.device)
                
                # Generate embedding
                embedding = speaker_system(speaker_tensor)
                
                # Verify embedding properties
                assert embedding.shape == (1, 256), f"Wrong shape: {embedding.shape}"
                assert torch.isfinite(embedding).all(), "Non-finite values in embedding"
                
                # Get speaker info
                info = speaker_system.get_speaker_info(speaker_id)
                results[speaker_id] = {
                    "name": info["name"],
                    "description": info["description"],
                    "embedding_norm": torch.norm(embedding).item()
                }
                
                logger.info(f"  Speaker {speaker_id} ({info['name']}): ✓")
            
            # Test speaker interpolation
            interpolated = speaker_system.interpolate_speakers(0, 2, alpha=0.5)
            assert interpolated.shape == (256,), "Interpolation failed"
            
            self.results["speakers"] = {
                "status": "PASS",
                "num_speakers": 4,
                "speakers": results,
                "interpolation": "Working"
            }
            
            logger.info("✓ All 4 speakers loaded and working")
            return True
            
        except Exception as e:
            logger.error(f"Speaker test failed: {e}")
            self.results["speakers"] = {"status": "FAIL", "error": str(e)}
            return False
    
    def test_s2s_model(self) -> bool:
        """Test S2S model and streaming"""
        try:
            logger.info("Testing S2S Model...")
            
            # Load models
            codec = TeluCodec().to(self.device)
            codec_checkpoint = torch.load(
                self.model_dir / "best_codec.pt",
                map_location=self.device
            )
            codec.load_state_dict(codec_checkpoint["model_state"])
            codec.eval()
            
            config = S2SConfig(use_flash_attn=torch.cuda.is_available())
            s2s = TeluguS2STransformer(config).to(self.device)
            s2s_path = self.model_dir / "s2s_best.pt"
            
            if not s2s_path.exists():
                logger.error(f"S2S model not found at {s2s_path}")
                self.results["s2s"] = {"status": "FAIL", "error": "File not found"}
                return False
            
            s2s_checkpoint = torch.load(s2s_path, map_location=self.device)
            s2s.load_state_dict(s2s_checkpoint["model_state"])
            s2s.eval()
            
            # Test inference
            test_audio = torch.randn(1, 1, 16000).to(self.device)
            
            with torch.no_grad():
                # Encode
                codes = codec.encode(test_audio)
                
                # Test each emotion
                latencies = {}
                for emotion_name, emotion_id in EMOTION_IDS.items():
                    speaker_tensor = torch.tensor([0], device=self.device)
                    emotion_tensor = torch.tensor([emotion_id], device=self.device)
                    
                    # Measure streaming latency (first chunk)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start = time.time()
                    
                    # Get first chunk
                    for chunk in s2s.generate_streaming(
                        codes, speaker_tensor, emotion_tensor, max_new_tokens=10
                    ):
                        first_chunk = chunk
                        break
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    latency = (time.time() - start) * 1000
                    latencies[emotion_name] = latency
                    
                    # Decode first chunk
                    audio_chunk = codec.decode(first_chunk)
                    assert audio_chunk.shape[2] > 0, "Empty audio chunk"
            
            # Calculate average latency
            avg_latency = np.mean(list(latencies.values()))
            
            self.results["s2s"] = {
                "status": "PASS" if avg_latency < 150 else "WARN",
                "average_latency_ms": avg_latency,
                "emotion_latencies": latencies,
                "streaming": "Working"
            }
            
            logger.info(f"✓ S2S Model: Avg latency={avg_latency:.1f}ms")
            for emotion, latency in latencies.items():
                logger.info(f"  {emotion}: {latency:.1f}ms")
            
            return avg_latency < 150
            
        except Exception as e:
            logger.error(f"S2S test failed: {e}")
            self.results["s2s"] = {"status": "FAIL", "error": str(e)}
            return False
    
    def test_context_manager(self) -> bool:
        """Test context management"""
        try:
            logger.info("Testing Context Manager...")
            
            manager = ConversationContextManager(max_turns=10)
            session_id = "test_session"
            
            # Add 15 turns to test sliding window
            for i in range(15):
                user_codes = torch.randn(1, 10, 768)
                bot_codes = torch.randn(1, 10, 768)
                
                emotion = ["neutral", "happy", "excited"][i % 3]
                
                turn = manager.add_turn(
                    session_id=session_id,
                    user_codes=user_codes,
                    bot_codes=bot_codes,
                    emotion=emotion,
                    speaker="female_young"
                )
            
            # Get context
            current_input = torch.randn(768)
            context = manager.get_context(session_id, current_input)
            
            # Verify context
            assert context["has_context"] == True
            assert context["conversation_length"] == 10  # Max 10 turns maintained
            assert len(context["recent_turns"]) <= 10
            
            # Test suggestion system
            suggestion = manager.suggest_response_style(session_id, 0.5)
            assert "suggested_emotion" in suggestion
            
            # Test save/load
            manager.save_session(session_id)
            assert (Path("context_data") / f"session_{session_id}.pkl").exists()
            
            self.results["context"] = {
                "status": "PASS",
                "max_turns": 10,
                "sliding_window": "Working",
                "context_retrieval": "Working",
                "save_load": "Working"
            }
            
            logger.info("✓ Context Manager: 10-turn memory working")
            return True
            
        except Exception as e:
            logger.error(f"Context test failed: {e}")
            self.results["context"] = {"status": "FAIL", "error": str(e)}
            return False
    
    async def test_server(self, port: int = 8000) -> bool:
        """Test WebSocket server"""
        try:
            logger.info(f"Testing Server on port {port}...")
            
            # Try to connect
            uri = f"ws://localhost:{port}/ws"
            
            try:
                async with websockets.connect(uri, timeout=2) as websocket:
                    # Send init message
                    init_msg = {
                        "session_id": "test_client",
                        "mode": "stream",
                        "interruption": True
                    }
                    await websocket.send(json.dumps(init_msg))
                    
                    # Send test audio
                    test_audio = np.random.randn(1600).astype(np.float32)
                    audio_msg = {
                        "type": "audio",
                        "audio": base64.b64encode(test_audio.tobytes()).decode()
                    }
                    await websocket.send(json.dumps(audio_msg))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response_data = json.loads(response)
                    
                    self.results["server"] = {
                        "status": "PASS",
                        "port": port,
                        "websocket": "Connected",
                        "streaming": "Working"
                    }
                    
                    logger.info(f"✓ Server running on port {port}")
                    return True
                    
            except (ConnectionRefusedError, OSError):
                # Try alternate ports
                for alt_port in [8080, 8010]:
                    if alt_port != port:
                        logger.info(f"  Trying port {alt_port}...")
                        return await self.test_server(alt_port)
                
                self.results["server"] = {
                    "status": "WARN",
                    "error": "Server not running",
                    "suggestion": "Start server with: python streaming_server_advanced.py"
                }
                logger.warning("⚠ Server not running (start it separately)")
                return True  # Not a critical failure
                
        except Exception as e:
            logger.error(f"Server test failed: {e}")
            self.results["server"] = {"status": "FAIL", "error": str(e)}
            return False
    
    def test_latency_benchmark(self) -> bool:
        """Comprehensive latency test"""
        try:
            logger.info("Running Latency Benchmark...")
            
            # Load all models
            codec = TeluCodec().to(self.device)
            codec.load_state_dict(
                torch.load(self.model_dir / "best_codec.pt", map_location=self.device)["model_state"]
            )
            codec.eval()
            
            config = S2SConfig(use_flash_attn=torch.cuda.is_available())
            s2s = TeluguS2STransformer(config).to(self.device)
            s2s.load_state_dict(
                torch.load(self.model_dir / "s2s_best.pt", map_location=self.device)["model_state"]
            )
            s2s.eval()
            
            # Run multiple tests
            latencies = []
            for _ in range(10):
                audio = torch.randn(1, 1, 1600).to(self.device)  # 100ms chunk
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                
                with torch.no_grad():
                    # Full pipeline
                    codes = codec.encode(audio)
                    
                    # Generate first chunk
                    for chunk in s2s.generate_streaming(
                        codes,
                        torch.tensor([0], device=self.device),
                        torch.tensor([0], device=self.device),
                        max_new_tokens=10
                    ):
                        response_codes = chunk
                        break
                    
                    response_audio = codec.decode(response_codes)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            
            # Calculate statistics
            mean_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            std_latency = np.std(latencies)
            
            self.results["latency"] = {
                "status": "PASS" if mean_latency < 150 else "FAIL",
                "mean_ms": mean_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
                "std_ms": std_latency,
                "target_met": mean_latency < 150
            }
            
            logger.info(f"✓ Latency: Mean={mean_latency:.1f}ms, Min={min_latency:.1f}ms, Max={max_latency:.1f}ms")
            
            if mean_latency < 150:
                logger.info("✅ TARGET LATENCY ACHIEVED (<150ms)")
            else:
                logger.warning(f"⚠ Latency above target: {mean_latency:.1f}ms > 150ms")
            
            return mean_latency < 150
            
        except Exception as e:
            logger.error(f"Latency test failed: {e}")
            self.results["latency"] = {"status": "FAIL", "error": str(e)}
            return False
    
    def run_all_tests(self) -> bool:
        """Run all system tests"""
        logger.info("="*60)
        logger.info("Starting Comprehensive System Test")
        logger.info("="*60)
        
        all_passed = True
        
        # Test each component
        tests = [
            ("Codec", self.test_codec),
            ("Speakers", self.test_speakers),
            ("S2S Model", self.test_s2s_model),
            ("Context Manager", self.test_context_manager),
            ("Latency", self.test_latency_benchmark)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*30}")
            logger.info(f"Testing: {test_name}")
            logger.info('='*30)
            
            passed = test_func()
            all_passed = all_passed and passed
            
            if passed:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        
        # Test server (async)
        logger.info(f"\n{'='*30}")
        logger.info("Testing: Server")
        logger.info('='*30)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_passed = loop.run_until_complete(self.test_server())
        loop.close()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        for component, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            emoji = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
            logger.info(f"{emoji} {component.upper()}: {status}")
            
            if component == "latency" and "mean_ms" in result:
                logger.info(f"   Average latency: {result['mean_ms']:.1f}ms")
        
        logger.info("="*60)
        
        if all_passed:
            logger.info("🎊 ALL TESTS PASSED!")
            logger.info("System is ready for production!")
        else:
            logger.warning("⚠ Some tests failed. Check logs above.")
        
        # Save results
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="System Test for Telugu S2S")
    parser.add_argument("--model_dir", type=str, default="/workspace/models",
                        help="Directory containing trained models")
    parser.add_argument("--full", action="store_true",
                        help="Run all tests")
    args = parser.parse_args()
    
    # Initialize tester
    tester = SystemTester(args.model_dir)
    
    # Run tests
    if args.full:
        success = tester.run_all_tests()
    else:
        # Run quick test
        success = tester.test_codec() and tester.test_speakers()
    
    # Exit code
    exit(0 if success else 1)

if __name__ == "__main__":
    import base64  # Add this import for server test
    main()