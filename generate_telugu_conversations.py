#!/usr/bin/env python3
"""
Generate Telugu Conversation Training Data
==========================================

Creates (question, answer) audio pairs for training S2S conversation model.

Methods:
1. Synthetic: Use predefined Q&A pairs + TTS
2. LLM-generated: Use Qwen to generate Q&A, then TTS
3. Real data: Process Kathbath/other datasets

Output format:
data/telugu_conversations/
├── pair_0001/
│   ├── question.wav
│   ├── answer.wav
│   ├── question_codes.pt  # Encoded with YOUR codec
│   └── answer_codes.pt
├── pair_0002/
│   └── ...
"""

import torch
import numpy as np
import os
import json
import asyncio
import argparse
from pathlib import Path
from tqdm import tqdm
import random

# Telugu conversation templates
TELUGU_QA_PAIRS = [
    # Greetings
    ("నమస్కారం", "నమస్కారం! మీకు ఎలా సహాయం చేయగలను?"),
    ("హలో", "హలో! చెప్పండి, మీకు ఏమి కావాలి?"),
    ("శుభోదయం", "శుభోదయం! మీ రోజు బాగుండాలని కోరుకుంటున్నాను."),
    ("శుభ సాయంత్రం", "శుభ సాయంత్రం! ఎలా ఉన్నారు?"),
    
    # How are you
    ("మీరు ఎలా ఉన్నారు?", "నేను బాగున్నాను, ధన్యవాదాలు! మీరు ఎలా ఉన్నారు?"),
    ("ఎలా ఉన్నావు?", "బాగున్నాను! మీ గురించి చెప్పండి."),
    ("క్షేమంగా ఉన్నారా?", "చాలా క్షేమంగా ఉన్నాను, మీరు?"),
    
    # Name queries
    ("మీ పేరు ఏమిటి?", "నా పేరు తెలుగు అసిస్టెంట్. మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను."),
    ("నీ పేరు చెప్పు", "నేను తెలుగు వాయిస్ అసిస్టెంట్ని."),
    ("నువ్వు ఎవరు?", "నేను మీ తెలుగు AI అసిస్టెంట్ని."),
    
    # Weather
    ("వాతావరణం ఎలా ఉంది?", "ఈ రోజు వాతావరణం బాగుంది. మీ ప్రాంతంలో ఎలా ఉంది?"),
    ("ఈ రోజు వర్షం వస్తుందా?", "వాతావరణ సమాచారం ప్రకారం చెప్పగలను."),
    
    # Time
    ("సమయం ఎంత?", "ప్రస్తుత సమయం చెప్పగలను."),
    ("ఈ రోజు ఏ తేదీ?", "ఈ రోజు తేదీ చెప్పగలను."),
    
    # Help
    ("నాకు సహాయం కావాలి", "తప్పకుండా! మీకు ఏ విధంగా సహాయం చేయగలను?"),
    ("నీవు ఏమి చేయగలవు?", "నేను మీతో తెలుగులో మాట్లాడగలను, ప్రశ్నలకు సమాధానాలు ఇవ్వగలను."),
    
    # Thank you
    ("ధన్యవాదాలు", "మీకు స్వాగతం! మరేదైనా సహాయం కావాలా?"),
    ("థాంక్స్", "ఏ మాత్రం! మీకు సహాయం చేయడం సంతోషంగా ఉంది."),
    
    # Goodbye
    ("వెళ్ళొస్తాను", "సరే, మళ్ళీ కలుద్దాం! జాగ్రత్త!"),
    ("బై", "బై! మంచి రోజు గడపండి!"),
    
    # General questions
    ("తెలుగు భాష గురించి చెప్పు", "తెలుగు ఒక అందమైన ద్రావిడ భాష. ఆంధ్ర ప్రదేశ్ మరియు తెలంగాణలో మాట్లాడతారు."),
    ("హైదరాబాద్ గురించి చెప్పు", "హైదరాబాద్ తెలంగాణ రాజధాని. చార్మినార్ మరియు బిర్యానీకి ప్రసిద్ధి."),
    ("భారతదేశం గురించి చెప్పు", "భారతదేశం ఒక గొప్ప దేశం. వివిధ భాషలు, సంస్కృతులు ఉన్నాయి."),
    
    # More conversational
    ("ఏం చేస్తున్నావు?", "మీతో మాట్లాడటానికి సిద్ధంగా ఉన్నాను!"),
    ("బోర్ కొడుతోంది", "అయ్యో! ఏదైనా కథ చెప్పమంటారా?"),
    ("ఒక జోక్ చెప్పు", "ఒక మనిషి డాక్టర్ దగ్గరికి వెళ్ళాడు. డాక్టర్ అన్నాడు: మీరు బాగానే ఉన్నారు!"),
    
    # Numbers and counting
    ("ఒకటి నుండి పది వరకు చెప్పు", "ఒకటి, రెండు, మూడు, నాలుగు, ఐదు, ఆరు, ఏడు, ఎనిమిది, తొమ్మిది, పది."),
    ("ఒకటి కలపండి ఒకటి", "ఒకటి కలపండి ఒకటి సమానం రెండు."),
    
    # Food
    ("మీకు ఇష్టమైన ఆహారం ఏమిటి?", "నేను AI ని, కానీ తెలుగు వంటకాలు చాలా రుచిగా ఉంటాయి!"),
    ("బిర్యానీ ఎలా చేయాలి?", "బిర్యానీ చేయడానికి బియ్యం, మసాలాలు, మాంసం అవసరం."),
]

# Extended pairs can be generated by LLM
TOPICS_FOR_GENERATION = [
    "చరిత్ర", "విజ్ఞానం", "సాంకేతికత", "క్రీడలు", "సంగీతం",
    "సినిమాలు", "ఆరోగ్యం", "విద్య", "ప్రయాణం", "ఆహారం",
    "కుటుంబం", "పండుగలు", "రాజకీయాలు", "వ్యాపారం", "పర్యావరణం"
]


class TeluguConversationGenerator:
    """Generate Telugu conversation pairs for S2S training"""
    
    def __init__(self, codec_path: str, output_dir: str, device: str = "cuda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load YOUR codec
        print("📥 Loading Telugu Codec...")
        from telugu_codec_fixed import TeluCodec
        self.codec = TeluCodec().to(device)
        checkpoint = torch.load(codec_path, map_location=device)
        if 'codec_state_dict' in checkpoint:
            self.codec.load_state_dict(checkpoint['codec_state_dict'])
        else:
            self.codec.load_state_dict(checkpoint)
        self.codec.eval()
        print("✅ Codec loaded!")
        
        # TTS will be initialized when needed
        self.tts = None
        
    async def text_to_audio(self, text: str) -> np.ndarray:
        """Convert Telugu text to audio using Edge TTS"""
        import edge_tts
        from pydub import AudioSegment
        import io
        
        communicate = edge_tts.Communicate(text, "te-IN-ShrutiNeural")
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        if not audio_data:
            return None
            
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        return samples / 32768.0
    
    @torch.no_grad()
    def encode_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Encode audio using YOUR codec"""
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        codes = self.codec.encode(audio_tensor)
        return codes.cpu()
    
    async def generate_pair(self, question: str, answer: str, pair_id: int) -> bool:
        """Generate a single conversation pair"""
        pair_dir = self.output_dir / f"pair_{pair_id:05d}"
        pair_dir.mkdir(exist_ok=True)
        
        try:
            # Generate question audio
            q_audio = await self.text_to_audio(question)
            if q_audio is None or len(q_audio) < 1600:  # Min 0.1s
                return False
            
            # Generate answer audio
            a_audio = await self.text_to_audio(answer)
            if a_audio is None or len(a_audio) < 1600:
                return False
            
            # Save audio files
            import soundfile as sf
            sf.write(pair_dir / "question.wav", q_audio, 16000)
            sf.write(pair_dir / "answer.wav", a_audio, 16000)
            
            # Encode with YOUR codec
            q_codes = self.encode_audio(q_audio)
            a_codes = self.encode_audio(a_audio)
            
            # Save codes
            torch.save(q_codes, pair_dir / "question_codes.pt")
            torch.save(a_codes, pair_dir / "answer_codes.pt")
            
            # Save metadata
            metadata = {
                "question_text": question,
                "answer_text": answer,
                "question_audio_length": len(q_audio),
                "answer_audio_length": len(a_audio),
                "question_codes_shape": list(q_codes.shape),
                "answer_codes_shape": list(a_codes.shape)
            }
            with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error generating pair {pair_id}: {e}")
            return False
    
    async def generate_from_templates(self, num_pairs: int):
        """Generate pairs from predefined templates"""
        print(f"\n📝 Generating {num_pairs} pairs from templates...")
        
        # Expand templates with variations
        expanded_pairs = []
        for q, a in TELUGU_QA_PAIRS:
            expanded_pairs.append((q, a))
            # Add variations
            if "?" in q:
                expanded_pairs.append((q.replace("?", ""), a))
        
        # Repeat to get desired count
        while len(expanded_pairs) < num_pairs:
            expanded_pairs.extend(TELUGU_QA_PAIRS)
        
        expanded_pairs = expanded_pairs[:num_pairs]
        random.shuffle(expanded_pairs)
        
        success_count = 0
        for i, (q, a) in enumerate(tqdm(expanded_pairs, desc="Generating")):
            if await self.generate_pair(q, a, i):
                success_count += 1
            await asyncio.sleep(0.5)  # Rate limit TTS
        
        print(f"\n✅ Generated {success_count}/{num_pairs} pairs successfully!")
        return success_count
    
    async def generate_with_llm(self, num_pairs: int, llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Generate more diverse pairs using LLM"""
        print(f"\n🤖 Generating {num_pairs} pairs using LLM...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        system_prompt = """Generate a Telugu question and answer pair. 
The question should be a natural conversational question in Telugu.
The answer should be a helpful, friendly response in Telugu.
Keep both short (1-2 sentences).

Format:
Q: [Telugu question]
A: [Telugu answer]"""
        
        success_count = 0
        start_id = len(list(self.output_dir.glob("pair_*")))
        
        for i in tqdm(range(num_pairs), desc="LLM Generating"):
            topic = random.choice(TOPICS_FOR_GENERATION)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a Telugu Q&A about: {topic}"}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, do_sample=True)
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse Q and A
            try:
                if "Q:" in response and "A:" in response:
                    parts = response.split("A:")
                    q = parts[0].replace("Q:", "").strip()
                    a = parts[1].strip().split("\n")[0]
                    
                    if len(q) > 5 and len(a) > 5:
                        if await self.generate_pair(q, a, start_id + i):
                            success_count += 1
            except:
                pass
            
            await asyncio.sleep(0.5)
        
        print(f"\n✅ LLM generated {success_count}/{num_pairs} pairs!")
        return success_count


async def main():
    parser = argparse.ArgumentParser(description="Generate Telugu conversation training data")
    parser.add_argument("--codec", default="best_codec.pt", help="Path to your codec")
    parser.add_argument("--output", default="data/telugu_conversations", help="Output directory")
    parser.add_argument("--num_template", type=int, default=100, help="Number of template pairs")
    parser.add_argument("--num_llm", type=int, default=0, help="Number of LLM-generated pairs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    generator = TeluguConversationGenerator(args.codec, args.output, args.device)
    
    total = 0
    
    if args.num_template > 0:
        total += await generator.generate_from_templates(args.num_template)
    
    if args.num_llm > 0:
        total += await generator.generate_with_llm(args.num_llm)
    
    print(f"\n🎉 Total pairs generated: {total}")
    print(f"📁 Output directory: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
