# Phase 1: Model Training Plan
## Telugu Speech-to-Speech Fine-Tuning Strategy

---

## Executive Summary

**Approach**: Fine-tune Moshi (7B) using LoRA on Telugu datasets  
**Time**: 5-7 days total training  
**Cost**: ~$35 (L4 GPU training) + $8 storage  
**Data**: 150-200 hours Telugu speech (YouTube + optional professional recordings)

---

## 1. Why Fine-Tuning (Not Training from Scratch)

### 1.1 Comparison Table

| Metric | Fine-Tuning | Training from Scratch |
|--------|-------------|----------------------|
| **Data Required** | 100-200 hours | 10,000-100,000 hours |
| **Training Time** | 5-7 days | 6-12 months |
| **GPU Cost** | $35-50 | $50,000-100,000 |
| **Expertise Needed** | Moderate | Expert |
| **Success Risk** | Low | High |
| **Quality** | 85-90% of native | 90-95% (if successful) |

### 1.2 Rationale

**Why Fine-Tuning Works**:
1. Moshi already learned universal speech patterns (100K+ hours)
2. Phonetic features transfer across languages
3. Need only to adapt:
   - Telugu-specific phonemes
   - Accent characteristics
   - Cultural speech patterns

**LoRA (Low-Rank Adaptation)**:
- Adds small trainable matrices to frozen model
- Trains only 0.1-1% of parameters
- Preserves base model knowledge
- Fast convergence (hours not months)

---

## 2. Three-Stage Training Strategy

```
Stage 1: Telugu Speech Adaptation (150 hours)
    ↓ (2 days training)
Stage 2: Voice Cloning (12 hours, 4 speakers)
    ↓ (0.5 days training)
Stage 3: Emotional Expression (50 hours)
    ↓ (1.5 days training)
═══════════════════════════════════════════════
Total: 4 days training + 1-2 days data prep
```

---

## 3. Stage 1: Telugu Speech Adaptation

### 3.1 Objective
Adapt Moshi to recognize and generate Telugu phonetics with natural accent.

### 3.2 Dataset Requirements

**Target Size**: 150-200 hours  
**Content Type**: Conversational Telugu (not scripted/formal)  
**Quality**: SNR > 15dB, minimal background noise  
**Diversity**: Multiple speakers, ages, accents  

**Recommended Sources**:
1. **Telugu Podcasts** (conversational, natural)
2. **YouTube Interviews** (high-quality audio)
3. **News Broadcasts** (clear pronunciation)
4. **Educational Content** (slower, clearer speech)

### 3.3 Data Collection Pipeline

#### Step 1: YouTube Extraction
```bash
# Install yt-dlp
pip install yt-dlp

# Download audio from Telugu channels
yt-dlp \
  --format 'bestaudio[ext=m4a]' \
  --extract-audio \
  --audio-format wav \
  --audio-quality 0 \
  --output '%(title)s.%(ext)s' \
  <YOUTUBE_PLAYLIST_URL>
```

**Recommended YouTube Channels**:
- Telugu podcasts: "Telugu Podcast", "Talking Movies"
- News: "TV9 Telugu", "ABN Telugu"
- Interviews: "Unstoppable with NBK", "Alitho Saradaga"
- Educational: "Easy Telugu", "Telugu Vlogs"

#### Step 2: Audio Quality Filtering
```python
import librosa
import numpy as np

def calculate_snr(audio_file):
    """Calculate Signal-to-Noise Ratio"""
    y, sr = librosa.load(audio_file)
    
    # Estimate noise (first 0.5 seconds assumed silent)
    noise = y[:int(0.5 * sr)]
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Filter: Keep only SNR > 15dB
for audio_file in audio_files:
    snr = calculate_snr(audio_file)
    if snr > 15:
        keep_file(audio_file)
    else:
        discard_file(audio_file)
```

#### Step 3: Speaker Diarization
```bash
# Install pyannote.audio
pip install pyannote.audio

# Separate speakers in multi-speaker audio
python diarize.py --input video.wav --output segments/
```

```python
from pyannote.audio import Pipeline

# Load pre-trained diarization model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

# Process audio
diarization = pipeline("audio.wav")

# Extract segments per speaker
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
    # Extract segment and save
```

#### Step 4: Segmentation
```python
from pydub import AudioSegment

def segment_audio(input_file, segment_duration=15):
    """
    Split long audio into 10-30 second segments
    (optimal for Moshi training)
    """
    audio = AudioSegment.from_wav(input_file)
    duration_ms = len(audio)
    
    segments = []
    for start_ms in range(0, duration_ms, segment_duration * 1000):
        end_ms = min(start_ms + segment_duration * 1000, duration_ms)
        segment = audio[start_ms:end_ms]
        
        # Ensure minimum length (10s)
        if len(segment) >= 10000:
            segments.append(segment)
    
    return segments
```

#### Step 5: Transcription (Whisper)
```bash
# Install Whisper
pip install openai-whisper

# Transcribe Telugu audio
whisper audio.wav \
  --model large-v3 \
  --language te \
  --output_format json \
  --output_dir transcripts/
```

```python
import whisper

model = whisper.load_model("large-v3")

result = model.transcribe(
    "audio.wav",
    language="te",
    word_timestamps=True
)

# Output: { "text": "తెలుగు transcript...", "segments": [...] }
```

#### Step 6: Quality Validation
```python
def validate_sample(audio_file, transcript):
    """
    Validate audio-transcript pair quality
    """
    # 1. Check duration (10-30 seconds)
    duration = librosa.get_duration(filename=audio_file)
    if not (10 <= duration <= 30):
        return False
    
    # 2. Check SNR
    if calculate_snr(audio_file) < 15:
        return False
    
    # 3. Check transcript length
    if len(transcript) < 20:  # Too short
        return False
    
    # 4. Check for Telugu script
    if not contains_telugu(transcript):
        return False
    
    return True
```

### 3.4 Dataset Statistics

**Target**:
- Total Hours: 150 hours
- Number of Samples: ~27,000 (at 20s avg)
- Unique Speakers: 50-100+
- File Format: 24kHz WAV, mono, 16-bit

**Expected Collection Time**:
- YouTube download: 2-3 days (automated)
- Processing: 1 day (diarization, segmentation)
- Transcription: 1 day (Whisper)
- Manual QA: 0.5 days (sample 5-10%)

**Total**: 4-5 days data preparation

### 3.5 Training Configuration

```yaml
# LoRA Configuration
model: moshi-7b
adapter_type: LoRA
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

# Trainable Modules
target_modules:
  - mimi_encoder.transformer.layers[8:12]  # Last 4 encoder layers
  - temporal_transformer.attention         # All attention layers
  - mimi_decoder.transformer.layers[8:12]  # Last 4 decoder layers

# Training Hyperparameters
learning_rate: 1e-5
batch_size: 4  # Per L4 GPU (24GB VRAM)
gradient_accumulation_steps: 8  # Effective batch = 32
max_epochs: 5
warmup_steps: 500
optimizer: AdamW
weight_decay: 0.01
scheduler: cosine_with_restarts

# Mixed Precision
fp16: true  # Faster training, lower memory

# Data Loading
num_workers: 4
prefetch_factor: 2
pin_memory: true
```

### 3.6 Training Script (Pseudo-code)

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# Load base Moshi model
base_model = MoshiModel.from_pretrained("kyutai/moshi-7b")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["attention", "encoder", "decoder"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA adapters
model = get_peft_model(base_model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./moshi-telugu",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500
)

# Load Telugu dataset
dataset = TeluguSpeechDataset(
    audio_dir="data/telugu_audio/",
    transcript_dir="data/telugu_transcripts/"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val
)

trainer.train()

# Save LoRA adapters (only ~50MB)
model.save_pretrained("./moshi-telugu-lora")
```

### 3.7 Training Time & Cost

```
Dataset: 150 hours (27,000 samples)
Batch Size: 32 (effective)
Steps per Epoch: 27,000 / 32 = 844 steps
Total Epochs: 5
Total Steps: 4,220 steps

Time per Step (L4 GPU, FP16): ~4 seconds
Total Training Time: 4,220 × 4 = 16,880 seconds ≈ 4.7 hours per epoch

Total: 4.7 hours × 5 epochs = 23.5 hours (≈1 day)

Cost: 24 hours × $0.49/hour (L4 on-demand) = $11.76
```

### 3.8 Expected Outcomes

**After Stage 1**:
- Telugu phoneme recognition: 85-90% accuracy
- Accent: Natural (native-level prosody)
- Latency: No change (still 200ms on L4)
- Quality: Comparable to English/French output

---

## 4. Stage 2: Voice Cloning (4 Speakers)

### 4.1 Objective
Add 2 male + 2 female Telugu speaker voices with distinct characteristics.

### 4.2 Dataset Requirements (Per Speaker)

**Per Voice**:
- Duration: 2-4 hours clean speech
- Content: Varied sentences (emotions, tones)
- Quality: Studio-grade or high-quality field recording
- Format: 24kHz, mono, 16-bit WAV

**Totaldataset**: 12 hours (4 speakers × 3 hours avg)

### 4.3 Data Collection Options

#### Option A: Professional Voice Actors (Recommended)
**Cost**: $100-150 per voice actor × 4 = $400-600
**Quality**: Excellent (studio recordings)
**Time**: 1-2 weeks (hiring + recording)

**Script Requirements**:
- 500-1000 sentences covering:
  - Greetings and farewells
  - Questions and answers
  - Emotional expressions
  - Common phrases

#### Option B: Extract from High-Quality Content
**Cost**: Free
**Quality**: Good (if source is clean)
**Time**: 1 week (finding + extracting)

**Sources**:
- Telugu audiobooks (narrator voices)
- Professional Telugu news anchors
- Telugu YouTube educators

### 4.4 Speaker Embedding Training

Moshi doesn't have explicit voice cloning, so we use **speaker-conditioned fine-tuning**:

```python
# Add speaker embeddings to model
speaker_embeddings = torch.nn.Embedding(
    num_embeddings=4,  # 4 speakers
    embedding_dim=256
)

# Modify Moshi decoder to condition on speaker
class SpeakerConditionedMoshi(MoshiModel):
    def forward(self, audio_input, speaker_id):
        # Encode audio
        encoded = self.mimi_encoder(audio_input)
        
        # Get speaker embedding
        speaker_emb = speaker_embeddings(speaker_id)
        
        # Condition decoder on speaker
        output = self.mimi_decoder(encoded, speaker_emb)
        return output
```

### 4.5 Training Configuration

```yaml
# Continue from Stage 1 checkpoint
base_model: moshi-telugu-stage1

# Add speaker conditioning
speaker_count: 4
speaker_embedding_dim: 256

# Fine-tune decoder only
trainable_modules:
  - mimi_decoder  # Speaker-specific
  - speaker_embeddings

# Training params
batch_size: 8  # Smaller dataset
epochs: 10
learning_rate: 5e-6  # Lower LR
```

### 4.6 Training Time & Cost

```
Dataset: 12 hours (4 speakers × 3 hours)
Samples: ~2,160 (at 20s avg)
Steps per Epoch: 2,160 / 32 = 68 steps
Epochs: 10
Total Steps: 680 steps

Time: 680 × 3 seconds = 2,040 seconds ≈ 0.6 hours per epoch
Total: 6 hours

Cost: 6 hours × $0.49/hour = $2.94
```

### 4.7 Expected Outcomes

**After Stage 2**:
- 4 distinct speaker voices available
- Voice selection via speaker_id parameter
- Natural prosody per speaker
- Consistent voice across sessions

---

## 5. Stage 3: Emotional Expression

### 5.1 Objective
Generate speech with natural emotions (happy, sad, neutral, excited, angry).

### 5.2 Dataset Requirements

**Size**: 50 hours with emotion labels  
**Emotions**: 5 categories (10 hours each)
- Happy/Joyful
- Sad/Melancholic
- Neutral/Calm
- Excited/Energetic
- Angry/Frustrated

**Format**: Audio + emotion label

### 5.3 Data Collection

**Sources**:
1. **Telugu Movies** (emotional dialogues)
2. **Drama Performances** (exaggerated but useful)
3. **Emotional News** (varied tones)
4. **Podcasts** (natural emotion variation)

**Emotion Annotation**:
- Automatic (via speech emotion recognition model)
- Manual validation (sample 10-20%)

**Tool: Speech Emotion Recognition**
```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Load pre-trained emotion model
processor = Wav2Vec2Processor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

def predict_emotion(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    outputs = model(**inputs)
    emotion = torch.argmax(outputs.logits, dim=-1)
    
    # Map to label
    emotion_map = {0: "angry", 1: "happy", 2: "sad", 3: "neutral"}
    return emotion_map[emotion.item()]
```

### 5.4 Training Configuration

```yaml
# Continue from Stage 2
base_model: moshi-telugu-stage2

# Add emotion conditioning
emotion_count: 5
emotion_embedding_dim: 128

# Condition on emotion + speaker
trainable_modules:
  - temporal_transformer  # Emotion affects content
  - depth_transformer     # Emotion affects prosody
  - emotion_embeddings

# Training
batch_size: 4
epochs: 5
learning_rate: 3e-6
```

### 5.5 Training Time & Cost

```
Dataset: 50 hours
Samples: ~9,000
Steps per Epoch: 9,000 / 32 = 281 steps
Epochs: 5
Total Steps: 1,405 steps

Time: 1,405 × 4 seconds = 5,620 seconds ≈ 1.6 hours per epoch
Total: 8 hours

Cost: 8 hours × $0.49/hour = $3.92
```

### 5.6 Expected Outcomes

**After Stage 3**:
- Emotionally expressive speech
- 5 emotion categories available
- Controllable via `emotion_id` parameter
- Natural prosody variation

---

## 6. Complete Training Timeline

```
┌─────────────────────────────────────────────────────────┐
│ DAY 1-4: Data Collection & Preprocessing                │
│   ├─ YouTube download (automated)                       │
│   ├─ Diarization & segmentation                         │
│   ├─ Transcription (Whisper)                            │
│   └─ Quality filtering                                  │
├─────────────────────────────────────────────────────────┤
│ DAY 5-6: Stage 1 - Telugu Adaptation (24 hours)        │
│   └─ Fine-tune on 150 hours dataset                     │
├─────────────────────────────────────────────────────────┤
│ DAY 7: Stage 2 - Voice Cloning (6 hours)               │
│   └─ Add 4 speaker voices                               │
├─────────────────────────────────────────────────────────┤
│ DAY 8: Stage 3 - Emotional Expression (8 hours)        │
│   └─ Train emotion conditioning                         │
├─────────────────────────────────────────────────────────┤
│ DAY 9: Validation & Testing                             │
│   ├─ Generate samples                                   │
│   ├─ Measure quality metrics                            │
│   └─ Deploy final model                                 │
└─────────────────────────────────────────────────────────┘

Total: 9 days (4 days data + 5 days training/validation)
```

---

## 7. Cost Summary

### 7.1 Training Costs (L4 GPU on RunPod)

| Stage | Duration | Cost |
|-------|----------|------|
| Stage 1: Telugu Adaptation | 24 hours | $11.76 |
| Stage 2: Voice Cloning | 6 hours | $2.94 |
| Stage 3: Emotional Expression | 8 hours | $3.92 |
| **Training Total** | **38 hours** | **$18.62** |

### 7.2 Storage Costs

| Item | Size | Cost/Month |
|------|------|------------|
| Raw audio dataset | 50 GB | $5.00 |
| Processed dataset | 30 GB | $3.00 |
| Model checkpoints | 30 GB | $3.00 |
| **Storage Total** | **110 GB** | **$11.00** |

### 7.3 Optional: Professional Data Collection

| Item | Cost |
|------|------|
| Voice actors (4 × $125) | $500 |
| Emotion annotation (50 hours) | $200 |
| **Optional Total** | **$700** |

### 7.4 Grand Total

**Minimum Budget** (YouTube data only):
```
Training: $18.62
Storage (1 month): $11.00
═══════════════════════════════
Total: $29.62
```

**With Professional Data**:
```
Training: $18.62
Storage: $11.00
Professional data: $700
═══════════════════════════════
Total: $729.62
```

**Recommendation**: Start with YouTube data ($30), add professional voices if needed.

---

## 8. Quality Metrics & Validation

### 8.1 Evaluation Metrics

1. **Word Error Rate (WER)**: < 10% on Telugu test set
2. **Accent Naturalness**: MOS (Mean Opinion Score) > 4.0/5.0
3. **Speaker Similarity**: Cosine similarity > 0.85
4. **Emotion Recognition**: Accuracy > 80%
5. **Latency**: < 200ms on L4 GPU (no degradation)

### 8.2 Validation Pipeline

```bash
# Generate test samples
python generate_samples.py \
  --model moshi-telugu-final \
  --test_set telugu_test_100.json \
  --output_dir samples/

# Evaluate WER
python evaluate_wer.py \
  --audio_dir samples/ \
  --reference_transcripts telugu_test_100.json

# Human evaluation (MOS)
python mos_survey.py \
  --samples samples/ \
  --num_raters 10
```

---

## Next Steps

✅ Training plan completed  
⏭️ GPU recommendations and cost-benefit analysis (next document)  
⏭️ Phase 2: RunPod configuration  
⏭️ Phase 3: Application development
