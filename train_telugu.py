"""Fine-tune SpeechT5 on Telugu data"""
import torch
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech,
    Trainer, TrainingArguments
)
from datasets import load_dataset, Audio
import os
from config import *

def train_telugu_model():
    print("="*60)
    print("Telugu Fine-Tuning")
    print("="*60)
    
    # Load base model
    print("\nLoading base model...")
    processor = SpeechT5Processor.from_pretrained(f"{MODELS_DIR}/speecht5")
    model = SpeechT5ForTextToSpeech.from_pretrained(f"{MODELS_DIR}/speecht5")
    
    # Freeze encoder (faster training)
    for param in model.speecht5.encoder.parameters():
        param.requires_grad = False
    
    print("✓ Base model loaded (encoder frozen)")
    
    # Load Telugu dataset
    print("\nPreparing Telugu dataset...")
    dataset = load_dataset("audiofolder", data_dir=DATA_DIR)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    print(f"✓ Dataset loaded: {len(dataset['train'])} samples")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=f"{MODELS_DIR}/speecht5_telugu",
        per_device_train_batch_size=TRAINING_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        dataloader_num_workers=4,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )
    
    print("\nStarting training...")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Batch size: {TRAINING_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    trainer.train()
    
    # Save final model
    model.save_pretrained(f"{MODELS_DIR}/speecht5_telugu")
    processor.save_pretrained(f"{MODELS_DIR}/speecht5_telugu")
    
    print("\n"+"="*60)
    print("✓ Telugu model training complete!")
    print(f"Saved to: {MODELS_DIR}/speecht5_telugu")
    print("="*60)

if __name__ == "__main__":
    train_telugu_model()
