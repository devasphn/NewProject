"""
Fine-tune EnCodec on Telugu Audio - POC Version
Quick fine-tuning for demonstration purposes
"""

import torch
import torchaudio
from encodec import EncodecModel
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

class TeluguAudioDataset(Dataset):
    def __init__(self, audio_dir, segment_length=24000*10):  # 10 seconds at 24kHz
        self.audio_files = list(Path(audio_dir).rglob("*.wav"))
        self.segment_length = segment_length
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed (EnCodec uses 24kHz)
            if sr != 24000:
                waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Get random segment or pad
            if waveform.shape[1] > self.segment_length:
                start = np.random.randint(0, waveform.shape[1] - self.segment_length)
                waveform = waveform[:, start:start + self.segment_length]
            elif waveform.shape[1] < self.segment_length:
                # Pad
                padding = self.segment_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence on error
            return torch.zeros(1, self.segment_length)


def train_encodec(train_dir, val_dir, output_dir, epochs=10, batch_size=4):
    """
    Fine-tune EnCodec on Telugu data
    """
    print("="*70)
    print("FINE-TUNING ENCODEC FOR TELUGU - POC")
    print("="*70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load pretrained EnCodec
    print("Loading pretrained EnCodec model (24kHz)...")
    model = EncodecModel.encodec_model_24khz()
    model = model.to(device)
    model.set_target_bandwidth(6.0)  # 6 kbps
    print("✅ Model loaded")
    print()
    
    # Freeze encoder, only train decoder (faster fine-tuning)
    print("Freezing encoder (training decoder only)...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TeluguAudioDataset(train_dir)
    val_dataset = TeluguAudioDataset(val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    print("="*70)
    print()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                encoded_frames = model.encode(batch)
                decoded = model.decode(encoded_frames)
                
                # Ensure same length
                min_len = min(decoded.shape[-1], batch.shape[-1])
                decoded = decoded[..., :min_len]
                batch = batch[..., :min_len]
                
                # Simple L1 loss
                loss = torch.nn.functional.l1_loss(decoded, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for batch in pbar:
                batch = batch.to(device)
                
                with torch.cuda.amp.autocast():
                    encoded_frames = model.encode(batch)
                    decoded = model.decode(encoded_frames)
                    
                    # Ensure same length
                    min_len = min(decoded.shape[-1], batch.shape[-1])
                    decoded = decoded[..., :min_len]
                    batch = batch[..., :min_len]
                    
                    loss = torch.nn.functional.l1_loss(decoded, batch)
                
                val_loss += loss.item()
                val_batches += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / val_batches
        
        # Save history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print()
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print("="*70)
        print()
    
    # Save model
    print("Saving fine-tuned model...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "encodec_telugu_poc.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to {history_path}")
    
    print()
    print("="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Test the model: python test_telugu_codec.py")
    print("  2. Check training history: cat /workspace/models/training_history.json")
    print()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune EnCodec on Telugu audio")
    parser.add_argument("--train_dir", type=str, default="/workspace/telugu_poc_data/raw/train",
                        help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="/workspace/telugu_poc_data/raw/val",
                        help="Validation data directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/models",
                        help="Output directory for saved model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    
    args = parser.parse_args()
    
    print()
    print("Configuration:")
    print(f"  Train dir: {args.train_dir}")
    print(f"  Val dir: {args.val_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print()
    
    model = train_encodec(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
