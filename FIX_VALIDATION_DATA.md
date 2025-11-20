# 🚨 FIX: 0 Validation Samples

## 📊 Problem

Your dataset after balancing:
```
Total: 36 files → Balanced to 8 files (2 per speaker)
Split: Train=4, Val=0, Test=4
```

**Issue**: Speaker 2 (Priya) only had 2 samples, so balancing reduced ALL speakers to 2!

## ✅ SOLUTION: Use All Files Without Balancing

Run this command to use **all 36 files**:

```bash
# Re-run preparation WITHOUT balancing
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data \
    --no_balance
```

## 📈 Expected Results

With `--no_balance`:
```
Speaker 0 (Arjun): 8 samples
  Train: 6, Val: 1, Test: 1

Speaker 1 (Ravi): 8 samples
  Train: 6, Val: 1, Test: 1

Speaker 2 (Priya): 2 samples
  Train: 1, Val: 0, Test: 1

Speaker 3 (Lakshmi): 18 samples
  Train: 14, Val: 2, Test: 2

TOTAL: 36 samples
  Train: 27
  Val: 4 ← Fixed!
  Test: 5
```

## 🎯 Why This Works

1. **Keeps all data** - No wasteful balancing
2. **Val gets samples** - Enough for validation
3. **Imbalance is OK** - Model learns from all speakers
4. **More training data** - 27 vs 4 samples!

## 🚀 Full Command Sequence

```bash
# 1. Re-prepare data (no balancing)
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data \
    --no_balance

# 2. Verify splits
echo "Train: $(cat /workspace/speaker_data/train_split.json | grep -o '"speaker_id"' | wc -l)"
echo "Val: $(cat /workspace/speaker_data/val_split.json | grep -o '"speaker_id"' | wc -l)"
echo "Test: $(cat /workspace/speaker_data/test_split.json | grep -o '"speaker_id"' | wc -l)"

# 3. Continue with training
python train_speaker_embeddings.py \
    --data_dir /workspace/speaker_data \
    --model_dir /workspace/models/speaker_model \
    --batch_size 8 \
    --num_epochs 50
```

## 💡 Alternative: If You Want Some Balancing

Use `--min_samples` to set a floor (default is 5):

```bash
# Balance but keep at least 5 per speaker (if available)
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data \
    --min_samples 2
```

This will still balance but won't be as aggressive.

## ✅ Run This NOW

```bash
python prepare_speaker_data.py \
    --data_dir /workspace/telugu_data/raw \
    --output_dir /workspace/speaker_data \
    --no_balance
```

Then continue with your training!
