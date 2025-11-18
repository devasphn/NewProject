# Quick Installation Fix

## Issue
`flash-attn` installation failed because it needs PyTorch installed first.

## Solution
Install packages in the correct order:

```bash
# 1. STOP the current installation if it's still running
# Press Ctrl+C if needed

# 2. Install base requirements (already started)
# This should complete now that flash-attn is removed from requirements_new.txt
pip install -r requirements_new.txt

# 3. Install Flash Attention SEPARATELY
pip install flash-attn --no-build-isolation

# 4. Install additional packages
pip install websockets==12.0 python-multipart==0.0.9 uvloop==0.19.0

# 5. Verify everything installed correctly
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import flash_attn; print('✓ Flash Attention: Installed')"
python -c "import einops; print('✓ Einops: Installed')"
```

## What Changed
- Removed `flash-attn==2.5.0` from `requirements_new.txt`
- Install it separately AFTER torch is installed
- Updated `DEPLOYMENT_MANUAL_V2.md` with correct order

## Continue From Here
Once all packages are installed, continue with **Step 6: Environment Variables** in the deployment manual (line 204).

---

## If Flash Attention Still Fails

Try these alternatives:

### Option 1: Install latest version (recommended)
```bash
pip install flash-attn --no-build-isolation
```

### Option 2: Use pre-built wheel
```bash
pip install flash-attn --no-build-isolation --no-deps
pip install einops packaging ninja
```

### Option 3: Skip Flash Attention (slower but works)
```bash
# System will work but with slightly higher latency (~180ms instead of 150ms)
# Flash Attention is an optimization, not a requirement
```

To disable Flash Attention in code if needed:
```python
# In s2s_transformer.py line 26:
config = S2SConfig(use_flash_attn=False)  # Changed from True
```

---

## Current Status
✅ Fixed: `requirements_new.txt` updated  
✅ Fixed: `DEPLOYMENT_MANUAL_V2.md` updated  
▶️ **Next**: Run the commands above to complete installation

All files pushed to GitHub with the fix!