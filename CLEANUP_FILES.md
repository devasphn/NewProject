# Project Cleanup Guide

## Files to DELETE (Unnecessary/Duplicate)

### Documentation Files to DELETE (24 → 5)
```bash
# DELETE these (outdated/duplicate documentation)
rm ADVANCED_ARCHITECTURE_ANALYSIS.md      # Old analysis
rm COMPLETE_COMMAND_REFERENCE.md          # Outdated commands
rm COMPLETE_SETUP_COMMANDS.md             # Duplicate setup
rm COMPREHENSIVE_PROJECT_ANALYSIS.md      # Old analysis
rm FIX_RATE_LIMIT_CHECKLIST.md            # YouTube related - not needed
rm FIX_YOUTUBE_BOT_DETECTION.md           # YouTube related - not needed
rm FROM_SCRATCH_SETUP_GUIDE.md            # Duplicate
rm FULL_S2S_RESEARCH_AND_PLAN.md          # Outdated plan
rm MASTER_DOWNLOAD_GUIDE.md               # Replaced by DATA_1000H_PER_LANGUAGE.md
rm PRODUCTION_DATA_PLAN.md                # Outdated
rm PRODUCTION_DOWNLOAD_GUIDE.md           # Duplicate
rm QUICK_START_RUNPOD.md                  # Duplicate setup
rm RECOVERY_PLAN_V1.md                    # Outdated recovery
rm RUNPOD_SETUP_COMMANDS.md               # Duplicate
rm RUNPOD_TEMPLATE_SETUP.md               # Not needed
rm START_DATA_COLLECTION.md               # Outdated
rm STORAGE_CALCULATOR.md                  # One-time calculation
rm TECHNICAL_DOCUMENTATION.md             # Outdated
rm TELUGU_200H_DOWNLOAD.md                # Replaced by new guide
rm TRAINING_S2S_FOR_CONVERSATION.md       # Outdated
```

### KEEP these documentation files:
- `PRODUCTION_CODEC_GUIDE.md` - Main codec guide
- `PROJECT_SUMMARY.md` - Project overview
- `DATA_1000H_PER_LANGUAGE.md` - NEW data guide
- `BUSINESS_PLAN_INDIA.md` - NEW business plan
- `README.md` (create if not exists)

---

### Download Scripts to DELETE (16 → 3)
```bash
# DELETE these Python download scripts (duplicates)
rm download_all_telugu.py                 # Duplicate
rm download_all_telugu_data.py            # Duplicate  
rm download_telugu_data_PRODUCTION.py     # Duplicate
rm download_telugu_fixed.py               # Duplicate

# DELETE these Shell download scripts (outdated)
rm download_all_channels.sh               # YouTube related
rm download_free_datasets.sh              # Outdated
rm download_massive_data.sh               # Too many languages
rm download_remaining_data.sh             # Outdated
rm download_single_channel.sh             # YouTube related
rm download_telugu_datasets.sh            # Duplicate
rm download_telugu_wget.sh                # Duplicate
rm download_tier1_SAFE.sh                 # Outdated tier
rm download_tier1_only.sh                 # Outdated tier
rm download_tier1_optimized.sh            # Outdated tier
```

### KEEP these download scripts:
- `download_indicvoices.py` - For IndicVoices HuggingFace
- `download_kathbath.py` - For Kathbath HuggingFace
- Create new: `download_production_data.sh` - Single source of truth

---

### Python Files to DELETE (Duplicates/Outdated)
```bash
# Telugu-specific files (consolidate)
rm telugu_agent_fast.py                   # Duplicate agent
rm telugu_agent_streaming.py              # Duplicate agent
rm telugu_voice_agent.py                  # Duplicate agent
rm telugu_voice_agent_complete.py         # Duplicate agent
rm telugu_voice_agent_realtime.py         # Duplicate agent
rm telugu_voice_assistant_hybrid.py       # Duplicate agent
rm telugu_codec.py                        # Old codec version
rm realtime_telugu_s2s.py                 # Duplicate

# Demo/test files
rm demo_complete_s2s.py                   # Demo only
rm demo_voice_poc.py                      # PoC only
rm diagnose_s2s.py                        # One-time diagnostic

# Old training scripts
rm train_codec.py                         # Old version
rm train_s2s.py                           # Old version
rm train_s2s_conversation.py              # Old version

# Streaming servers (keep only one)
rm streaming_server.py                    # Keep advanced version

# Misc
rm cleanup_and_organize.py                # One-time use
rm generate_telugu_conversations.py       # Specific to Telugu
rm prepare_speaker_data.py                # One-time use
rm context_manager.py                     # May not be needed
rm upload_server.py                       # Not needed now
rm data_collection.py                     # YouTube related
rm debug_validation_data.py               # Debug only
rm QUICK_START_AFTER_COOKIES.sh           # YouTube related
rm COMPLETE_FIX_COMMANDS.sh               # Old fixes
rm verify_setup.sh                        # One-time use
rm RUNPOD_ENV_VARS.txt                    # Environment specific
```

---

## Files to KEEP (Essential)

### Core Model Files (6)
```
codec_production.py           # Main production codec
telugu_codec_fixed.py         # Backup/reference codec
discriminator_dac.py          # GAN discriminator
s2s_transformer.py            # S2S model architecture
speaker_embeddings.py         # Speaker embedding module
config.py                     # Configuration
```

### Training Scripts (4)
```
train_codec_production.py     # Main codec training
train_codec_dac.py            # Alternative training
train_s2s_production.py       # S2S training
train_speakers.py             # Speaker training
```

### Server/Inference (3)
```
streaming_server_advanced.py  # Main streaming server
realtime_codec_server.py      # Codec server
realtime_s2s_agent.py         # S2S agent
```

### Utilities (5)
```
verify_everything.py          # Verification script
preflight_check.py            # Pre-training check
benchmark_latency.py          # Latency testing
system_test.py                # System testing
test_s2s_model.py             # Model testing
realtime_codec_test.py        # Codec testing
```

### Download Scripts (3)
```
download_indicvoices.py       # HuggingFace download
download_kathbath.py          # HuggingFace download
(create) download_production_data.sh  # Main download script
```

### Configuration (3)
```
requirements_new.txt          # Dependencies
data_sources_PRODUCTION.yaml  # Data sources config
.gitignore                    # Git ignore
```

### Documentation (5)
```
PRODUCTION_CODEC_GUIDE.md     # Codec guide
PROJECT_SUMMARY.md            # Overview
DATA_1000H_PER_LANGUAGE.md    # Data guide (NEW)
BUSINESS_PLAN_INDIA.md        # Business plan (NEW)
README.md                     # Main readme
```

---

## Quick Cleanup Command

```bash
cd /workspace/NewProject

# Delete unnecessary .md files
rm -f ADVANCED_ARCHITECTURE_ANALYSIS.md COMPLETE_COMMAND_REFERENCE.md \
      COMPLETE_SETUP_COMMANDS.md COMPREHENSIVE_PROJECT_ANALYSIS.md \
      FIX_RATE_LIMIT_CHECKLIST.md FIX_YOUTUBE_BOT_DETECTION.md \
      FROM_SCRATCH_SETUP_GUIDE.md FULL_S2S_RESEARCH_AND_PLAN.md \
      MASTER_DOWNLOAD_GUIDE.md PRODUCTION_DATA_PLAN.md \
      PRODUCTION_DOWNLOAD_GUIDE.md QUICK_START_RUNPOD.md \
      RECOVERY_PLAN_V1.md RUNPOD_SETUP_COMMANDS.md \
      RUNPOD_TEMPLATE_SETUP.md START_DATA_COLLECTION.md \
      STORAGE_CALCULATOR.md TECHNICAL_DOCUMENTATION.md \
      TELUGU_200H_DOWNLOAD.md TRAINING_S2S_FOR_CONVERSATION.md

# Delete unnecessary download scripts
rm -f download_all_telugu.py download_all_telugu_data.py \
      download_telugu_data_PRODUCTION.py download_telugu_fixed.py \
      download_all_channels.sh download_free_datasets.sh \
      download_massive_data.sh download_remaining_data.sh \
      download_single_channel.sh download_telugu_datasets.sh \
      download_telugu_wget.sh download_tier1_SAFE.sh \
      download_tier1_only.sh download_tier1_optimized.sh

# Delete duplicate Telugu agents
rm -f telugu_agent_fast.py telugu_agent_streaming.py \
      telugu_voice_agent.py telugu_voice_agent_complete.py \
      telugu_voice_agent_realtime.py telugu_voice_assistant_hybrid.py \
      telugu_codec.py realtime_telugu_s2s.py

# Delete misc files
rm -f demo_complete_s2s.py demo_voice_poc.py diagnose_s2s.py \
      train_codec.py train_s2s.py train_s2s_conversation.py \
      streaming_server.py cleanup_and_organize.py \
      generate_telugu_conversations.py prepare_speaker_data.py \
      context_manager.py upload_server.py data_collection.py \
      debug_validation_data.py QUICK_START_AFTER_COOKIES.sh \
      COMPLETE_FIX_COMMANDS.sh verify_setup.sh RUNPOD_ENV_VARS.txt

echo "✅ Cleanup complete!"
echo ""
echo "Remaining files:"
ls -la *.py *.sh *.md 2>/dev/null | wc -l
```

---

## Final File Count

| Category | Before | After |
|----------|--------|-------|
| .md files | 24 | 5 |
| .py files | 47 | 20 |
| .sh files | 12 | 1 |
| **Total** | **83** | **26** |

**Reduction: 69% fewer files!**
