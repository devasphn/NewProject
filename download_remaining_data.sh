#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#   DOWNLOAD REMAINING DATA - Automated Script
#   Current: ~509 hours downloaded
#   After this script: ~2,500+ hours
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd /workspace/NewProject

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "   DOWNLOADING REMAINING DATA"
echo "   Current: ~509 hours | Target: ~2,500+ hours"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# BENGALI - Remaining 12 files (~150 hours more)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📥 Downloading Bengali (remaining 12 files, ~150h)..."

wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_4.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_5.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_6.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_7.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_8.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_9.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_a.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_b.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_c.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_d.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_e.zip
wget -c -P data/bengali https://www.openslr.org/resources/53/asr_bengali_f.zip

# ═══════════════════════════════════════════════════════════════════════════════
# ENGLISH - LibriSpeech (remaining ~860 hours)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📥 Downloading English LibriSpeech (remaining ~860h)..."

wget -c -P data/english https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c -P data/english https://www.openslr.org/resources/12/train-other-500.tar.gz

# ═══════════════════════════════════════════════════════════════════════════════
# LJSPEECH - High quality English (24 hours)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📥 Downloading LJSpeech (24h high quality)..."

wget -c -P data/english https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# ═══════════════════════════════════════════════════════════════════════════════
# MLS - Multilingual (Optional but recommended for robust codec)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📥 Downloading MLS German (~2000h) - for robust multilingual codec..."

wget -c -P data/german https://dl.fbaipublicfiles.com/mls/mls_german_opus.tar.gz

# Uncomment these if you have storage space:
# wget -c -P data/french https://dl.fbaipublicfiles.com/mls/mls_french_opus.tar.gz
# wget -c -P data/spanish https://dl.fbaipublicfiles.com/mls/mls_spanish_opus.tar.gz

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT ALL
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📦 Extracting all archives..."

# Bengali
cd data/bengali
for f in *.zip; do
    [ -f "$f" ] && unzip -o "$f" && rm "$f"
done
cd ../..

# English tar.gz
cd data/english
for f in *.tar.gz; do
    [ -f "$f" ] && tar -xzf "$f" && rm "$f"
done
for f in *.tar.bz2; do
    [ -f "$f" ] && tar -xjf "$f" && rm "$f"
done
cd ../..

# German MLS
if [ -d "data/german" ]; then
    cd data/german
    for f in *.tar.gz; do
        [ -f "$f" ] && tar -xzf "$f" && rm "$f"
    done
    cd ../..
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "✅ DOWNLOAD COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Data Summary:"
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  Telugu:     ~10 hours"
echo "  Tamil:      ~15 hours"
echo "  Kannada:    ~18 hours"
echo "  Malayalam:  ~12 hours"
echo "  Hindi:      ~95 hours"
echo "  Bengali:    ~200 hours (all 16 files)"
echo "  Odia:       ~95 hours"
echo "  Marathi:    ~104 hours"
echo "  Gujarati:   ~10 hours"
echo "  English:    ~984 hours (LibriSpeech + LJSpeech)"
echo "  German:     ~2000 hours (MLS)"
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  TOTAL:      ~3,543 hours"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "To add Telugu HuggingFace data (460+ more hours), run:"
echo "  python download_all_telugu.py"
echo ""
echo "To start training, run:"
echo "  python train_codec_production.py --data_dirs data/telugu data/tamil data/kannada data/malayalam data/hindi data/bengali data/odia data/marathi data/gujarati data/english data/german --batch_size 32 --num_epochs 100"
