# Telugu YouTube Data Sources
## For 24-Hour POC Training Data Collection

**Goal**: Collect 10-20 hours of Telugu speech in 2-3 hours

---

## 🎯 RECOMMENDED SOURCES

### 1. Telugu News Channels (Best Quality)

**TV9 Telugu**
- Channel: https://www.youtube.com/@TV9Telugu
- Search: "TV9 Telugu news today"
- Quality: Excellent (professional audio)
- Content: News bulletins, debates
- Duration: 30-60 min videos

**ABN Telugu**
- Channel: https://www.youtube.com/@abntelugu
- Search: "ABN Telugu live"
- Quality: Excellent
- Content: News, interviews
- Duration: Long streams available

**10TV Telugu**
- Channel: https://www.youtube.com/@10TVTelugu
- Search: "10TV Telugu news"
- Quality: Very good
- Content: News coverage

### 2. Telugu Podcasts (Natural Conversation)

**Telugu Podcast**
- Search: "telugu podcast 2024"
- Quality: Good
- Content: Interviews, discussions
- Duration: 30-90 min

**Talking Movies with iDream**
- Search: "talking movies telugu"
- Quality: Good
- Content: Film industry interviews
- Duration: 45-60 min

### 3. Telugu Educational Content

**Easy Telugu**
- Search: "easy telugu lessons"
- Quality: Excellent (clear speech)
- Content: Language lessons
- Duration: 10-30 min

**Telugu Vlogs**
- Search: "telugu daily vlogs"
- Quality: Variable
- Content: Daily life, travel
- Duration: 15-45 min

---

## 📋 SEARCH QUERIES TO USE

Copy these into YouTube search:

```
telugu podcast interview 2024
telugu news today live
telugu conversation podcast
telugu interview latest
telugu documentary
telugu educational videos
telugu talk show
telugu radio show
```

---

## 🎬 SAMPLE URLs (Replace with Real Ones)

**You need to find current videos. Here's the format:**

```python
# In download_telugu.py, replace URLs with real ones:

urls = [
    # News (5-7 videos, 30-60 min each)
    "https://www.youtube.com/watch?v=XXXXX",  # TV9 news
    "https://www.youtube.com/watch?v=YYYYY",  # ABN news
    
    # Podcasts (3-5 videos, 45-90 min each)
    "https://www.youtube.com/watch?v=ZZZZZ",  # Telugu podcast
    
    # Interviews (3-5 videos, 30-60 min each)
    "https://www.youtube.com/watch?v=AAAAA",  # Celebrity interview
    
    # Educational (2-3 videos, 20-40 min each)
    "https://www.youtube.com/watch?v=BBBBB",  # Telugu lessons
]

# Target: 10-15 videos = 10-20 hours total
```

---

## ⚡ QUICK COLLECTION STRATEGY (3 Hours)

### Hour 1: Find Videos
1. Open YouTube
2. Search each query above
3. Filter by: Upload date (recent), Duration (>20 min)
4. Copy 10-15 video URLs
5. Verify they're Telugu (not dubbed)

### Hour 2: Download
1. Add URLs to script
2. Run download script
3. Let it run in background

### Hour 3: Verify
1. Check downloaded files
2. Play random samples
3. Verify audio quality
4. Delete any non-Telugu or poor quality

---

## 🔍 HOW TO VERIFY TELUGU AUDIO

**Good Audio:**
- Clear speech
- Minimal background noise
- Native Telugu (not code-switching)
- Professional recording quality

**Bad Audio (Avoid):**
- Music-heavy (songs)
- Heavy background noise
- Dubbed content (originally other language)
- Low volume/quality

---

## 💡 PRO TIPS

### For Best Results:
1. **Prefer news channels** - Best audio quality
2. **Avoid music videos** - Not useful for speech training
3. **Check recent uploads** - Better quality
4. **Longer is better** - 30-60 min videos ideal
5. **Verify language** - Some channels mix Telugu/Hindi

### Time Saving:
1. **Download in parallel** - 5-10 videos at once
2. **Don't wait to finish** - Start training with partial data
3. **Quality > Quantity** - 10 hours good audio > 20 hours bad
4. **Use playlists** - News channels have daily playlists

### Storage Management:
- Each hour ≈ 1-2 GB
- 20 hours ≈ 20-40 GB
- Make sure you have 50GB free on RunPod

---

## 📊 TARGET METRICS

| Metric | Target |
|--------|--------|
| **Total Duration** | 10-20 hours |
| **Number of Videos** | 10-15 |
| **Audio Quality** | >128 kbps |
| **SNR** | >15 dB |
| **Language** | 100% Telugu |
| **Speakers** | 10+ different |

---

## 🚨 BACKUP PLAN

**If you can't find enough videos:**

1. **Use existing datasets**:
   - Common Voice Telugu (if available)
   - Mozilla Common Voice
   - Search for Telugu speech datasets

2. **Mix with English**:
   - Fine-tune on available Telugu
   - Demo with Telugu + English hybrid

3. **Focus on architecture**:
   - Demo with English first
   - Show MD the architecture works
   - Explain Telugu training is next step

---

## ✅ VERIFICATION CHECKLIST

Before proceeding to training:

- [ ] Downloaded 10+ Telugu videos
- [ ] Total duration >10 hours
- [ ] Audio quality checked
- [ ] All files in WAV format
- [ ] Files in /workspace/telugu_data/
- [ ] Disk space sufficient (50GB+)

---

## 🎯 READY TO DOWNLOAD?

1. Find 10-15 Telugu YouTube videos (1 hour)
2. Copy URLs to download script
3. Run download (2 hours)
4. Verify quality
5. Proceed to fine-tuning!

**Start searching now!** 🔍
