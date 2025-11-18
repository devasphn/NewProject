"""Download Telugu training data from YouTube"""
import yt_dlp
import os
from config import *

# Telugu video URLs - REPLACE WITH REAL URLS from telugu_videos.txt
urls = [
    # Add your URLs here from telugu_videos.txt
    # Example: "https://www.youtube.com/watch?v=xxxxx",
]

def download_telugu_data():
    """Download Telugu videos as WAV files"""
    
    if not urls or urls[0].startswith("REPLACE"):
        print("❌ Error: Please add real YouTube URLs to download_telugu.py")
        print("See telugu_videos.txt for instructions")
        return False
    
    print(f"Downloading {len(urls)} Telugu videos...")
    print(f"Target: {TELUGU_TRAINING_HOURS} hours")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': f'{DATA_DIR}/%(id)s.%(ext)s',
        'quiet': False,
        'no_warnings': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
    
    # Count total duration
    import soundfile as sf
    total_duration = 0
    for file in os.listdir(DATA_DIR):
        if file.endswith('.wav'):
            filepath = os.path.join(DATA_DIR, file)
            data, sr = sf.read(filepath)
            duration = len(data) / sr / 3600  # hours
            total_duration += duration
    
    print(f"\n✓ Downloaded {total_duration:.1f} hours of Telugu audio")
    return total_duration >= TELUGU_TRAINING_HOURS

if __name__ == "__main__":
    success = download_telugu_data()
    exit(0 if success else 1)
