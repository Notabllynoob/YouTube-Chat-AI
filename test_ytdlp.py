import yt_dlp
import webvtt
import glob
import os

def test_video(url):
    print(f"Testing URL: {url}")
    
    # Configure yt-dlp to download subtitles only
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US', 'en-orig', 'en-GB'], 
        'outtmpl': 'test_sub',
        'quiet': True,
        'no_warnings': True
    }

    try:
        # cleanup old
        for f in glob.glob("test_sub*"):
            os.remove(f)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'Unknown Title')
            print(f"✅ Success! Found video: '{video_title}'")

        # Check for file
        vtt_files = glob.glob("test_sub*.vtt")
        if vtt_files:
            print(f"✅ Success! Downloaded subtitle file: {vtt_files[0]}")
            # cleanup
            for f in glob.glob("test_sub*"):
                os.remove(f)
        else:
            print("❌ Warning: Video found, but no subtitles downloaded.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test with a known safe video (Google DeepMind)
    test_video("https://www.youtube.com/watch?v=AJP6K2_rr90")
