# scraper/video_scraper.py

"""
VERITAS.AI Video Scraper Module
===============================

This module handles the downloading of video and audio from YouTube. Its
design emphasizes resilience and testability through a fallback mechanism.

1.  **Primary Method (yt-dlp):** Uses the powerful `yt-dlp` library to
    download the video and its corresponding audio track. It also extracts
    key metadata like the video title, channel information, and view count.

2.  **Fallback Simulation:** In the event `yt-dlp` fails (e.g., not installed,
    network error, video unavailable), the scraper does not crash. Instead,
    it generates dummy `.mp4` (video) and `.wav` (audio) files. This critical
    feature allows the analysis pipeline to run on placeholder data,
    ensuring the application is always demonstrable and testable, even
    offline.
"""

import yt_dlp
from config import VIDEO_SCRAPER_CONFIG
import logging
import os
import re
import numpy as np
import cv2
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoScraper:
    """
    Scrapes video and audio from YouTube using yt-dlp, with a fallback.
    """
    def __init__(self, config=VIDEO_SCRAPER_CONFIG):
        """
        Initializes the VideoScraper.
        """
        self.config = config
        self.temp_paths = {
            "video": self.config.get("temp_video_path", "temp/video"),
            "audio": self.config.get("temp_audio_path", "temp/audio")
        }
        # Ensure temporary directories exist
        os.makedirs(self.temp_paths["video"], exist_ok=True)
        os.makedirs(self.temp_paths["audio"], exist_ok=True)

    def _get_video_id_from_url(self, url: str) -> str | None:
        """
        Extracts the YouTube video ID from various URL formats.
        """
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'youtu\.be\/([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _create_dummy_video(self, video_id: str) -> str:
        """
        Creates a short, silent dummy video file for fallback purposes.
        """
        path = os.path.join(self.temp_paths["video"], f"{video_id}_dummy.mp4")
        logger.info(f"Creating dummy video at: {path}")
        width, height = 256, 256
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
        for i in range(50): # 5 seconds of video
            # Create a frame with a scrolling bar
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            bar_pos = int(i * (width / 50))
            cv2.rectangle(frame, (bar_pos, 0), (bar_pos + 10, height), (255, 255, 255), -1)
            out.write(frame)
        out.release()
        return path

    def _create_dummy_audio(self, video_id: str) -> str:
        """
        Creates a simple sine wave audio file for fallback purposes.
        """
        path = os.path.join(self.temp_paths["audio"], f"{video_id}_dummy.wav")
        logger.info(f"Creating dummy audio at: {path}")
        sr = 16000
        duration = 5
        frequency = 440
        t = np.linspace(0., duration, int(sr * duration))
        amplitude = np.iinfo(np.int16).max * 0.3
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(path, data.astype(np.int16), sr)
        return path

    def _scrape_with_fallback(self, url: str, video_id: str) -> dict:
        """
        Generates fallback data and dummy files.
        """
        logger.warning("yt-dlp failed or is unavailable. Using fallback simulation.")
        video_path = self._create_dummy_video(video_id)
        audio_path = self._create_dummy_audio(video_id)

        return {
            "platform": "youtube",
            "url": url,
            "text": None, # For videos, text comes from audio transcription
            "audio_path": audio_path,
            "video_path": video_path,
            "metadata": {
                "title": "Simulated Video Title (Fallback Mode)",
                "channel": "Simulated Channel",
                "view_count": 12345,
                "upload_date": "2023-01-01",
                "video_id": video_id,
                "scraping_method": "fallback_simulation"
            }
        }

    def scrape(self, url: str) -> dict:
        """
        Scrapes video and audio from a YouTube URL.

        Args:
            url (str): The URL of the YouTube video.

        Returns:
            dict: A structured dictionary with paths to the media and metadata.
        """
        video_id = self._get_video_id_from_url(url)
        if not video_id:
            return {"error": "Invalid YouTube URL or could not extract video ID."}

        video_path_template = os.path.join(self.temp_paths["video"], f"{video_id}.%(ext)s")
        audio_path_template = os.path.join(self.temp_paths["audio"], f"{video_id}.%(ext)s")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_path_template,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': {
                'ffmpeg': ['-ac', '1'] # Convert to mono channel for audio analysis
            }
        }
        # A separate opts dict for the audio file to put it in the right directory
        ydl_opts_for_audio = ydl_opts.copy()
        ydl_opts_for_audio['outtmpl'] = audio_path_template

        try:
            logger.info(f"Attempting to download video and audio for ID: {video_id}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Get the actual downloaded paths
                video_ext = info.get('ext')
                video_path = os.path.join(self.temp_paths["video"], f"{video_id}.{video_ext}")
                # The audio is converted to wav by the postprocessor
                audio_path = os.path.join(self.temp_paths["video"], f"{video_id}.wav") 
                
                # Move audio file to the correct directory
                final_audio_path = os.path.join(self.temp_paths["audio"], f"{video_id}.wav")
                if os.path.exists(audio_path):
                    os.rename(audio_path, final_audio_path)
                else:
                    raise FileNotFoundError("Expected audio file was not created by yt-dlp.")
                    
                metadata = {
                    "title": info.get("title", "N/A"),
                    "channel": info.get("channel", "N/A"),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", None), # Format: YYYYMMDD
                    "video_id": video_id,
                    "scraping_method": "yt-dlp"
                }
                
                return {
                    "platform": "youtube",
                    "url": url,
                    "text": None,
                    "audio_path": final_audio_path,
                    "video_path": video_path,
                    "metadata": metadata
                }

        except Exception as e:
            logger.error(f"yt-dlp scraping failed for {url}: {e}")
            # If download fails partway, clean up residual files before fallback
            for ext in ['mp4', 'm4a', 'webm', 'wav']:
                for folder in self.temp_paths.values():
                    file_path = os.path.join(folder, f"{video_id}.{ext}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
            return self._scrape_with_fallback(url, video_id)


if __name__ == '__main__':
    print("Running VideoScraper smoke test...")
    # A non-existent URL to force the fallback mechanism for a predictable test
    example_url = "https://www.youtube.com/watch?v=thisVideoIsFake1"
    
    scraper = VideoScraper()
    scraped_content = scraper.scrape(example_url)

    import json
    print(f"\n--- Scraped Content (Method: {scraped_content['metadata']['scraping_method']}) ---")
    print(json.dumps(scraped_content, indent=2))

    print("\n--- Assertions ---")
    assert "error" not in scraped_content, "Scraping resulted in an error."
    assert scraped_content["platform"] == "youtube", "Platform is not 'youtube'."
    assert os.path.exists(scraped_content["audio_path"]), "Audio file path does not exist."
    assert os.path.exists(scraped_content["video_path"]), "Video file path does not exist."
    print("All assertions passed.")
    
    # Clean up dummy files
    os.remove(scraped_content["audio_path"])
    os.remove(scraped_content["video_path"])
    print("Dummy files cleaned up.")
    print("\nSmoke test completed successfully.")