# analyzer/pipeline.py

"""
VERITAS.AI Analysis Pipeline
============================

This module orchestrates the entire analysis process, from data input to
final report generation. It integrates all other analyzer components to provide
a streamlined workflow for evaluating content from different sources.

The pipeline is designed to be flexible and handle various content types,
including news articles, Reddit posts, and YouTube videos.
"""

from .text_analysis import TextAnalyzer
from .audio_analysis import AudioAnalyzer
from .video_analysis import VideoAnalyzer
from .trust_credibility import ScoreCalculator
from .xai_explanations import XAI
from config import (
    TEXT_ANALYZER_CONFIG,
    AUDIO_ANALYZER_CONFIG,
    VIDEO_ANALYZER_CONFIG
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """
    Coordinates the full analysis workflow.
    """
    def __init__(self):
        """
        Initializes all the necessary analyzer components.
        """
        logger.info("Initializing VERITAS.AI Analysis Pipeline...")
        self.text_analyzer = TextAnalyzer(config=TEXT_ANALYZER_CONFIG)
        self.audio_analyzer = AudioAnalyzer(config=AUDIO_ANALYZER_CONFIG)
        self.video_analyzer = VideoAnalyzer(config=VIDEO_ANALYZER_CONFIG)
        self.score_calculator = ScoreCalculator()
        self.xai = XAI()
        logger.info("Analysis Pipeline initialized successfully.")

    def run(self, scraped_data: dict) -> dict:
        """
        Executes the full analysis pipeline on the scraped data.

        Args:
            scraped_data (dict): A dictionary containing the scraped content.
                                 Expected keys vary based on content type:
                                 - 'platform': 'news', 'reddit', or 'youtube'
                                 - 'text': The main text content (article body, post text)
                                 - 'audio_path': Path to the downloaded audio file (for YouTube)
                                 - 'video_path': Path to the downloaded video file (for YouTube)
                                 - 'metadata': A dict with source-specific info

        Returns:
            dict: A comprehensive dictionary containing all analysis results.
        """
        platform = scraped_data.get("platform")
        if not platform:
            return {"error": "Platform not specified in scraped data."}

        logger.info(f"Starting analysis for platform: {platform.upper()}")

        # --- Initialize result containers ---
        text_analysis_results = {}
        audio_analysis_results = {}
        video_analysis_results = {}

        # --- Text Analysis (Common to all platforms) ---
        text_content = scraped_data.get("text")
        if text_content:
            logger.info("Performing text analysis...")
            text_analysis_results = self.text_analyzer.analyze(text_content)
        else:
            logger.warning("No text content provided for analysis.")

        # --- Audio/Video Analysis (YouTube) ---
        if platform == "youtube":
            audio_path = scraped_data.get("audio_path")
            video_path = scraped_data.get("video_path")

            if audio_path:
                logger.info("Performing audio analysis...")
                audio_analysis_results = self.audio_analyzer.analyze(audio_path)
                # If transcription succeeded, run text analysis on it
                if "transcription" in audio_analysis_results and \
                   "text" in audio_analysis_results["transcription"]:
                    transcribed_text = audio_analysis_results["transcription"]["text"]
                    logger.info("Performing text analysis on transcribed audio...")
                    text_analysis_results = self.text_analyzer.analyze(transcribed_text)
            else:
                logger.warning("No audio path provided for YouTube analysis.")

            if video_path:
                logger.info("Performing video analysis...")
                video_analysis_results = self.video_analyzer.analyze(video_path)
            else:
                logger.warning("No video path provided for YouTube analysis.")

        # --- Scoring ---
        logger.info("Calculating final scores...")
        scores = self.score_calculator.calculate_all_scores(
            text_results=text_analysis_results,
            audio_results=audio_analysis_results,
            video_results=video_analysis_results,
            metadata=scraped_data.get("metadata", {})
        )

        # --- Explainable AI (XAI) ---
        logger.info("Generating XAI explanation...")
        explanation = self.xai.explain_results(
            scores=scores,
            platform=platform
        )

        # --- Compile Final Report ---
        final_report = {
            "platform": platform,
            "source": scraped_data.get("url", "N/A"),
            "scores": scores,
            "explanation": explanation,
            "detailed_analysis": {
                "text_analysis": text_analysis_results,
                "audio_analysis": audio_analysis_results,
                "video_analysis": video_analysis_results
            },
            "metadata": scraped_data.get("metadata", {})
        }

        logger.info("Analysis pipeline completed.")
        return final_report

if __name__ == '__main__':
    print("Running AnalysisPipeline smoke test...")
    # This test demonstrates how the pipeline would be called with mock data.
    # In a real scenario, this data would come from the scraper modules.

    # --- Mock Scraped Data (Simulating a YouTube video) ---
    mock_youtube_data = {
        "platform": "youtube",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "text": None,  # Text comes from audio transcription for videos
        "audio_path": "sample.wav", # Dummy path, requires a real file for a real test
        "video_path": "sample.mp4", # Dummy path, requires a real file for a real test
        "metadata": {
            "channel_name": "Official Channel",
            "views": 1000000,
            "likes": 50000,
            "upload_date": "2023-10-26"
        }
    }

    # To run a real test, you'd need dummy files and libraries.
    # We will simulate the output instead to avoid heavy dependencies in a simple test.
    print("\n--- Simulating Pipeline Run ---")
    try:
        # We can't run the full pipeline without actual models and files.
        # This block serves as a conceptual demonstration.
        
        # pipeline = AnalysisPipeline()
        # To run this, you would need to:
        # 1. Set up a 'sample.wav' and 'sample.mp4' file.
        # 2. Have all model dependencies (BERT, Whisper, R3D_18) downloaded.
        # results = pipeline.run(mock_youtube_data)
        # import json
        # print(json.dumps(results, indent=2))
        
        print("Conceptual run demonstration:")
        print("1. Pipeline receives scraped data (e.g., for a YouTube video).")
        print("2. It calls AudioAnalyzer on the audio file -> gets transcription and anomaly scores.")
        print("3. It calls VideoAnalyzer on the video file -> gets deepfake probability.")
        print("4. It calls TextAnalyzer on the transcribed text -> gets misinformation and LLM origin scores.")
        print("5. It calls ScoreCalculator with all results -> gets final Trust and Credibility Scores.")
        print("6. It calls XAI to generate a human-readable summary.")
        print("7. It compiles and returns a final, structured JSON report.")

        print("\nSmoke test passed (conceptual).")

    except Exception as e:
        logger.error(f"An error occurred during the smoke test: {e}")