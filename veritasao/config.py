# config.py

"""
VERITAS.AI Configuration File
=============================

This file centralizes all configuration settings for the application.
It includes API keys, model names, scraper settings, and scoring weights.

Sensitive information like API keys are loaded from a `.env` file in the
root directory to keep them secure and out of version control.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- General Scraper Configuration ---
SCRAPER_CONFIG = {
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
}

# --- News Scraper Configuration ---
NEWS_SCRAPER_CONFIG = {
    "min_text_length": 150,  # Minimum characters for a valid scrape
    "timeout": 15  # Seconds to wait for a response
}

# --- Reddit Scraper Configuration ---
# Load API credentials from environment variables
REDDIT_API_CREDENTIALS = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT")
}
REDDIT_SCRAPER_CONFIG = {
    "num_top_comments": 5  # Number of top comments to fetch
}

# --- Video Scraper Configuration ---
VIDEO_SCRAPER_CONFIG = {
    "temp_video_path": "temp/video",
    "temp_audio_path": "temp/audio"
}

# --- Text Analyzer Configuration ---
TEXT_ANALYZER_CONFIG = {
    # BERT-based model for misinformation detection
    "misinfo_model_name": "distilbert-base-uncased-finetuned-sst-2-english",
    "fallback_misinfo_model": "distilbert-base-uncased",
    # Model specialized in detecting AI-generated text
    "llm_origin_model_name": "roberta-base-openai-detector",
}

# --- Audio Analyzer Configuration ---
AUDIO_ANALYZER_CONFIG = {
    "model_name": "base",  # Whisper model size (tiny, base, small, medium, large)
    "transcription_params": {
        "fp16": False  # Set to True if using CUDA and have a compatible GPU
    },
    "mfcc_params": {
        "n_mfcc": 20
    },
    "sample_rate": 16000,
    "anomaly_threshold": 2.5  # Z-score threshold for detecting anomalies
}

# --- Video Analyzer Configuration ---
VIDEO_ANALYZER_CONFIG = {
    "temp_frame_path": "temp/frames",
    "frames_per_second": 2,  # How many frames to sample per second of video
    "batch_size": 16  # Number of frames to process at once
}

# --- Scoring System Configuration ---
SCORING_CONFIG = {
    # Weights for calculating the overall Trust Score
    "trust_weights": {
        "misinformation": 0.5,
        "llm_origin": 0.25,
        "audio_anomaly": 0.25
    },
    # Weights for calculating the overall Confidence Score
    "confidence_weights": [0.4, 0.3, 0.3],  # misinfo, llm, deepfake
    # Weights for calculating the Credibility Score
    "credibility_weights": {
        "confidence": 0.8,
        "engagement": 0.2
    },
    # Factor to normalize the audio anomaly score
    "anomaly_normalization_factor": 3.0
}

# --- Explainable AI (XAI) Configuration ---
# Load OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_CONFIG = {
    "openai_model": "gpt-3.5-turbo",
    "max_tokens": 300,
    "temperature": 0.4  # Lower for more deterministic, factual explanations
}

# --- Database Configuration ---
DATABASE_CONFIG = {
    "db_file": "veritas_ai.db"
}

# --- Print a status message to confirm which APIs are configured ---
print("--- VERITAS.AI Configuration Loaded ---")
if REDDIT_API_CREDENTIALS.get("client_id"):
    print("[✓] Reddit API credentials found.")
else:
    print("[!] Reddit API credentials NOT found. Scraper will use fallback simulation.")

if OPENAI_API_KEY:
    print("[✓] OpenAI API key found.")
else:
    print("[!] OpenAI API key NOT found. XAI module will use template-based explanations.")
print("---------------------------------------")