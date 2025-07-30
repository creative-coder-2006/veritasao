# scraper/__init__.py

"""
VERITAS.AI Scraper Package
==========================

This package is responsible for all content retrieval from external platforms.
It includes specialized modules for scraping:
- News articles from various websites.
- Posts and comments from Reddit.
- Video and audio content from YouTube.

Each scraper is designed to be robust, with fallbacks for when primary
methods fail. The goal is to extract text, media, and relevant metadata
for the analysis pipeline.
"""

from .news_scraper import NewsScraper
from .reddit_scraper import RedditScraper
from .video_scraper import VideoScraper

__all__ = [
    "NewsScraper",
    "RedditScraper",
    "VideoScraper"
]

print("VERITAS.AI Scraper package initialized.")