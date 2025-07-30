# analyzer/__init__.py

"""
VERITAS.AI Analyzer Package
===========================

This package contains all the core AI and ML analysis modules for the VERITAS.AI application.
It provides functionalities for:
- Text analysis (misinformation, LLM origin)
- Video analysis (deepfake detection)
- Audio analysis (transcription, anomaly detection)
- Trust and credibility scoring
- Explainable AI (XAI) integration

The modules are designed to be used together in a comprehensive analysis pipeline.
"""

from .audio_analysis import AudioAnalyzer
from .text_analysis import TextAnalyzer
from .video_analysis import VideoAnalyzer
from .trust_credibility import ScoreCalculator
from .xai_explanations import XAI
from .pipeline import AnalysisPipeline

__all__ = [
    "AudioAnalyzer",
    "TextAnalyzer",
    "VideoAnalyzer",
    "ScoreCalculator",
    "XAI",
    "AnalysisPipeline"
]

print("VERITAS.AI Analyzer package initialized.")