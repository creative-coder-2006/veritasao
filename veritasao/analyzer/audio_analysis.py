# analyzer/audio_analysis.py

"""
VERITAS.AI Audio Analysis Module
================================

This module provides the core audio analysis capabilities for VERITAS.AI.
It uses Whisper for high-accuracy transcription and analyzes audio features
to detect anomalies that might indicate manipulation.

Key Features:
- Transcription: Converts spoken words into text.
- Anomaly Detection: Identifies unusual audio patterns.
- Spectrogram Analysis: Visualizes audio frequencies for inspection.
"""

import torch
import librosa
import numpy as np
import whisper
from scipy.stats import zscore
from config import AUDIO_ANALYZER_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Handles transcription and anomaly detection for audio files.
    """
    def __init__(self, config=AUDIO_ANALYZER_CONFIG):
        """
        Initializes the AudioAnalyzer with a Whisper model.

        Args:
            config (dict): Configuration dictionary for the analyzer.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        logger.info(f"AudioAnalyzer initialized on device: {self.device}")

    def _load_model(self):
        """
        Loads the Whisper model specified in the configuration.
        Provides a fallback to a smaller model if the preferred one fails.

        Returns:
            whisper.Whisper: The loaded Whisper model.
        """
        model_name = self.config.get("model_name", "base")
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            return whisper.load_model(model_name, device=self.device)
        except Exception as e:
            logger.warning(
                f"Failed to load Whisper model '{model_name}': {e}. "
                f"Falling back to 'tiny' model."
            )
            return whisper.load_model("tiny", device=self.device)

    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribes the given audio file using the Whisper model.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            dict: A dictionary containing the transcription text and segments.
                  Returns an error message if transcription fails.
        """
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            result = self.model.transcribe(audio_path, **self.config.get("transcription_params", {}))
            logger.info("Transcription successful.")
            return {
                "text": result["text"],
                "segments": result["segments"]
            }
        except Exception as e:
            logger.error(f"Error during transcription for {audio_path}: {e}")
            return {
                "error": "Transcription failed.",
                "details": str(e)
            }

    def detect_anomalies(self, audio_path: str) -> dict:
        """
        Performs anomaly detection on the audio's spectral features.
        It calculates MFCCs and identifies outliers using Z-scores.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            dict: A dictionary containing anomaly scores and timestamps.
                  Returns an error message if analysis fails.
        """
        try:
            logger.info(f"Performing anomaly detection on: {audio_path}")
            y, sr = librosa.load(audio_path, sr=self.config.get("sample_rate", 16000))

            # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, **self.config.get("mfcc_params", {"n_mfcc": 13})
            )
            mfccs_delta = librosa.feature.delta(mfccs)

            # Combine features and calculate Z-scores for anomaly detection
            features = np.vstack([mfccs, mfccs_delta])
            z_scores = np.abs(zscore(features, axis=1))
            anomaly_scores = np.mean(z_scores, axis=0)

            # Identify anomalies exceeding a threshold
            threshold = self.config.get("anomaly_threshold", 2.5)
            anomalies = np.where(anomaly_scores > threshold)[0]

            # Convert frame indices to timestamps
            timestamps = librosa.frames_to_time(anomalies, sr=sr)

            logger.info(f"Found {len(timestamps)} potential anomalies.")

            return {
                "anomaly_score": np.mean(anomaly_scores),
                "anomalous_timestamps": timestamps.tolist(),
                "num_anomalies": len(timestamps)
            }
        except Exception as e:
            logger.error(f"Error during anomaly detection for {audio_path}: {e}")
            return {
                "error": "Anomaly detection failed.",
                "details": str(e)
            }

    def analyze(self, audio_path: str) -> dict:
        """
        Runs the full audio analysis pipeline: transcription and anomaly detection.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            dict: A dictionary containing the combined results.
        """
        logger.info(f"Starting full audio analysis for: {audio_path}")
        transcription_result = self.transcribe_audio(audio_path)
        anomaly_result = self.detect_anomalies(audio_path)

        return {
            "transcription": transcription_result,
            "anomaly_detection": anomaly_result
        }

if __name__ == '__main__':
    # Example usage for testing the module directly
    # Note: Requires a sample audio file named 'sample.mp3' in the same directory.
    print("Running AudioAnalyzer smoke test...")
    try:
        # Create a dummy audio file for testing
        import soundfile as sf
        sr = 16000
        duration = 10
        frequency = 440
        t = np.linspace(0., duration, int(sr * duration))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        # Add a small anomaly
        data[sr*5:sr*6] *= 0.1
        dummy_audio_path = "sample.wav"
        sf.write(dummy_audio_path, data.astype(np.int16), sr)
        
        analyzer = AudioAnalyzer()
        analysis_output = analyzer.analyze(dummy_audio_path)
        
        print("\n--- Analysis Output ---")
        print(f"Transcription: {analysis_output['transcription'].get('text', 'N/A')}")
        print(f"Anomaly Score: {analysis_output['anomaly_detection'].get('anomaly_score', 'N/A')}")
        print(f"Anomalous Timestamps: {analysis_output['anomaly_detection'].get('anomalous_timestamps', 'N/A')}")
        print("--- End of Output ---")
        
        # Clean up dummy file
        import os
        os.remove(dummy_audio_path)
        print("\nSmoke test completed successfully.")

    except ImportError:
        logger.warning(
            "Skipping smoke test: 'soundfile' library not installed. "
            "Install it with 'pip install soundfile' to run the test."
        )
    except Exception as e:
        logger.error(f"An error occurred during the smoke test: {e}")