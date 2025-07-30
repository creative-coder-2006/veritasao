# analyzer/trust_credibility.py

"""
VERITAS.AI Trust and Credibility Scoring Module
===============================================

This module is responsible for calculating the final scores that quantify the
trustworthiness and credibility of the analyzed content. It synthesizes the
outputs from various analyzers (text, audio, video) into a set of
easy-to-understand metrics.

The primary scores calculated are:
- Misinformation Score: The direct probability of misinformation.
- Confidence Score: The reliability of the AI models' own predictions.
- Trust Score: A holistic measure of content trustworthiness.
- Credibility Score: An assessment of the source's and content's quality.
"""

import numpy as np
from config import SCORING_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoreCalculator:
    """
    Calculates final trust and credibility scores from analysis results.
    """
    def __init__(self, config=SCORING_CONFIG):
        """
        Initializes the ScoreCalculator with weighting configurations.
        
        Args:
            config (dict): A dictionary containing weights for score calculations.
        """
        self.config = config
        logger.info("ScoreCalculator initialized.")

    def _get_nested_value(self, data, path, default=0.0):
        """
        Safely retrieves a nested value from a dictionary.
        
        Args:
            data (dict): The dictionary to search.
            path (list): A list of keys representing the path.
            default: The value to return if the path doesn't exist.
            
        Returns:
            The retrieved value or the default.
        """
        for key in path:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        # Ensure the final value is a float, not another dict/list
        return float(data) if isinstance(data, (int, float)) else default

    def calculate_misinformation_score(self, text_results: dict) -> float:
        """
        Calculates the misinformation score based on text analysis.
        
        Args:
            text_results (dict): The output from the TextAnalyzer.
            
        Returns:
            float: The misinformation probability (0.0 to 1.0).
        """
        return self._get_nested_value(text_results, ['misinformation_detection', 'misinformation_probability'])

    def calculate_confidence_score(self, text_results: dict, video_results: dict) -> float:
        """
        Calculates an overall confidence score based on the reliability
        of the underlying AI models.
        
        Args:
            text_results (dict): The output from the TextAnalyzer.
            video_results (dict): The output from the VideoAnalyzer.
            
        Returns:
            float: The weighted average confidence of the models.
        """
        misinfo_confidence = self._get_nested_value(text_results, ['misinformation_detection', 'confidence'])
        llm_confidence = self._get_nested_value(text_results, ['llm_origin_detection', 'model_confidence'])
        deepfake_confidence = self._get_nested_value(video_results, ['deepfake_detection', 'confidence'])

        confidences = [misinfo_confidence, llm_confidence, deepfake_confidence]
        weights = self.config.get('confidence_weights', [1, 1, 1])
        
        # Filter out zero-confidence scores to avoid division by zero if no models ran
        valid_confidences = [(c, w) for c, w in zip(confidences, weights) if c > 0]
        if not valid_confidences:
            return 0.0

        total_confidence = sum(c * w for c, w in valid_confidences)
        total_weight = sum(w for c, w in valid_confidences)
        
        return total_confidence / total_weight if total_weight > 0 else 0.0

    def calculate_trust_score(self, scores: dict, text_results: dict, audio_results: dict) -> float:
        """
        Calculates the final Trust Score. This is a high-level metric that
        inversely correlates with threats.
        
        A high Trust Score means low misinformation, low deepfake probability,
        low LLM origin probability, and low audio anomalies.
        
        Args:
            scores (dict): A dictionary of already computed scores (misinformation).
            text_results (dict): The output from TextAnalyzer.
            audio_results (dict): The output from AudioAnalyzer.
            
        Returns:
            float: The calculated Trust Score (0.0 to 1.0).
        """
        weights = self.config.get('trust_weights', {})
        
        # Threat factors (higher value = less trustworthy)
        misinfo_prob = scores.get('misinformation_score', 0.0)
        llm_prob = self._get_nested_value(text_results, ['llm_origin_detection', 'llm_origin_probability'])
        anomaly_score = self._get_nested_value(audio_results, ['anomaly_detection', 'anomaly_score'], default=None)
        
        # Normalize anomaly score (assuming it's a z-score-like value, cap at a reasonable max)
        if anomaly_score is not None:
            normalized_anomaly = min(anomaly_score / self.config.get('anomaly_normalization_factor', 3.0), 1.0)
        else:
            normalized_anomaly = 0.0 # No audio, no audio threat

        threat_score = (
            misinfo_prob * weights.get('misinformation', 1) +
            llm_prob * weights.get('llm_origin', 1) +
            normalized_anomaly * weights.get('audio_anomaly', 1)
        )
        
        total_weight = sum(weights.values())
        normalized_threat = threat_score / total_weight if total_weight > 0 else 0
        
        # Trust score is the inverse of the threat score
        trust_score = 1.0 - normalized_threat
        return max(0.0, min(1.0, trust_score)) # Clamp between 0 and 1

    def calculate_credibility_score(self, scores: dict, metadata: dict) -> float:
        """
        Calculates the Credibility Score based on source reputation and content quality.
        This is a placeholder for a more complex system that might track sources over time.
        
        Args:
            scores (dict): A dictionary of already computed scores.
            metadata (dict): Platform-specific metadata (views, likes, etc.).
            
        Returns:
            float: The calculated Credibility Score (0.0 to 1.0).
        """
        # For this implementation, credibility is a simple combination of the
        # confidence score and a proxy for source reputation.
        # A real system would have a database of source reputations.
        
        confidence = scores.get('confidence_score', 0.0)
        
        # Simple proxy for reputation based on metadata (e.g., YouTube stats)
        # This is highly simplistic and platform-dependent.
        views = metadata.get('views', 0)
        likes = metadata.get('likes', 0)
        
        # A very basic engagement metric. High engagement could be good or bad.
        # Here, we just use it as a small positive factor.
        engagement_factor = np.log1p(views + likes) / np.log1p(10000000) # Normalize against 10M
        engagement_factor = min(engagement_factor, 1.0) # Cap at 1.0

        weights = self.config.get('credibility_weights', {})
        
        credibility_score = (
            confidence * weights.get('confidence', 0.7) +
            engagement_factor * weights.get('engagement', 0.3)
        )
        
        return max(0.0, min(1.0, credibility_score))


    def calculate_all_scores(self, text_results: dict, audio_results: dict, video_results: dict, metadata: dict) -> dict:
        """
        Runs all score calculations and returns a compiled dictionary.
        
        Args:
            text_results (dict): The output from the TextAnalyzer.
            audio_results (dict): The output from the AudioAnalyzer.
            video_results (dict): The output from the VideoAnalyzer.
            metadata (dict): Platform-specific metadata.
            
        Returns:
            dict: A dictionary containing all final scores.
        """
        logger.info("Calculating all final scores...")
        scores = {}

        scores['misinformation_score'] = self.calculate_misinformation_score(text_results)
        scores['confidence_score'] = self.calculate_confidence_score(text_results, video_results)
        
        # Pass intermediate scores to subsequent calculations
        scores['trust_score'] = self.calculate_trust_score(scores, text_results, audio_results)
        scores['credibility_score'] = self.calculate_credibility_score(scores, metadata)

        # Include direct probabilities from sub-modules for clarity
        scores['llm_origin_probability'] = self._get_nested_value(text_results, ['llm_origin_detection', 'llm_origin_probability'])
        scores['deepfake_probability'] = self._get_nested_value(video_results, ['deepfake_detection', 'deepfake_probability'])
        
        logger.info(f"Final scores calculated: {scores}")
        return scores


if __name__ == '__main__':
    print("Running ScoreCalculator smoke test...")
    calculator = ScoreCalculator()

    # --- Mock Analysis Results ---
    mock_text_results = {
        "misinformation_detection": {"misinformation_probability": 0.85, "confidence": 0.92},
        "llm_origin_detection": {"llm_origin_probability": 0.76, "model_confidence": 0.95}
    }
    mock_audio_results = {
        "anomaly_detection": {"anomaly_score": 2.8} # Moderately high
    }
    mock_video_results = {
        "deepfake_detection": {"deepfake_probability": 0.1, "confidence": 0.88}
    }
    mock_metadata = {
        "views": 50000, "likes": 1200
    }
    
    # --- Calculate Scores ---
    final_scores = calculator.calculate_all_scores(
        mock_text_results, mock_audio_results, mock_video_results, mock_metadata
    )

    import json
    print("\n--- Final Scores ---")
    print(json.dumps(final_scores, indent=2))
    print("\n--- Assertions ---")
    assert 0 <= final_scores['trust_score'] <= 1, "Trust score out of bounds"
    assert 0 <= final_scores['credibility_score'] <= 1, "Credibility score out of bounds"
    assert final_scores['misinformation_score'] == 0.85, "Misinformation score mismatch"
    print("All assertions passed.")
    print("\nSmoke test completed successfully.")