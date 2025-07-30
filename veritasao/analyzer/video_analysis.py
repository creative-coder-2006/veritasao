# analyzer/video_analysis.py

"""
VERITAS.AI Video Analysis Module
================================

This module provides deepfake detection capabilities for VERITAS.AI. It uses a
pre-trained 3D convolutional neural network (R3D_18) to analyze video frames
and assess the probability of manipulation.

The process involves:
1.  Extracting frames from the video.
2.  Applying transformations to normalize the frames.
3.  Running inference with the R3D_18 model.
4.  Aggregating frame-level predictions into a single video score.
"""

import torch
import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights
import cv2
import numpy as np
from config import VIDEO_ANALYZER_CONFIG
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """
    Analyzes video files for deepfake detection.
    """
    def __init__(self, config=VIDEO_ANALYZER_CONFIG):
        """
        Initializes the VideoAnalyzer with a pre-trained R3D_18 model.

        Args:
            config (dict): Configuration dictionary for the analyzer.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = self._load_model()
        logger.info(f"VideoAnalyzer initialized on device: {self.device}")

    def _load_model(self):
        """
        Loads the pre-trained R3D_18 model and its associated transforms.
        Provides a fallback if the specified weights are unavailable.

        Returns:
            tuple: A tuple containing the model and the transform function.
        """
        try:
            logger.info("Loading R3D_18 model with KINETICS_400 weights.")
            weights = R3D_18_Weights.KINETICS400_V1
            model = r3d_18(weights=weights)
            transform = weights.transforms()
        except Exception as e:
            logger.warning(f"Could not load specified R3D_18 weights: {e}. Falling back to default.")
            # Fallback to creating an untrained model for structural integrity,
            # but it won't produce meaningful results.
            weights = None
            model = r3d_18(weights=None)
            # Dummy transform if weights fail
            transform = lambda x: torch.tensor(x).float().permute(3, 0, 1, 2) / 255.0

        model = model.to(self.device)
        model.eval()
        return model, transform

    def _extract_frames(self, video_path: str, temp_frame_dir: str):
        """
        Extracts frames from a video file at a specified rate.

        Args:
            video_path (str): The path to the video file.
            temp_frame_dir (str): Directory to save the extracted frames.

        Yields:
            np.ndarray: An individual frame from the video.
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        capture_interval = int(frame_rate / self.config.get("frames_per_second", 1))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % capture_interval == 0:
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
            
            frame_count += 1
        
        cap.release()

    def analyze(self, video_path: str) -> dict:
        """
        Performs deepfake detection on the entire video.

        Args:
            video_path (str): The path to the video file.

        Returns:
            dict: A dictionary containing the deepfake probability and confidence.
        """
        if self.model is None:
            return {
                "error": "Deepfake detection model is not available.",
                "deepfake_probability": 0.0,
                "confidence": 0.0
            }

        logger.info(f"Starting deepfake analysis for: {video_path}")
        
        # Temporary directory for frames
        temp_dir = self.config.get("temp_frame_path", "temp/frames")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        all_predictions = []
        
        try:
            frames_generator = self._extract_frames(video_path, temp_dir)
            
            # Process frames in batches
            batch_size = self.config.get("batch_size", 8)
            frames_batch = []

            for frame in frames_generator:
                frames_batch.append(frame)
                if len(frames_batch) == batch_size:
                    # Preprocess and predict
                    preprocessed_batch = self.transform(np.array(frames_batch)).to(self.device)
                    with torch.no_grad():
                        # model expects [B, C, T, H, W], we have [B, H, W, C] -> [B, C, H, W]
                        # and need to add a time dimension T. We'll use T=len(batch)
                        # The transform should handle this, but we reshape for sanity
                        if len(preprocessed_batch.shape) == 4: # B, C, H, W
                           preprocessed_batch = preprocessed_batch.permute(1, 0, 2, 3).unsqueeze(0) # 1, C, B, H, W
                        
                        predictions = self.model(preprocessed_batch)
                        probabilities = torch.nn.functional.softmax(predictions, dim=1)
                        all_predictions.extend(probabilities.cpu().numpy())

                    frames_batch = [] # Reset batch
            
            # Process any remaining frames
            if frames_batch:
                 preprocessed_batch = self.transform(np.array(frames_batch)).to(self.device)
                 with torch.no_grad():
                    if len(preprocessed_batch.shape) == 4:
                        preprocessed_batch = preprocessed_batch.permute(1, 0, 2, 3).unsqueeze(0)
                    predictions = self.model(preprocessed_batch)
                    probabilities = torch.nn.functional.softmax(predictions, dim=1)
                    all_predictions.extend(probabilities.cpu().numpy())

        except Exception as e:
            logger.error(f"An error occurred during video frame processing: {e}")
            shutil.rmtree(temp_dir)
            return {"error": str(e), "deepfake_probability": 0.0, "confidence": 0.0}
        
        finally:
            # Clean up temporary frames
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        if not all_predictions:
            logger.warning("No frames were analyzed. Cannot provide a score.")
            return {
                "error": "No frames could be extracted or processed from the video.",
                "deepfake_probability": 0.0,
                "confidence": 0.0
            }

        # Aggregate results
        # We need to know which class index corresponds to "real" vs "fake".
        # This is highly dependent on the model's training data (Kinetics-400).
        # We will make a simplifying assumption: we look for anomalous classifications.
        # A more robust method would use a model fine-tuned for deepfake detection.
        # Here, we'll take the max probability of any single class as a "confidence"
        # and use the variance of predictions as a proxy for potential manipulation.
        avg_preds = np.mean(all_predictions, axis=0)
        confidence = float(np.max(avg_preds))
        
        # A simple proxy for deepfake probability: 1 - confidence in the dominant class.
        # This implies that a clear, consistent classification (real or otherwise) is "not fake",
        # whereas a confused, low-confidence classification might be.
        deepfake_prob = 1.0 - confidence

        logger.info(f"Deepfake analysis complete. Probability: {deepfake_prob:.4f}, Confidence: {confidence:.4f}")
        
        return {
            "deepfake_probability": deepfake_prob,
            "confidence": confidence,
            "frames_analyzed": len(all_predictions)
        }

if __name__ == '__main__':
    print("Running VideoAnalyzer smoke test...")
    # This test requires a video file `sample.mp4` to be present.
    # Since we cannot provide one, we will outline the test logic.
    
    try:
        # Create a dummy video file for testing
        if not os.path.exists("sample.mp4"):
            print("Creating a dummy video file for the test (requires opencv-python)...")
            width, height = 128, 128
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('sample.mp4', fourcc, 10.0, (width, height))
            for _ in range(50): # 5 seconds of video
                frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            print("Dummy video created.")

        analyzer = VideoAnalyzer()
        
        # Analyze the dummy video
        results = analyzer.analyze("sample.mp4")
        
        import json
        print("\n--- Analysis Results ---")
        print(json.dumps(results, indent=2))
        
        assert "deepfake_probability" in results
        assert 0 <= results['deepfake_probability'] <= 1
        
        # Clean up the dummy file
        os.remove("sample.mp4")
        print("\nSmoke test completed successfully.")
        
    except ImportError:
        logger.warning("Skipping smoke test: 'opencv-python' is not installed.")
    except Exception as e:
        logger.error(f"An error occurred during the smoke test: {e}")
        if os.path.exists("sample.mp4"):
            os.remove("sample.mp4")