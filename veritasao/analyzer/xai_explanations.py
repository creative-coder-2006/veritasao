# analyzer/xai_explanations.py

"""
VERITAS.AI Explainable AI (XAI) Module
======================================

This module leverages a large language model (LLM) to generate human-readable
explanations for the analysis results produced by the VERITAS.AI pipeline.
It translates complex numerical scores into a coherent, qualitative summary.

The core functionality includes:
- Using platform-specific "master prompts" to guide the LLM.
- Synthesizing scores and metadata into a clear narrative.
- Providing a fallback mechanism if the primary LLM API is unavailable.
"""

import openai
from config import XAI_CONFIG, OPENAI_API_KEY
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- OpenAI API Configuration ---
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    IS_OPENAI_CONFIGURED = True
    logger.info("OpenAI API key found and configured.")
else:
    IS_OPENAI_CONFIGURED = False
    logger.warning("OpenAI API key not found. XAI will use a template-based fallback.")


class XAI:
    """
    Generates human-readable explanations of the analysis results.
    """

    def __init__(self, config=XAI_CONFIG):
        """
        Initializes the XAI module with configuration.

        Args:
            config (dict): Configuration dictionary for XAI.
        """
        self.config = config
        self.master_prompts = self._load_master_prompts()

    def _load_master_prompts(self) -> dict:
        """
        Defines the expert master prompts for different platforms.
        These prompts instruct the LLM on how to behave during analysis.
        """
        return {
            "news": """
You are a senior news analyst and fact-checker. Your task is to provide a concise, balanced, and evidence-based explanation of the provided analysis scores for a news article. Focus on the likelihood of misinformation and the credibility of the text. Avoid definitive statements and use cautious, probabilistic language. Explain what the 'Misinformation Score' and 'LLM Origin Probability' mean in this context.
""",
            "reddit": """
You are a social media intelligence analyst specializing in online communities. Your task is to interpret the analysis scores for a Reddit post. Explain the results in a clear, accessible way. Consider the context of Reddit (anonymity, community norms) when explaining the 'Trust Score' and 'LLM Origin Probability'. Focus on potential red flags for manipulation or artificial content.
""",
            "youtube": """
You are a digital media forensics expert. Your task is to explain the analysis results for a YouTube video. You must synthesize information from the video's content (deepfake score), audio (anomaly score), and transcribed text (misinformation score). Explain each key score (Trust, Deepfake, Misinformation) and how they combine to give an overall picture of the video's authenticity.
""",
            "default": """
You are an AI analysis expert. Your task is to provide a clear and concise summary of a content analysis report. Explain what each score means in simple terms. The scores you need to interpret are: Trust Score, Credibility Score, Misinformation Score, and LLM Origin Probability. Be objective and straightforward.
"""
        }

    def _generate_fallback_explanation(self, scores: dict, platform: str) -> dict:
        """
        Generates a simple, template-based explanation if the LLM API is not available.

        Args:
            scores (dict): The dictionary of calculated scores.
            platform (str): The content platform (e.g., 'news', 'reddit').

        Returns:
            dict: A dictionary containing the fallback explanation.
        """
        logger.info("Generating fallback explanation (OpenAI API not configured).")
        explanation = (
            f"**Analysis Summary (Template-Based)**\n\n"
            f"This content from **{platform.capitalize()}** has been analyzed with the following results:\n\n"
            f"- **Trust Score**: {scores.get('trust_score', 0):.2f}/1.00\n"
            f"  - This score reflects the overall trustworthiness. A lower score indicates the presence of risk factors like potential misinformation or signs of artificial generation.\n"
            f"- **Misinformation Probability**: {scores.get('misinformation_score', 0):.0%}\n"
            f"  - Our text analysis model estimates this probability that the content contains misleading or false information.\n"
            f"- **AI-Generated Text Probability**: {scores.get('llm_origin_probability', 0):.0%}\n"
            f"  - This is the likelihood that the text was written by a large language model.\n"
        )
        if platform == "youtube":
            explanation += (
                f"- **Deepfake Video Probability**: {scores.get('deepfake_probability', 0):.0%}\n"
                f"  - This score reflects the chance that the video has been manipulated using deepfake technology.\n"
            )

        return {
            "summary": explanation,
            "details": "This is a basic summary because the advanced Explainable AI service is not configured.",
            "status": "fallback"
        }

    def explain_results(self, scores: dict, platform: str) -> dict:
        """
        Generates a detailed, AI-powered explanation of the analysis results.

        Args:
            scores (dict): The dictionary of calculated scores.
            platform (str): The content platform ('news', 'reddit', 'youtube').

        Returns:
            dict: A dictionary containing the detailed explanation.
        """
        if not IS_OPENAI_CONFIGURED:
            return self._generate_fallback_explanation(scores, platform)

        logger.info(f"Generating OpenAI-powered explanation for {platform} content.")
        system_prompt = self.master_prompts.get(platform, self.master_prompts["default"])

        # Sanitize scores for the prompt, ensuring they are readable
        scores_for_prompt = {k: f"{v:.2%}" if "prob" in k.lower() or "score" in k.lower() else v for k, v in scores.items()}

        user_prompt = f"""
Here is the analysis report. Please provide your expert explanation based on the instructions.

**Analysis Report:**
- Platform: {platform.capitalize()}
- Scores: {json.dumps(scores_for_prompt, indent=2)}

Please generate a concise summary explaining these results to a non-expert.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.config.get("openai_model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.config.get("max_tokens", 250),
                temperature=self.config.get("temperature", 0.5),
            )
            explanation = response.choices[0].message['content'].strip()
            return {
                "summary": explanation,
                "model_used": self.config.get("openai_model", "gpt-3.5-turbo"),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "error": "Failed to generate AI explanation.",
                "details": str(e),
                "status": "error"
            }

if __name__ == '__main__':
    print("Running XAI smoke test...")
    xai_module = XAI()

    # --- Mock Scores ---
    mock_scores = {
        'misinformation_score': 0.78,
        'confidence_score': 0.91,
        'trust_score': 0.25,
        'credibility_score': 0.40,
        'llm_origin_probability': 0.65,
        'deepfake_probability': 0.15
    }

    # --- Test Case 1: YouTube Platform ---
    print("\n--- Testing Explanation for YouTube ---")
    youtube_explanation = xai_module.explain_results(mock_scores, "youtube")
    print("Status:", youtube_explanation.get('status'))
    print("Explanation:\n", youtube_explanation.get('summary', youtube_explanation.get('error')))

    # --- Test Case 2: News Platform ---
    print("\n--- Testing Explanation for News ---")
    news_explanation = xai_module.explain_results(mock_scores, "news")
    print("Status:", news_explanation.get('status'))
    print("Explanation:\n", news_explanation.get('summary', news_explanation.get('error')))

    # --- Test Case 3: Fallback (if no API key) ---
    if not IS_OPENAI_CONFIGURED:
        print("\n--- Testing Fallback Explanation ---")
        fallback_expl = xai_module._generate_fallback_explanation(mock_scores, "reddit")
        print("Status:", fallback_expl.get('status'))
        print("Explanation:\n", fallback_expl.get('summary'))

    print("\nSmoke test completed.")