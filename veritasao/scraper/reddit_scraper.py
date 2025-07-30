# scraper/reddit_scraper.py

"""
VERITAS.AI Reddit Scraper Module
================================

This module scrapes content from Reddit posts. It is designed with a
critical fallback mechanism to ensure functionality even without API keys.

1.  **Primary Method (PRAW):** When configured with Reddit API credentials,
    it uses the `praw` library to authentically fetch post data, including
    the selftext, title, score, and top comments.

2.  **Fallback Simulation:** If PRAW is not configured (i.e., no API keys
    in `config.py`), it does not fail. Instead, it returns a hard-coded,
    realistic-looking data structure. This allows the application to be
    developed, tested, and demonstrated without requiring live Reddit API
    credentials, a key feature for portability and ease of use.
"""

import praw
from config import REDDIT_SCRAPER_CONFIG, REDDIT_API_CREDENTIALS
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditScraper:
    """
    Scrapes a Reddit post using PRAW, with a built-in fallback simulation.
    """
    def __init__(self, config=REDDIT_SCRAPER_CONFIG):
        """
        Initializes the RedditScraper, setting up PRAW if credentials are available.
        """
        self.config = config
        self.reddit = None
        self.is_configured = False

        # Check if all necessary credentials are provided and not None
        if all(REDDIT_API_CREDENTIALS.get(key) for key in ['client_id', 'client_secret', 'user_agent']):
            try:
                self.reddit = praw.Reddit(**REDDIT_API_CREDENTIALS)
                # A quick check to see if credentials are valid
                self.reddit.user.me() # This will raise an exception if auth fails
                self.is_configured = True
                logger.info("PRAW initialized successfully with Reddit API credentials.")
            except Exception as e:
                logger.warning(f"PRAW initialization failed: {e}. Scraper will use fallback mode.")
                self.is_configured = False
        else:
            logger.info("Reddit API credentials not found. Scraper will use fallback simulation mode.")
            self.is_configured = False

    def _get_submission_id_from_url(self, url: str) -> str | None:
        """
        Extracts the Reddit submission ID from a URL.
        
        Example URL: https://www.reddit.com/r/python/comments/17g4y3d/whats_new_in_python_312/
        Submission ID: 17g4y3d
        """
        match = re.search(r"/comments/([a-zA-Z0-9]+)/", url)
        return match.group(1) if match else None

    def _get_fallback_data(self, url: str, submission_id: str) -> dict:
        """
        Generates realistic placeholder data for when the API is not configured.
        """
        logger.info(f"Generating fallback data for submission ID: {submission_id}")
        return {
            "platform": "reddit",
            "url": url,
            "text": (
                "This is a simulated Reddit post body used for demonstration purposes. "
                "In a live environment, with proper API keys, this text would be the actual content of the Reddit post. "
                "This fallback ensures the application can run end-to-end without credentials. "
                "The post discusses important topics and includes several key phrases designed to test the analysis pipeline. "
                "It mentions recent scientific breakthroughs and political events to trigger the misinformation detector."
            ),
            "metadata": {
                "title": "Simulated Reddit Post Title (Fallback Mode)",
                "submission_id": submission_id,
                "score": 1234,
                "upvote_ratio": 0.85,
                "num_comments": 56,
                "top_comments": [
                    "This is a simulated top comment. It agrees with the post.",
                    "This is another simulated comment that expresses some skepticism.",
                    "A third comment provides a link, www.example.com, for further reading."
                ],
                "scraping_method": "fallback_simulation"
            }
        }

    def scrape(self, url: str) -> dict:
        """
        Scrapes a Reddit post from its URL. Uses PRAW if configured,
        otherwise returns simulated data.

        Args:
            url (str): The full URL of the Reddit post.

        Returns:
            dict: A structured dictionary with the post's content and metadata.
        """
        submission_id = self._get_submission_id_from_url(url)
        if not submission_id:
            return {"error": "Could not extract submission ID from URL."}

        if not self.is_configured:
            return self._get_fallback_data(url, submission_id)

        logger.info(f"Scraping submission ID {submission_id} using PRAW.")
        try:
            submission = self.reddit.submission(id=submission_id)
            
            # Eagerly load comments
            submission.comments.replace_more(limit=0)
            
            # Get top comments
            top_comments = [
                comment.body for comment in submission.comments[:self.config.get("num_top_comments", 3)]
            ]

            # Combine title and selftext for a full text analysis
            full_text = f"{submission.title}\n\n{submission.selftext}"

            return {
                "platform": "reddit",
                "url": url,
                "text": full_text,
                "metadata": {
                    "title": submission.title,
                    "submission_id": submission.id,
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "top_comments": top_comments,
                    "scraping_method": "praw_api"
                }
            }
        except Exception as e:
            logger.error(f"Failed to scrape submission {submission_id} with PRAW: {e}")
            return {"error": f"PRAW API request failed: {e}"}

if __name__ == '__main__':
    print("Running RedditScraper smoke test...")
    # This URL is just for parsing, the actual content depends on whether API keys are set
    example_url = "https://www.reddit.com/r/technology/comments/17f0u4b/meta_is_reportedly_planning_to_charge_14_a_month/"
    
    scraper = RedditScraper()
    scraped_content = scraper.scrape(example_url)

    import json
    print(f"\n--- Scraped Content (Mode: {'PRAW API' if scraper.is_configured else 'Fallback Simulation'}) ---")
    print(json.dumps(scraped_content, indent=2))

    print("\n--- Assertions ---")
    assert "error" not in scraped_content, "Scraping resulted in an error."
    assert scraped_content["platform"] == "reddit", "Platform is not 'reddit'."
    assert len(scraped_content["text"]) > 0, "Scraped text is empty."
    assert "submission_id" in scraped_content["metadata"], "Submission ID is missing."
    print("All assertions passed.")
    print("\nSmoke test completed successfully.")