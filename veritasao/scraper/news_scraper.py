# scraper/news_scraper.py

"""
VERITAS.AI News Scraper Module
==============================

This module is responsible for scraping news articles from various online
sources. It employs a two-tier strategy for robustness:

1.  **Primary Method (newspaper3k):** Utilizes the advanced `newspaper3k`
    library to intelligently extract the main article content, authors,
    publication date, and other metadata.

2.  **Fallback Method (requests + BeautifulSoup):** If the primary method
    fails or returns insufficient content, it falls back to a simpler, more
    direct scraping approach using `requests` to fetch the HTML and
    `BeautifulSoup` to parse it. This ensures that some content is almost
    always retrieved.
"""

import newspaper
import requests
from bs4 import BeautifulSoup
from config import SCRAPER_CONFIG, NEWS_SCRAPER_CONFIG
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsScraper:
    """
    Scrapes news articles using newspaper3k with a requests/BS4 fallback.
    """
    def __init__(self, config=NEWS_SCRAPER_CONFIG):
        """
        Initializes the NewsScraper.

        Args:
            config (dict): Configuration dictionary for the news scraper.
        """
        self.config = config
        self.headers = SCRAPER_CONFIG.get('headers', {})

    def _is_valid_url(self, url: str) -> bool:
        """
        Checks if the URL is well-formed.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _scrape_with_newspaper(self, url: str) -> dict:
        """
        Primary scraping method using the newspaper3k library.

        Args:
            url (str): The URL of the news article.

        Returns:
            dict: A dictionary containing the scraped data, or an empty dict on failure.
        """
        logger.info(f"Attempting to scrape {url} with newspaper3k...")
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()

            if len(article.text) < self.config.get("min_text_length", 100):
                raise ValueError("Extracted text is too short. Potential paywall or parsing error.")

            return {
                "text": article.text,
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "method": "newspaper3k"
            }
        except Exception as e:
            logger.warning(f"newspaper3k failed for {url}: {e}")
            return {}

    def _scrape_with_fallback(self, url: str) -> dict:
        """
        Fallback scraping method using requests and BeautifulSoup.
        This is less precise but more robust against anti-scraping measures.

        Args:
            url (str): The URL of the news article.

        Returns:
            dict: A dictionary containing the scraped data, or an empty dict on failure.
        """
        logger.info(f"Falling back to requests+BeautifulSoup for {url}...")
        try:
            response = requests.get(url, headers=self.headers, timeout=self.config.get("timeout", 10))
            response.raise_for_status() # Will raise an HTTPError for bad responses

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No Title Found'

            # Extract text by joining all paragraph tags
            paragraphs = soup.find_all('p')
            text = '\n'.join([p.get_text() for p in paragraphs])

            if len(text) < self.config.get("min_text_length", 100):
                 raise ValueError("Fallback extracted text is too short.")

            return {
                "text": text,
                "title": title,
                "authors": [], # Fallback doesn't reliably get authors
                "publish_date": None, # Fallback doesn't reliably get date
                "method": "fallback_bs4"
            }
        except Exception as e:
            logger.error(f"Fallback scraping failed for {url}: {e}")
            return {}

    def scrape(self, url: str) -> dict:
        """
        Scrapes a news article from a given URL.

        Args:
            url (str): The URL of the article to scrape.

        Returns:
            dict: A structured dictionary with the scraped content and metadata.
        """
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL provided: {url}")
            return {"error": "Invalid URL format."}

        # Try primary method first
        data = self._scrape_with_newspaper(url)

        # If primary method fails or returns empty text, try fallback
        if not data or not data.get("text"):
            logger.warning(f"Primary method failed for {url}, trying fallback.")
            data = self._scrape_with_fallback(url)

        # If both fail, return an error
        if not data or not data.get("text"):
            return {"error": f"Failed to scrape content from {url} using all methods."}

        # Structure the final output
        return {
            "platform": "news",
            "url": url,
            "text": data.get("text"),
            "metadata": {
                "title": data.get("title"),
                "authors": data.get("authors"),
                "publish_date": data.get("publish_date"),
                "scraping_method": data.get("method")
            }
        }

if __name__ == '__main__':
    print("Running NewsScraper smoke test...")
    # Example URL (use a reliable, generally accessible news source)
    example_url = "https://www.theverge.com/2023/10/26/23933428/google-search-generative-ai-images-midjourney-dall-e-3-feature"
    
    scraper = NewsScraper()
    scraped_content = scraper.scrape(example_url)

    import json
    print("\n--- Scraped Content ---")
    print(json.dumps(scraped_content, indent=2))

    print("\n--- Assertions ---")
    assert "error" not in scraped_content, "Scraping resulted in an error."
    assert scraped_content["platform"] == "news", "Platform is not 'news'."
    assert len(scraped_content["text"]) > 100, "Scraped text is too short."
    assert scraped_content["metadata"]["title"] is not None, "Title was not scraped."
    print("All assertions passed.")
    print("\nSmoke test completed successfully.")