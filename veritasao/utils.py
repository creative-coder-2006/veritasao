# utils.py

"""
VERITAS.AI Utility Functions
============================

This module contains miscellaneous helper functions used across the
VERITAS.AI application. Centralizing these functions here prevents
code duplication and improves maintainability.
"""

from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_domain_from_url(url: str) -> str:
    """
    Extracts the network location (domain) from a given URL.

    For example, 'https://www.example.co.uk/some/path' becomes 'www.example.co.uk'.

    Args:
        url (str): The full URL to parse.

    Returns:
        str: The extracted domain, or an empty string if parsing fails.
    """
    if not isinstance(url, str) or not url:
        return ""
    try:
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        # The 'netloc' attribute correctly handles the main part of the domain,
        # including subdomains.
        return domain
    except Exception as e:
        logger.error(f"Could not parse domain from URL '{url}': {e}")
        return ""

if __name__ == '__main__':
    # This block is for direct testing of the utility functions.
    print("Running utils.py smoke test...")

    # --- Test Cases for get_domain_from_url ---
    test_urls = {
        "https://www.theverge.com/2023/10/26/some-article": "www.theverge.com",
        "https://img.youtube.com/vi/video_": "googleusercontent.com",
        "https://www.reddit.com/r/python/comments/17g4y3d/whats_new/": "www.reddit.com",
        "ftp://files.example.com/data.zip": "files.example.com",
        "https://localhost:8501": "localhost:8501",
        "invalid-url": "",
        None: "",
        "": ""
    }

    all_passed = True
    for url, expected in test_urls.items():
        result = get_domain_from_url(url)
        if result == expected:
            print(f"✓ Passed: '{url}' -> '{result}'")
        else:
            print(f"✗ Failed: '{url}' -> Expected '{expected}', Got '{result}'")
            all_passed = False

    print("\n--- Assertions ---")
    if all_passed:
        print("All assertions passed.")
    else:
        print("Some assertions failed.")

    print("\nSmoke test completed.")