"""
Web Scraper for AI Game Bot
Handles extracting content from web pages for knowledge learning
"""

import trafilatura
import logging
import requests
from typing import Optional
from urllib.parse import urlparse
import time

class WebScraper:
    """Web scraper for extracting readable content from websites"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Configure session with reasonable defaults
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Request settings
        self.timeout = 30
        self.max_retries = 3
        
        self.logger.info("Web scraper initialized")
    
    def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape content from a URL using trafilatura
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            self.logger.info(f"Scraping URL: {url}")
            
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                self.logger.error(f"Invalid URL: {url}")
                return None
            
            # Download content with retries
            downloaded = None
            for attempt in range(self.max_retries):
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        break
                    else:
                        self.logger.warning(f"Attempt {attempt + 1}: No content downloaded from {url}")
                        time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
            
            if not downloaded:
                self.logger.error(f"Failed to download content from {url}")
                return None
            
            # Extract text content
            text = trafilatura.extract(downloaded)
            
            if text:
                self.logger.info(f"Successfully extracted {len(text)} characters from {url}")
                return text
            else:
                self.logger.warning(f"No text content extracted from {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def get_website_text_content(self, url: str) -> str:
        """
        This function takes a url and returns the main text content of the website.
        The text content is extracted using trafilatura and easier to understand.
        The results is not directly readable, better to be summarized by LLM before consume
        by the user.

        Some common website to crawl information from:
        MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
        """
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text or ""
    
    def scrape_multiple_urls(self, urls: list) -> dict:
        """
        Scrape content from multiple URLs
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dictionary mapping URLs to their extracted content
        """
        results = {}
        
        for url in urls:
            try:
                content = self.scrape_url(url)
                results[url] = content
                
                # Small delay between requests to be respectful
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to scrape {url}: {e}")
                results[url] = None
        
        return results
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Quick HEAD request to check if URL is accessible
            response = self.session.head(url, timeout=10)
            return response.status_code < 400
            
        except Exception:
            return False
