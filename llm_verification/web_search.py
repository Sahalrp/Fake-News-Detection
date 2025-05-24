"""
Web search module for retrieving information from the internet.
"""
import requests
from typing import List, Dict, Any, Optional, Union
import time
import re
import json
import random
from bs4 import BeautifulSoup
import urllib.parse
import hashlib
from datetime import datetime, timedelta
from .logger import logger, log_request, log_response, log_exception, log_data

# Simple in-memory cache for search results
SEARCH_CACHE = {}
CACHE_EXPIRY = timedelta(minutes=15)  # Cache results for 15 minutes to ensure fresh information

class WebSearch:
    """Class for performing web searches and retrieving content."""

    def __init__(self, user_agent: str = None):
        """
        Initialize the web search client.

        Args:
            user_agent: User agent string to use for requests. If None, a random one is selected.
        """
        if user_agent is None:
            # List of common user agents for better web scraping
            user_agents = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
            ]
            self.user_agent = random.choice(user_agents)
        else:
            self.user_agent = user_agent

        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        # Initialize cache
        self.cache_hits = 0
        self.cache_misses = 0

    def search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for information.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        logger.info(f"Searching DuckDuckGo for: {query}")

        # Validate and sanitize query
        if not query or not query.strip():
            logger.error("Empty search query provided")
            return []

        query = query.strip()

        # Check if this is a time-sensitive query (elections, current events, etc.)
        time_sensitive_keywords = [
            "election", "president", "vote", "elected", "won", "winner", "2024", "2023",
            "current", "latest", "recent", "today", "yesterday", "this week",
            "this month", "this year", "breaking news", "inauguration", "appointed",
            "announced", "launched", "released", "published", "unveiled", "discovered",
            "prime minister", "chancellor", "governor", "mayor", "senator", "congress",
            "parliament", "supreme court", "cabinet", "administration", "government",
            "war", "conflict", "crisis", "disaster", "pandemic", "outbreak", "emergency"
        ]

        # People who might be subjects of time-sensitive queries
        notable_people = [
            "biden", "trump", "harris", "obama", "putin", "zelensky", "netanyahu",
            "pope", "francis", "king charles", "queen camilla", "prince william",
            "elon musk", "bezos", "zuckerberg", "gates"
        ]

        # Combine all keywords
        all_time_sensitive = time_sensitive_keywords + notable_people

        # Check if query contains any time-sensitive keywords
        is_time_sensitive = any(keyword in query.lower() for keyword in all_time_sensitive)

        # Always treat queries about people in positions of power as time-sensitive
        position_keywords = ["president", "prime minister", "chancellor", "ceo", "director", "secretary", "minister"]
        if any(position in query.lower() for position in position_keywords):
            is_time_sensitive = True

        logger.debug(f"Query is time-sensitive: {is_time_sensitive}")

        # Add recency terms to time-sensitive queries
        if is_time_sensitive and "current" not in query.lower() and "latest" not in query.lower() and "2024" not in query:
            query = f"{query} current 2024"
            logger.debug(f"Enhanced time-sensitive query: {query}")

        # URL encode the query
        encoded_query = urllib.parse.quote_plus(query)
        logger.debug(f"Encoded query: {encoded_query}")

        # DuckDuckGo search URL - add time parameters for time-sensitive queries
        if is_time_sensitive:
            # Add date filter for recent results (d=w means past week)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}&df=w"
            logger.debug(f"Using time-sensitive search URL: {url}")
        else:
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            logger.debug(f"Search URL: {url}")

        try:
            # Log the request
            log_request(url, "GET", headers=self.headers)

            # Make the request
            response = requests.get(url, headers=self.headers, timeout=30)

            # Log the response
            log_response(response)

            # Check for errors
            response.raise_for_status()

            # Log response size
            logger.debug(f"Response size: {len(response.text)} bytes")

            # Parse the HTML response
            logger.debug("Parsing HTML response")
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Extract search results
            result_elements = soup.select('.result')
            logger.debug(f"Found {len(result_elements)} raw result elements")

            for result in result_elements:
                title_elem = result.select_one('.result__title')
                url_elem = result.select_one('.result__url')
                snippet_elem = result.select_one('.result__snippet')

                if title_elem and url_elem:
                    title = title_elem.get_text(strip=True)
                    result_url = url_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    # Sanitize the results
                    title = title.replace('\n', ' ').strip()
                    result_url = result_url.replace('\n', ' ').strip()
                    snippet = snippet.replace('\n', ' ').strip()

                    result_item = {
                        "title": title,
                        "url": result_url,
                        "snippet": snippet
                    }

                    results.append(result_item)
                    logger.debug(f"Added result: {title[:50]}...")

                if len(results) >= num_results:
                    logger.debug(f"Reached maximum number of results ({num_results})")
                    break

            logger.info(f"Found {len(results)} search results for query: {query}")
            return results

        except requests.RequestException as e:
            log_exception(e, "search_duckduckgo - HTTP request")
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []
        except Exception as e:
            log_exception(e, "search_duckduckgo - Unexpected error")
            logger.error(f"Unexpected error in search_duckduckgo: {e}")
            return []

    def fetch_article_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract the main content from a web page.

        Args:
            url: URL of the web page to fetch

        Returns:
            Extracted text content or None if failed
        """
        logger.info(f"Fetching content from URL: {url}")

        # Validate URL
        if not url or not url.strip():
            logger.error("Empty URL provided")
            return None

        url = url.strip()

        # Check if URL has a scheme, add https:// if missing
        if not url.startswith(('http://', 'https://')):
            logger.debug(f"Adding https:// to URL: {url}")
            url = f"https://{url}"

        # Validate URL format
        try:
            # Parse the URL to check if it's valid
            parsed_url = urllib.parse.urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.error(f"Invalid URL format: {url}")
                return None
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return None

        # Check cache first
        cache_key = f"content_{hashlib.md5(url.encode()).hexdigest()}"
        cached_content = self._get_from_cache(cache_key)
        if cached_content:
            logger.info(f"Using cached content for URL: {url}")
            return cached_content

        try:
            # Log the request
            log_request(url, "GET", headers=self.headers)

            # Make the request with a shorter timeout and minimal delay
            time.sleep(0.1)  # Minimal delay to avoid rate limiting
            logger.debug(f"Sending GET request to {url}")
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10,  # Reduced timeout
                allow_redirects=True
            )

            # Log the response
            log_response(response)

            # Check for errors
            response.raise_for_status()

            # Log response size
            logger.debug(f"Response size: {len(response.text)} bytes")

            # Parse the HTML
            logger.debug("Parsing HTML content")
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.extract()

            # Try to find the main content
            main_content = None

            # Look for common content containers
            content_candidates = []

            # Try to find article or main content elements
            for tag in ["article", "main", "div.content", "div.article", "div.post", ".post-content", ".article-content"]:
                elements = soup.select(tag)
                if elements:
                    content_candidates.extend(elements)

            # If we found potential content containers, use the largest one
            if content_candidates:
                # Sort by content length
                content_candidates.sort(key=lambda x: len(x.get_text()), reverse=True)
                main_content = content_candidates[0].get_text(separator=' ', strip=True)

            # If we couldn't find a main content container, use the whole page
            if not main_content or len(main_content) < 100:
                # Remove very small text nodes (likely navigation, buttons, etc.)
                for element in soup.find_all(text=True):
                    if len(element.strip()) < 20:
                        element.extract()

                # Get text from the whole page
                main_content = soup.get_text(separator=' ', strip=True)

            # Clean up the text
            if main_content:
                # Replace multiple spaces with a single space
                main_content = re.sub(r'\s+', ' ', main_content)

                # Replace multiple newlines with a single newline
                main_content = re.sub(r'\n+', '\n', main_content)

                # Limit to 5000 characters to get more context while still being manageable
                main_content = main_content[:5000]

                logger.debug(f"Extracted {len(main_content)} characters of content")

                # Log a sample of the content
                if len(main_content) > 200:
                    logger.debug(f"Content sample: {main_content[:200]}...")
                else:
                    logger.debug(f"Content: {main_content}")

                # Cache the content
                self._add_to_cache(cache_key, main_content)

                return main_content
            else:
                logger.warning(f"No content extracted from {url}")
                return None

        except requests.RequestException as e:
            log_exception(e, f"fetch_article_content - {url}")
            logger.error(f"Error fetching content from {url}: {e}")
            return None
        except Exception as e:
            log_exception(e, f"fetch_article_content - Unexpected error - {url}")
            logger.error(f"Unexpected error fetching content from {url}: {e}")
            return None

    def search_google(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search Google for information.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        logger.info(f"Searching Google for: {query}")

        # Validate and sanitize query
        if not query or not query.strip():
            logger.error("Empty search query provided")
            return []

        query = query.strip()

        # Check if this is a time-sensitive query (elections, current events, etc.)
        time_sensitive_keywords = [
            "election", "president", "vote", "elected", "won", "winner", "2024", "2023",
            "current", "latest", "recent", "today", "yesterday", "this week",
            "this month", "this year", "breaking news", "inauguration", "appointed",
            "announced", "launched", "released", "published", "unveiled", "discovered",
            "prime minister", "chancellor", "governor", "mayor", "senator", "congress",
            "parliament", "supreme court", "cabinet", "administration", "government",
            "war", "conflict", "crisis", "disaster", "pandemic", "outbreak", "emergency"
        ]

        # People who might be subjects of time-sensitive queries
        notable_people = [
            "biden", "trump", "harris", "obama", "putin", "zelensky", "netanyahu",
            "pope", "francis", "king charles", "queen camilla", "prince william",
            "elon musk", "bezos", "zuckerberg", "gates"
        ]

        # Combine all keywords
        all_time_sensitive = time_sensitive_keywords + notable_people

        # Check if query contains any time-sensitive keywords
        is_time_sensitive = any(keyword in query.lower() for keyword in all_time_sensitive)

        # Always treat queries about people in positions of power as time-sensitive
        position_keywords = ["president", "prime minister", "chancellor", "ceo", "director", "secretary", "minister"]
        if any(position in query.lower() for position in position_keywords):
            is_time_sensitive = True

        logger.debug(f"Query is time-sensitive: {is_time_sensitive}")

        # Add recency terms to time-sensitive queries
        if is_time_sensitive and "current" not in query.lower() and "latest" not in query.lower() and "2024" not in query:
            query = f"{query} current 2024"
            logger.debug(f"Enhanced time-sensitive query: {query}")

        # URL encode the query
        encoded_query = urllib.parse.quote_plus(query)
        logger.debug(f"Encoded query: {encoded_query}")

        # Google search URL - add time parameters for time-sensitive queries
        if is_time_sensitive:
            # Add parameters to prioritize recent results - use last week for more recent results
            url = f"https://www.google.com/search?q={encoded_query}&num={num_results}&tbs=qdr:w"  # Last week
            logger.debug(f"Using time-sensitive search URL: {url}")
        else:
            url = f"https://www.google.com/search?q={encoded_query}&num={num_results}"
            logger.debug(f"Search URL: {url}")

        try:
            # Check cache first
            cache_key = f"google_{hashlib.md5(query.encode()).hexdigest()}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached Google search results for: {query}")
                return cached_result

            # Log the request
            log_request(url, "GET", headers=self.headers)

            # Make the request with minimal delay to avoid rate limiting
            time.sleep(0.1)
            response = requests.get(url, headers=self.headers, timeout=10)

            # Log the response
            log_response(response)

            # Check for errors
            response.raise_for_status()

            # Log response size
            logger.debug(f"Response size: {len(response.text)} bytes")

            # Parse the HTML response
            logger.debug("Parsing HTML response")
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Extract search results
            # Google search results are in divs with class 'g'
            result_elements = soup.select('div.g')
            logger.debug(f"Found {len(result_elements)} raw result elements")

            for result in result_elements:
                # Extract title and URL
                title_elem = result.select_one('h3')
                link_elem = result.select_one('a')
                snippet_elem = result.select_one('div.VwiC3b')

                if title_elem and link_elem and 'href' in link_elem.attrs:
                    title = title_elem.get_text(strip=True)
                    url = link_elem['href']

                    # Make sure URL is absolute
                    if url.startswith('/url?'):
                        url_parts = urllib.parse.urlparse(url)
                        query_parts = urllib.parse.parse_qs(url_parts.query)
                        if 'q' in query_parts:
                            url = query_parts['q'][0]

                    # Extract snippet
                    snippet = ""
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)

                    # Sanitize the results
                    title = title.replace('\n', ' ').strip()
                    url = url.replace('\n', ' ').strip()
                    snippet = snippet.replace('\n', ' ').strip()

                    # Skip if URL is not valid
                    if not url.startswith(('http://', 'https://')):
                        continue

                    result_item = {
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    }

                    results.append(result_item)
                    logger.debug(f"Added result: {title[:50]}...")

                if len(results) >= num_results:
                    logger.debug(f"Reached maximum number of results ({num_results})")
                    break

            logger.info(f"Found {len(results)} Google search results for query: {query}")

            # Cache the results
            self._add_to_cache(cache_key, results)

            return results

        except requests.RequestException as e:
            log_exception(e, "search_google - HTTP request")
            logger.error(f"Error searching Google: {e}")
            return []
        except Exception as e:
            log_exception(e, "search_google - Unexpected error")
            logger.error(f"Unexpected error in search_google: {e}")
            return []

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and is not expired."""
        global SEARCH_CACHE

        if key in SEARCH_CACHE:
            entry = SEARCH_CACHE[key]
            if datetime.now() - entry["timestamp"] < CACHE_EXPIRY:
                self.cache_hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry["data"]
            else:
                # Expired entry
                logger.debug(f"Cache entry expired for key: {key}")
                del SEARCH_CACHE[key]

        self.cache_misses += 1
        logger.debug(f"Cache miss for key: {key}")
        return None

    def _add_to_cache(self, key: str, data: Any) -> None:
        """Add a value to the cache."""
        global SEARCH_CACHE

        SEARCH_CACHE[key] = {
            "data": data,
            "timestamp": datetime.now()
        }
        logger.debug(f"Added to cache: {key}")

        # Clean up old entries if cache is getting too large
        if len(SEARCH_CACHE) > 100:
            self._clean_cache()

    def _clean_cache(self) -> None:
        """Remove expired entries from the cache."""
        global SEARCH_CACHE

        now = datetime.now()
        expired_keys = [
            k for k, v in SEARCH_CACHE.items()
            if now - v["timestamp"] > CACHE_EXPIRY
        ]

        for key in expired_keys:
            del SEARCH_CACHE[key]

        logger.debug(f"Cleaned {len(expired_keys)} expired entries from cache")

    def _extract_key_terms(self, query: str) -> str:
        """
        Extract key terms from a query to create a more general search query.

        Args:
            query: The original search query

        Returns:
            A more general search query with just the key terms
        """
        # Remove common words and punctuation
        stop_words = [
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "who", "which", "that", "this", "these", "those",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "of", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "s", "t", "just", "don", "now", "said"
        ]

        # Clean the query
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)  # Replace punctuation with spaces
        words = query.split()

        # Keep only non-stop words and words longer than 2 characters
        key_words = [word for word in words if word not in stop_words and len(word) > 2]

        # If we have too few words, keep some of the original words
        if len(key_words) < 2 and len(words) > 0:
            # Keep the longest words from the original query
            words.sort(key=len, reverse=True)
            key_words = words[:2]

        # Join the key words
        general_query = " ".join(key_words)

        logger.debug(f"Extracted key terms: '{general_query}' from original query: '{query}'")
        return general_query

    def search_and_summarize(self, query: str, num_results: int = 3) -> str:
        """
        Search for information and summarize the results.

        Args:
            query: Search query
            num_results: Number of results to process (default reduced to 3 for speed)

        Returns:
            Summarized information from search results
        """
        logger.info(f"Searching and summarizing for query: {query}")

        try:
            # Validate and sanitize the query
            if not query or not query.strip():
                logger.error("Empty search query provided")
                return "No valid search query provided."

            query = query.strip()
            logger.debug(f"Sanitized query: {query}")

            # Check cache first
            cache_key = f"summary_{hashlib.md5(query.encode()).hexdigest()}"
            cached_summary = self._get_from_cache(cache_key)
            if cached_summary:
                logger.info(f"Using cached summary for: {query}")
                return cached_summary

            # Use only one search engine for faster results
            all_results = []

            # Try DuckDuckGo as it's usually faster
            logger.debug(f"Searching DuckDuckGo for up to {num_results} results")
            duck_results = self.search_duckduckgo(query, num_results)
            if duck_results:
                all_results.extend(duck_results)
                logger.info(f"Found {len(duck_results)} results from DuckDuckGo")
            else:
                # Fall back to Google only if DuckDuckGo fails
                logger.warning("No results from DuckDuckGo, trying Google")
                google_results = self.search_google(query, num_results)
                if google_results:
                    all_results.extend(google_results)
                    logger.info(f"Found {len(google_results)} results from Google")
                else:
                    logger.warning("No results from Google either")

            # If we still have no results, try a more general search
            if not all_results:
                logger.warning(f"No search results found for specific query: {query}")
                # Try a more general search by extracting key terms
                general_query = self._extract_key_terms(query)
                if general_query != query:
                    logger.info(f"Trying more general search with: {general_query}")
                    general_results = self.search_duckduckgo(general_query, num_results)
                    if general_results:
                        all_results.extend(general_results)
                        logger.info(f"Found {len(general_results)} results from general search")

                if not all_results:
                    return "No search results found for the query. Please verify if this information exists or try a different search term."

            logger.debug(f"Processing {len(all_results)} search results")

            # Build the summary
            summary = f"Search results for: {query}\n\n"

            for i, result in enumerate(all_results, 1):
                logger.debug(f"Processing result {i}/{len(all_results)}")

                # Ensure all fields exist and are properly sanitized
                title = result.get('title', 'No title').replace('\n', ' ').strip()
                url = result.get('url', 'No URL').replace('\n', ' ').strip()
                snippet = result.get('snippet', 'No snippet').replace('\n', ' ').strip()

                logger.debug(f"Result {i} - Title: {title[:50]}...")

                # Add basic result information
                summary += f"{i}. {title}\n"
                summary += f"   URL: {url}\n"
                summary += f"   Summary: {snippet}\n\n"

                # Try to fetch additional content for all results to get better context
                try:
                    logger.debug(f"Fetching additional content for URL: {url}")
                    content = self.fetch_article_content(url)

                    if content:
                        # Sanitize content and add a more substantial excerpt
                        content = content.replace('\n', ' ').strip()
                        # Use longer excerpts for better context
                        excerpt_length = 1000 if i <= 2 else 500  # Longer excerpts for top results
                        excerpt = content[:excerpt_length] + "..." if len(content) > excerpt_length else content
                        summary += f"   Content excerpt: {excerpt}\n\n"
                        logger.debug(f"Added content excerpt for result {i}")
                    else:
                        logger.warning(f"No content fetched for URL: {url}")
                        summary += f"   Content excerpt: Could not fetch content.\n\n"

                except Exception as e:
                    # Log the error but continue with other results
                    log_exception(e, f"search_and_summarize - content fetch for {url}")
                    logger.error(f"Error fetching content for {url}: {e}")
                    summary += f"   Content excerpt: Could not fetch content.\n\n"

                # Stop after processing 5 results to keep the summary manageable
                if i >= 5:
                    break

            # Add a note about the number of results found
            if len(all_results) > 5:
                summary += f"\nNote: Found {len(all_results)} results in total, showing the top 5.\n"

            # If we have very few results, add a note
            if len(all_results) <= 2:
                summary += "\nNote: Limited search results found. This may indicate either that the information is not widely available or that the search terms need refinement.\n"

            # Log the final summary length
            logger.debug(f"Generated summary with {len(summary)} characters")

            # Log a sample of the summary
            if len(summary) > 500:
                logger.debug(f"Summary sample: {summary[:500]}...")
            else:
                logger.debug(f"Summary: {summary}")

            # Cache the summary
            self._add_to_cache(cache_key, summary)

            return summary

        except Exception as e:
            # Log the error and return a user-friendly message
            log_exception(e, "search_and_summarize")
            logger.error(f"Error in search_and_summarize: {e}")

            # Return a structured error message
            error_msg = f"Error performing search: {str(e)}"
            logger.debug(f"Returning error message: {error_msg}")
            return error_msg
