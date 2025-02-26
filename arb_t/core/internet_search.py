import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from urllib.parse import quote_plus
import logging
import time

@dataclass
class SearchResult:
    """Represents a search result with its content and metadata."""
    url: str
    title: str
    content: str
    relevance_score: float
    timestamp: float

class InternetSearchEngine:
    """Handles internet search and content retrieval."""
    
    def __init__(self, concurrent_requests: int = 5, rate_limit: float = 1.0):
        self.concurrent_requests = concurrent_requests
        self.rate_limit = rate_limit  # Minimum time between requests
        self.last_request_time = 0.0
        self.session: Optional[aiohttp.ClientSession] = None
        self.default_search_patterns = {
            "definition": "what is {topic} definition explanation",
            "examples": "{topic} examples code implementation",
            "tutorial": "how to {topic} tutorial guide",
            "best_practices": "{topic} best practices tips",
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _wait_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
            
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform an internet search using multiple search engines."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        await self._wait_rate_limit()

        # Run GitHub and Stack Overflow searches
        tasks = [
            self._search_github(query, max_results),
            self._search_stackoverflow(query, max_results)
        ]
        
        # Run searches concurrently with error handling
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results = []
            
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Search error: {str(result)}")
                    continue
                all_results.extend(result)
                
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []
                
        # Combine and deduplicate results
        seen_urls = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
                
            if len(unique_results) >= max_results:
                break
                
        return unique_results
            
    async def _search_github(self, query: str, max_results: int) -> List[SearchResult]:
        """Search GitHub repositories and discussions."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        await self._wait_rate_limit()
            
        # Clean and format query for GitHub
        clean_query = ' '.join(query.split()[:5])  # Take first 5 terms to avoid too long queries
        url = f"https://api.github.com/search/repositories?q={quote_plus(clean_query)}+language:python+sort:stars"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ARB-T-Agent"
        }
        
        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("items", [])[:max_results]:
                        results.append(
                            SearchResult(
                                url=item["html_url"],
                                title=item["full_name"],
                                content=item["description"] or "",
                                relevance_score=item["stargazers_count"] / 1000,  # Use stars as relevance
                                timestamp=0  # GitHub API provides timestamps but we'll skip for simplicity
                            )
                        )
                    return results
                else:
                    logging.error(f"GitHub API error: {response.status}")
                    return []
        except asyncio.TimeoutError:
            logging.error("GitHub API request timed out")
            return []
        except Exception as e:
            logging.error(f"Error searching GitHub: {str(e)}")
            return []
            
    async def _search_stackoverflow(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Stack Overflow for relevant discussions."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        await self._wait_rate_limit()
            
        url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=votes&intitle={quote_plus(query)}&site=stackoverflow"
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("items", [])[:max_results]:
                        results.append(
                            SearchResult(
                                url=item["link"],
                                title=item["title"],
                                content=item.get("body", "")[:5000],  # Limit content length
                                relevance_score=item["score"] / 100,  # Use SO score as relevance
                                timestamp=item["creation_date"]
                            )
                        )
                    return results
                else:
                    logging.error(f"Stack Overflow API error: {response.status}")
                    return []
        except asyncio.TimeoutError:
            logging.error("Stack Overflow API request timed out")
            return []
        except Exception as e:
            logging.error(f"Error searching Stack Overflow: {str(e)}")
            return []
            
    def generate_search_query(self, topic: str, context: Optional[Dict] = None) -> str:
        """Generate an appropriate search query based on context or defaults."""
        # Clean and normalize the topic
        topic = ' '.join(topic.split()[:10])  # Limit topic length
        
        if not context:
            # Use first two most relevant patterns
            patterns = list(self.default_search_patterns.values())[:2]
            queries = [pattern.format(topic=topic) for pattern in patterns]
            return " OR ".join(queries)
            
        # Use context to generate focused query
        return f"{topic} {context.get('compartment', '')} python"
        
    async def fetch_content(self, url: str, max_chars: int = 10000) -> str:
        """Fetch and parse content from a URL with size limit."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        await self._wait_rate_limit()
            
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove irrelevant elements
                    for element in soup(['script', 'style', 'head', 'header', 'footer', 'nav']):
                        element.decompose()
                        
                    # Extract text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Limit content length
                    return text[:max_chars]
                else:
                    logging.error(f"Error fetching content, status: {response.status}")
                    return ""
        except asyncio.TimeoutError:
            logging.error(f"Request timed out for URL: {url}")
            return ""
        except Exception as e:
            logging.error(f"Error fetching content from {url}: {str(e)}")
            return ""
