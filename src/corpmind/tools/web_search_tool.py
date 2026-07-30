# src/corpmind/tools/web_search.py

import logging
from tavily import TavilyClient
from corpmind.config import settings

logger = logging.getLogger(__name__)

_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

def web_search(query: str, max_results: int = 2) -> list[dict]: # Max results 3 se 2 kar diya tokens bachane ke liye
    try:
        response = _client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  
        )
    except Exception as e:
        # Minimum text pass karo taaki extra token load na create ho crash logs se
        logger.warning(f"Tavily search failed for query='{query[:40]}...': {str(e)[:50]}")
        return []

    results = response.get("results", [])
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", "")[:80], # Title limit to 80 chars
            "content": r.get("content", "")[:400], # HARD TRUNCATE: Content ko 400 chars tak limit karo taaki Llama model 413 hit na kare!
        }
        for r in results
    ]