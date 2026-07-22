import logging

from tavily import TavilyClient

from corpmind.config import settings

logger = logging.getLogger(__name__)

_client = TavilyClient(api_key=settings.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 3) -> list[dict]:
    try:
        response = _client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  
        )
    except Exception as e:
        logger.warning(f"Tavily search failed for query='{query}': {e}")
        return []

    results = response.get("results", [])
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]