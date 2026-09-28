"""
kanha/tools/search.py
Web search tool — uses DuckDuckGo (no API key needed).

Falls back gracefully if network is unavailable.
"""

from typing import List, Dict


def web_search(query: str, max_results: int = 3) -> List[Dict]:
    """
    Searches the web using DuckDuckGo and returns snippets.

    Args:
        query       : search query string
        max_results : max number of results to return

    Returns:
        List of {"title": ..., "url": ..., "snippet": ...} dicts
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [{"title": "N/A", "url": "", "snippet": "duckduckgo_search not installed. Run: pip install duckduckgo-search"}]

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        return [{"title": "Search Error", "url": "", "snippet": str(e)}]


def format_search_results(results: List[Dict]) -> str:
    """Formats search results as a readable string for the prompt."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    {r['snippet']}\n    Source: {r['url']}")
    return "\n\n".join(lines)