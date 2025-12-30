from ddgs import DDGS


def web_search(query: str) -> str:
    """Performs a web search and returns the top 3 results summary."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No search results found."

        summary = ""
        for r in results:
            summary += f"- Title: {r['title']}\n  Snippet: {r['body']}\n"
        return summary
    except Exception as e:
        return f"Search failed: {str(e)}"
