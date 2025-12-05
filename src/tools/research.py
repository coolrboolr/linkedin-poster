from typing import Optional

from langchain_core.tools import tool

from src.config.settings import settings
from src.services.arxiv_client import ArxivService
from src.services.logger import get_logger

logger = get_logger(__name__)

_tavily_client: Optional[object] = None


def _build_tavily_client() -> Optional[object]:
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set; search_web will return a placeholder response.")
        return None

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
    except ModuleNotFoundError:
        logger.error("langchain_community not installed; search_web will return a placeholder.")
        return None

    return TavilySearchResults(
        api_key=settings.tavily_api_key,
        max_results=5,
        search_depth="basic",
        include_answer=True,
    )


def _get_tavily_client() -> Optional[object]:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = _build_tavily_client()
    return _tavily_client


@tool
def search_web(query: str) -> str:
    """Search the web for recent context, examples, and definitions related to the query."""
    client = _get_tavily_client()
    if client is None:
        return "Web search unavailable: missing TAVILY_API_KEY."

    try:
        logger.info(f"[conversation tools] web search: {query}")
        return client.run(query)
    except Exception as exc:
        logger.error(f"Tavily search failed: {exc}")
        return "Web search unavailable right now."


@tool
def expand_paper_context(title: str, summary: str) -> str:
    """Expand a paper's context using ArXiv metadata and the provided summary."""
    if not title and not summary:
        return "No paper provided to expand."

    service = ArxivService()
    try:
        query = title or summary[:80]
        papers = service.search_papers(query=query, max_results=3)
    except Exception as exc:
        logger.error(f"ArXiv expansion failed: {exc}")
        papers = []

    if not papers:
        base_title = title or "Unknown title"
        base_summary = summary or "Summary unavailable."
        return f"Title: {base_title}\nSummary: {base_summary}"

    paper = papers[0]
    lines = [
        f"Title: {paper.get('title', title)}",
        f"Summary: {paper.get('summary', summary)}",
    ]
    if paper.get("authors"):
        lines.append(f"Authors: {', '.join(paper['authors'])}")
    if paper.get("published"):
        lines.append(f"Published: {paper['published']}")
    if paper.get("url"):
        lines.append(f"URL: {paper['url']}")

    return "\n".join(lines)
