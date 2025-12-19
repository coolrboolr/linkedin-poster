import pytest

from src.config.settings import settings
from src.tools import research


def reset_tavily(monkeypatch):
    monkeypatch.setattr(research, "_tavily_client", None)


@pytest.mark.asyncio
async def test_search_web_returns_placeholder_without_key(monkeypatch):
    reset_tavily(monkeypatch)
    monkeypatch.setattr(settings, "tavily_api_key", None, raising=False)

    result = await research.search_web.ainvoke({"query": "ai safety"})
    assert "unavailable" in result.lower()


@pytest.mark.asyncio
async def test_search_web_uses_tavily_when_configured(monkeypatch):
    class DummyTavily:
        def __init__(self, *_, **__):
            pass

        def run(self, query: str) -> str:
            return f"results for {query}"

    reset_tavily(monkeypatch)
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key", raising=False)
    monkeypatch.setattr(research, "_tavily_client", DummyTavily())

    result = await research.search_web.ainvoke({"query": "retrieval"})
    assert "results for retrieval" in result


@pytest.mark.asyncio
async def test_expand_paper_context_enriches_from_arxiv(monkeypatch):
    class DummyArxiv:
        async def search_papers(self, query: str, max_results: int = 3):
            return [
                {
                    "title": "Sample Paper",
                    "summary": "A concise summary.",
                    "url": "http://example.com",
                    "published": "2024-01-01",
                    "authors": ["A. Author"],
                }
            ]

    monkeypatch.setattr(research, "ArxivService", lambda: DummyArxiv())

    result = await research.expand_paper_context.ainvoke({"title": "Sample Paper", "summary": "short"})
    assert "Sample Paper" in result
    assert "2024-01-01" in result
    assert "Authors" in result


@pytest.mark.asyncio
async def test_expand_paper_context_handles_missing_inputs():
    result = await research.expand_paper_context.ainvoke({"title": "", "summary": ""})
    assert "No paper provided" in result
