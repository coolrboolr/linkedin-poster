import pytest

from src.config.settings import settings
from src.tools import research


def reset_tavily(monkeypatch):
    monkeypatch.setattr(research, "_tavily_client", None)


def test_search_web_returns_placeholder_without_key(monkeypatch):
    reset_tavily(monkeypatch)
    monkeypatch.setattr(settings, "tavily_api_key", None, raising=False)

    result = research.search_web.invoke({"query": "ai safety"})
    assert "unavailable" in result.lower()


def test_search_web_uses_tavily_when_configured(monkeypatch):
    class DummyTavily:
        def __init__(self, *_, **__):
            pass

        def run(self, query: str) -> str:
            return f"results for {query}"

    reset_tavily(monkeypatch)
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key", raising=False)
    monkeypatch.setattr(research, "_tavily_client", DummyTavily())

    result = research.search_web.invoke({"query": "retrieval"})
    assert "results for retrieval" in result


def test_expand_paper_context_enriches_from_arxiv(monkeypatch):
    class DummyArxiv:
        def search_papers(self, query: str, max_results: int = 3):
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

    result = research.expand_paper_context.invoke({"title": "Sample Paper", "summary": "short"})
    assert "Sample Paper" in result
    assert "2024-01-01" in result
    assert "Authors" in result


def test_expand_paper_context_handles_missing_inputs():
    result = research.expand_paper_context.invoke({"title": "", "summary": ""})
    assert "No paper provided" in result
