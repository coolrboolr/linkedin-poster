from src.agents.arxiv_fetcher import fetch_arxiv_papers
from src.state import AppState
from unittest.mock import patch
import pytest

@pytest.mark.asyncio
async def test_arxiv_fetcher_basic():
    with patch('src.services.arxiv_client.ArxivService.search_papers') as mock_search:
        mock_search.return_value = [{"title": "Test Paper"}]
        
        state = AppState(trending_keywords=["AI"])
        updates = await fetch_arxiv_papers(state)
        
        assert "paper_candidates" in updates
        assert len(updates["paper_candidates"]) == 1
        assert updates["paper_candidates"][0]["title"] == "Test Paper"
        
        # Verify query quoting
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert 'all:"AI"' in args[0]
