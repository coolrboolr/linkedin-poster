from src.agents.trend_scanner import scan_trending_topics
from src.state import AppState
from unittest.mock import patch, AsyncMock
import pytest

@pytest.mark.asyncio
async def test_trend_scanner_basic():
    with patch('src.services.google_trends.GoogleTrendsService.get_trending_topics', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = ["AI", "ML"]
        
        state = AppState()
        updates = await scan_trending_topics(state)
        
        assert "trending_keywords" in updates
        assert updates["trending_keywords"] == ["ai", "ml"]
        
        # Verify service was called with seeds
        mock_get.assert_called_once()
