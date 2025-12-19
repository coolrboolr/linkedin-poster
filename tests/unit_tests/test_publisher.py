import pytest
from unittest.mock import MagicMock, patch
from src.state import AppState
from src.agents.publisher import publisher_node
from src.config.settings import Settings

@pytest.mark.asyncio
async def test_publisher_node_success():
    state = AppState(
        approved=True,
        post_draft="Great content."
    )
    
    with patch("src.agents.publisher.LinkedInService") as MockService:
        mock_instance = MockService.return_value
        mock_instance.post_update.return_value = True
        
        result = await publisher_node(state)
        
        MockService.assert_called()
        mock_instance.post_update.assert_called_with("Great content.")
        assert result == {}

@pytest.mark.asyncio
async def test_publisher_node_not_approved():
    state = AppState(
        approved=False,
        post_draft="Great content."
    )
    
    with patch("src.agents.publisher.LinkedInService") as MockService:
        result = await publisher_node(state)
        MockService.assert_not_called()
        assert result == {}

@pytest.mark.asyncio
async def test_publisher_node_no_draft():
    state = AppState(
        approved=True,
        post_draft=""
    )
    
    with patch("src.agents.publisher.LinkedInService") as MockService:
        result = await publisher_node(state)
        MockService.assert_not_called()
        assert result == {}
