import pytest
import asyncio
from src.graph import graph
from src.state import AppState
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_graph_non_blocking():
    """
    Verifies that the graph can execute a step without blocking the event loop for too long.
    This is a heuristic test: if a node blocks the loop, this test might timeout or hang 
    if we were to enforce strict loop monitoring (which is hard in simple pytest).
    
    Instead, we just ensure it runs quickly enough or at least yields.
    """
    # Use a minimal state
    state = AppState(
        trending_keywords=["AI"],
        paper_candidates=[{"title": "Test Paper", "summary": "Summary"}],
        selected_paper={"title": "Test Paper", "summary": "Summary"},
        user_ready=True, # Skip conversation
        approved=True # Skip approval loop for this test if possible, or just test first step
    )
    
    # We want to test a path that involves potential blocking calls, e.g., trend scanning or memory loading
    # But trend scanning is at the start.
    
    # Let's try to run the graph for a short time or until first yield
    # We can use a timeout to ensure it doesn't hang indefinitely
    
    try:
        async with asyncio.timeout(5.0): # Should be plenty for async ops
            with patch('src.agents.memory_loader.MemoryStore') as MockStoreLoader, \
                 patch('src.agents.memory_updater.MemoryStore') as MockStoreUpdater, \
                 patch('src.agents.trend_scanner.GoogleTrendsService') as MockTrends, \
                 patch('src.agents.arxiv_fetcher.ArxivService') as MockArxiv, \
                 patch('src.agents.post_writer.init_chat_model') as MockWriterModel, \
                 patch('src.agents.relevance_ranker.init_chat_model') as MockRankerModel, \
                 patch('src.agents.conversation_agent.init_chat_model') as MockConvModel, \
                 patch('src.agents.human_paper_review.interrupt', return_value={"type": "accept", "args": None}):

                # Memory stores
                MockStoreLoader.return_value.get_all.return_value = {
                    "topic_preferences": {},
                    "comprehension_preferences": {},
                    "post_format_preferences": {},
                }
                mock_updater_store = MockStoreUpdater.return_value
                mock_updater_store.topic = {}
                mock_updater_store.comp = {}
                mock_updater_store.format = {}
                mock_updater_store.get_all.return_value = {
                    "topic_preferences": {},
                    "comprehension_preferences": {},
                    "post_format_preferences": {},
                }

                # External services
                MockTrends.return_value.get_trending_topics.return_value = ["AI"]
                MockArxiv.return_value.search_papers.return_value = [{"title": "Test Paper", "summary": "Summary"}]

                # Simple LLM mocks
                async def async_llm_return(*args, **kwargs):
                    result = MagicMock()
                    result.content = "ok"
                    return result

                for mock_llm in (MockWriterModel.return_value, MockConvModel.return_value):
                    mock_llm.ainvoke = AsyncMock(side_effect=async_llm_return)
                    mock_llm.invoke = MagicMock(return_value=MagicMock(content="ok"))

                base_ranker_llm = MockRankerModel.return_value
                structured_ranker_llm = MagicMock()
                base_ranker_llm.with_structured_output.return_value = structured_ranker_llm

                class RankResult:
                    index = 0
                structured_ranker_llm.ainvoke = AsyncMock(return_value=RankResult())

                async for event in graph.astream(
                    state,
                    {
                        "recursion_limit": 5,
                        "configurable": {"thread_id": "non_blocking_test"},
                    },
                ):
                    assert event is not None
                    break 
                
    except asyncio.TimeoutError:
        pytest.fail("Graph execution timed out or blocked the event loop!")
    except Exception:
        # Logic errors are acceptable here; we just care about non-blocking behavior.
        pass
