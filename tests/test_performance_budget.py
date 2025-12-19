import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph import graph
from src.state import AppState


@pytest.mark.asyncio
async def test_graph_completes_under_two_seconds():
    state = AppState()
    config = {"configurable": {"thread_id": "perf-budget"}}

    with patch("src.agents.trend_scanner.GoogleTrendsService") as MockTrends, \
         patch("src.agents.arxiv_fetcher.ArxivService") as MockArxiv, \
         patch("src.agents.relevance_ranker.init_chat_model") as MockRankerModel, \
         patch("src.agents.relevance_ranker.ChatPromptTemplate") as MockRankPrompt, \
         patch("src.agents.conversation_agent._invoke_with_tools", new=AsyncMock(return_value=("content", [], "question"))), \
         patch("src.agents.conversation_agent._invoke_legacy", new=AsyncMock(return_value=("content", [], "question"))), \
         patch("src.agents.conversation_agent.init_chat_model") as MockConvModel, \
         patch("src.agents.post_writer.init_chat_model") as MockWriterModel, \
         patch("src.agents.post_writer.ChatPromptTemplate") as MockPWPrompt, \
         patch("src.agents.human_approval.interrupt", return_value={"type": "accept", "args": "Looks good"}), \
         patch("src.agents.conversation_agent.interrupt", return_value={"type": "accept", "args": None}), \
         patch("src.agents.human_paper_review.interrupt", return_value={"type": "accept", "args": None}), \
         patch("src.agents.memory_loader.MemoryStore") as MockStoreLoader, \
         patch("src.agents.memory_updater.MemoryStore") as MockStoreUpdater:

        MockTrends.return_value.get_trending_topics = AsyncMock(return_value=["AI"])
        MockArxiv.return_value.search_papers = AsyncMock(return_value=[{"title": "Test Paper", "summary": "Summary"}])

        mock_loader_store = MockStoreLoader.return_value
        mock_loader_store.load = AsyncMock(return_value=None)
        mock_loader_store.get_all.return_value = {
            "topic_preferences": {"seeds": ["AI"], "avoid": []},
            "comprehension_preferences": {},
            "post_format_preferences": {},
        }
        mock_updater_store = MockStoreUpdater.return_value
        mock_updater_store.load = AsyncMock(return_value=None)
        mock_updater_store.save = AsyncMock(return_value=True)
        mock_updater_store.topic = {}
        mock_updater_store.comp = {}
        mock_updater_store.format = {}
        mock_updater_store.get_all.return_value = {
            "topic_preferences": {},
            "comprehension_preferences": {},
            "post_format_preferences": {},
        }

        base_ranker_llm = MockRankerModel.return_value
        structured_ranker_llm = MagicMock()
        base_ranker_llm.with_structured_output.return_value = structured_ranker_llm

        class RankResult:
            index = 0
            rationale = "pick first"

        structured_ranker_llm.ainvoke = AsyncMock(return_value=RankResult())
        structured_ranker_llm.invoke = MagicMock(return_value=RankResult())
        mock_rank_prompt = MockRankPrompt.from_template.return_value
        mock_rank_prompt.__or__.return_value = structured_ranker_llm

        class ConvResp:
            content = "Ready."

        class ConvLLM:
            def __init__(self):
                self.ainvoke = AsyncMock(return_value=ConvResp())
        MockConvModel.return_value = ConvLLM()

        class WriterResp:
            content = "Draft Post"

        class WriterLLM:
            def __init__(self):
                self.ainvoke = AsyncMock(return_value=WriterResp())
        MockWriterModel.return_value = WriterLLM()

        mock_pw_prompt = MockPWPrompt.from_template.return_value
        runnable = MagicMock()

        async def pw_async_return(inputs, *_, **__):
            return WriterResp()

        runnable.ainvoke.side_effect = pw_async_return
        mock_pw_prompt.__or__.return_value = runnable

        async with asyncio.timeout(2):
            final_state = await graph.ainvoke(state, config=config)

        assert final_state["post_draft"] == "Draft Post"
