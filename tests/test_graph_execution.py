import pytest
from langsmith.run_helpers import traceable
from src.graph import graph
from src.state import AppState
from unittest.mock import patch, MagicMock, AsyncMock

@traceable
@pytest.mark.asyncio
async def test_full_graph_execution():
    """
    Tests the full graph execution end-to-end with mocks.
    """
    # Mock all external services to avoid API calls and ensure deterministic flow
    with patch('src.agents.trend_scanner.GoogleTrendsService') as MockTrends, \
         patch('src.agents.arxiv_fetcher.ArxivService') as MockArxiv, \
         patch('src.agents.relevance_ranker.init_chat_model') as MockRankerModel, \
         patch('src.agents.relevance_ranker.ChatPromptTemplate') as MockRankPrompt, \
         patch('src.agents.conversation_agent.init_chat_model') as MockConvModel, \
         patch('src.agents.post_writer.init_chat_model') as MockWriterModel, \
         patch('src.agents.human_approval.interrupt', return_value={"type": "accept", "args": "Looks good"}) as mock_approval_interrupt, \
         patch('src.agents.conversation_agent.interrupt', return_value={"type": "accept", "args": None}) as mock_conv_interrupt, \
         patch('src.agents.human_paper_review.interrupt', return_value={"type": "accept", "args": None}) as mock_review_interrupt, \
         patch('src.agents.memory_loader.MemoryStore') as MockStoreLoader, \
         patch('src.agents.memory_updater.MemoryStore') as MockStoreUpdater:
         
        # Setup Mocks
        MockTrends.return_value.get_trending_topics = AsyncMock(return_value=["AI"])
        MockArxiv.return_value.search_papers = AsyncMock(return_value=[{"title": "Test Paper", "summary": "Summary"}])

        # Memory stores (keep everything in-memory)
        mock_loader_store = MockStoreLoader.return_value
        # If load() is called, it should be awaitable
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
        
        # Ranker LLM returns index 0 via structured output
        base_ranker_llm = MockRankerModel.return_value
        structured_ranker_llm = MagicMock()
        base_ranker_llm.with_structured_output.return_value = structured_ranker_llm

        class RankResult:
            index = 0
            rationale = "pick first"

        structured_ranker_llm.ainvoke = AsyncMock(return_value=RankResult())
        structured_ranker_llm.invoke = MagicMock(return_value=RankResult())
        # chain = prompt | structured_llm
        mock_rank_prompt = MockRankPrompt.from_template.return_value
        mock_rank_prompt.__or__.return_value = structured_ranker_llm
        
        # Conversation LLM returns question
        mock_conv_instance = MagicMock()
        mock_conv_response = MagicMock()
        mock_conv_response.content = "Any specific focus?"
        mock_conv_instance.ainvoke = AsyncMock(return_value=mock_conv_response)
        mock_conv_instance.invoke.return_value = mock_conv_response
        mock_conv_instance.return_value = mock_conv_response
        MockConvModel.return_value = mock_conv_instance

        # Writer LLM returns draft
        mock_writer_instance = MagicMock()
        mock_writer_response = MagicMock()
        mock_writer_response.content = "Draft Post"
        mock_writer_instance.ainvoke = AsyncMock(return_value=mock_writer_response)
        mock_writer_instance.invoke.return_value = mock_writer_response
        mock_writer_instance.return_value = mock_writer_response
        MockWriterModel.return_value = mock_writer_instance

        state = AppState()
        
        # Run the graph
        # We need to pass a config for the checkpointer
        config = {"configurable": {"thread_id": "test_thread"}}
        final_state = await graph.ainvoke(state, config=config)
        
        assert final_state["approved"] is True
        assert final_state["post_draft"] == "Draft Post"
        assert "memory" in final_state  # Verify memory loaded
        mem = final_state["memory"]
        assert "topic_preferences" in mem and "post_format_preferences" in mem
        assert final_state["next_step"] == "memory_updater"  # Verify execution router worked

@pytest.mark.asyncio
async def test_execution_router_revise():
    from src.graph import execution_router
    
    # Case: User asks to revise
    state = AppState(
        post_draft="Draft",
        approved=False,
        human_feedback="Please revise this.",
        revision_requested=True
    )
    
    result = await execution_router(state)
    assert result["next_step"] == "post_writer"
    
    # Case: User says no (not revise)
    state = AppState(
        post_draft="Draft",
        approved=False,
        human_feedback="No, I don't like it."
    )
    
    result = await execution_router(state)
    assert result["next_step"] == "human_approval"

@pytest.mark.asyncio
async def test_full_graph_revise_flow():
    """
    Tests the revision flow: Draft v1 -> Revise -> Draft v2 -> Approve.
    """
    with patch('src.agents.trend_scanner.GoogleTrendsService') as MockTrends, \
         patch('src.agents.arxiv_fetcher.ArxivService') as MockArxiv, \
         patch('src.agents.relevance_ranker.init_chat_model') as MockRankerModel, \
         patch('src.agents.relevance_ranker.ChatPromptTemplate') as MockRankPrompt, \
         patch('src.agents.conversation_agent.init_chat_model') as MockConvModel, \
         patch('src.agents.post_writer.init_chat_model') as MockWriterModel, \
         patch('src.agents.human_approval.interrupt') as mock_approval_interrupt, \
         patch('src.agents.conversation_agent.interrupt', return_value={"type": "accept", "args": None}) as mock_conv_interrupt, \
         patch('src.agents.human_paper_review.interrupt', return_value={"type": "accept", "args": None}) as mock_review_interrupt, \
         patch('src.agents.memory_loader.MemoryStore') as MockStoreLoader, \
         patch('src.agents.memory_updater.MemoryStore') as MockStoreUpdater:
         
        # Setup Mocks
        MockTrends.return_value.get_trending_topics = AsyncMock(return_value=["AI"])
        MockArxiv.return_value.search_papers = AsyncMock(return_value=[{"title": "Test Paper", "summary": "Summary"}])
        
        # Memory
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
        
        # Ranker
        base_ranker_llm = MockRankerModel.return_value
        structured_ranker_llm = MagicMock()
        base_ranker_llm.with_structured_output.return_value = structured_ranker_llm

        class RankResult:
            index = 0
            rationale = "v1"

        structured_ranker_llm.ainvoke = AsyncMock(return_value=RankResult())
        mock_rank_prompt = MockRankPrompt.from_template.return_value
        mock_rank_prompt.__or__.return_value = structured_ranker_llm
        
        # Conversation
        mock_conv_instance = MagicMock()
        mock_conv_response = MagicMock()
        mock_conv_response.content = "Any specific focus?"
        mock_conv_instance.ainvoke = AsyncMock(return_value=mock_conv_response)
        mock_conv_instance.return_value = mock_conv_response # Handle __call__
        MockConvModel.return_value = mock_conv_instance
        
        # Writer - returns v1 then v2
        mock_writer_instance = MagicMock()
        mock_writer_response_v1 = MagicMock()
        mock_writer_response_v1.content = "Draft v1"
        mock_writer_response_v2 = MagicMock()
        mock_writer_response_v2.content = "Draft v2"
        
        # Side effect for ainvoke to return v1 then v2
        mock_writer_instance.ainvoke = AsyncMock(side_effect=[mock_writer_response_v1, mock_writer_response_v2])
        mock_writer_instance.side_effect = [mock_writer_response_v1, mock_writer_response_v2] # Handle __call__
        MockWriterModel.return_value = mock_writer_instance
        
        # Approval interrupt - first request edit (revision), then accept
        mock_approval_interrupt.side_effect = [
            {"type": "edit", "args": {"draft": "Draft v1"}},
            {"type": "accept", "args": "yes"},
        ]

        state = AppState()
        config = {"configurable": {"thread_id": "test_thread_revise"}}
        
        # Mocks return immediately, so ainvoke reaches terminal state in one run.
        final_state = await graph.ainvoke(state, config=config)
        
        assert final_state["approved"] is True
        assert final_state["post_draft"] == "Draft v2"
        assert final_state["revision_requested"] is False
        assert final_state["human_feedback"] == "yes"


@pytest.mark.asyncio
async def test_graph_exit_requested_short_circuits():
    state = AppState(exit_requested=True)
    config = {"configurable": {"thread_id": "exit-thread"}}

    final_state = await graph.ainvoke(state, config=config)

    assert final_state["next_step"] == "exit"
    assert not final_state.get("trending_keywords")
