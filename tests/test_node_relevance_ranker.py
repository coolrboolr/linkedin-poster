from src.agents.relevance_ranker import rank_papers
from src.state import AppState
from unittest.mock import MagicMock, patch
import pytest

@pytest.mark.asyncio
async def test_relevance_ranker_openai():
    candidates = [{"title": "Paper 1", "summary": "Summary 1"}, {"title": "Paper 2", "summary": "Summary 2"}]
    state = AppState(paper_candidates=candidates, trending_keywords=["AI"])
    
    with patch('src.agents.relevance_ranker.init_chat_model') as MockInitModel, \
         patch('src.agents.relevance_ranker.ChatPromptTemplate') as MockPrompt:
        
        base_llm = MockInitModel.return_value
        structured_llm = MagicMock()
        base_llm.with_structured_output.return_value = structured_llm

        class RankResult:
            index = 1
            rationale = "pick second"

        async def async_return(*args, **kwargs):
            return RankResult()

        structured_llm.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = structured_llm
        
        updates = await rank_papers(state)
        
        assert "selected_paper" in updates
        assert updates["selected_paper"]["title"] == "Paper 2"
