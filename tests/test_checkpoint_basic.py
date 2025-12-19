import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph import graph
from src.state import AppState

pytestmark = pytest.mark.anyio


async def test_checkpoint_runs():
    config = {"configurable": {"thread_id": "checkpoint-test"}}
    async for _ in graph.astream(AppState(), config=config):
        pass  # we only care that it does not throw


@pytest.mark.asyncio
async def test_checkpoint_resumes_mid_run(monkeypatch):
    # Pre-load state to jump straight into execution.
    initial_state = AppState(
        trending_keywords=["ai"],
        paper_candidates=[{"title": "Paper", "summary": "Summary"}],
        selected_paper={"title": "Paper", "summary": "Summary"},
        user_ready=True,
        paper_approved=True,
    )

    writer_calls = {"count": 0}

    class PromptStub:
        def __init__(self, text):
            self.text = text

        def __or__(self, other):
            return other

    class LLMStub:
        def __init__(self):
            self.ainvoke = AsyncMock(side_effect=self._invoke)

        async def _invoke(self, *_args, **_kwargs):
            writer_calls["count"] += 1
            resp = MagicMock()
            resp.content = "Draft Post"
            return resp

    initial_state.post_draft = None
    initial_state.approved = False

    with patch('src.agents.post_writer.init_chat_model', return_value=LLMStub()), \
         patch('src.agents.post_writer.ChatPromptTemplate.from_template', return_value=PromptStub("template")), \
         patch('src.agents.post_writer.PROMPTS_DIR') as mock_prompts_dir, \
         patch('src.agents.human_approval.interrupt', return_value={"type": "accept", "args": "ok"}), \
         patch('src.agents.post_writer.settings.openai_api_key', "key", create=True), \
         patch('src.agents.post_writer.settings.llm_model', "gpt-stub", create=True):

        mock_prompt_file = MagicMock()
        mock_prompt_file.read_text.return_value = "Template"
        mock_prompts_dir.__truediv__.return_value = mock_prompt_file

        config = {"configurable": {"thread_id": "checkpoint-mid-run"}}

        first_nodes = []
        async for update in graph.astream(initial_state, config=config):
            first_nodes.append(next(iter(update)))
            if "execution_router" in update and "post_writer" in first_nodes:
                break

        assert "post_writer" in first_nodes
        assert writer_calls["count"] == 1

        resumed_nodes = []
        with patch('src.agents.memory_updater.settings.openai_api_key', None, create=True):
            async for update in graph.astream(None, config=config):
                resumed_nodes.append(next(iter(update)))

        assert "post_writer" not in resumed_nodes
        assert "load_memory" not in resumed_nodes
        assert "planning_router" not in resumed_nodes

        state_snapshot = graph.get_state(config)
        assert state_snapshot.values.get("post_draft") == "Draft Post"
