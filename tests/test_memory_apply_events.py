import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.memory.apply_events import apply_memory_events
from src.memory.models import MemoryEvent


@pytest.mark.asyncio
async def test_apply_memory_events_topics_and_feedback():
    store = MagicMock()
    store.topic = {}
    store.comp = {}
    store.format = {}

    events = [
        MemoryEvent(
            kind="paper_feedback",
            source="paper_review",
            message="Too dense",
            current_title="Paper A",
            topic="ai",
        ),
        MemoryEvent(
            kind="paper_selection",
            source="paper_review",
            selected_title="Paper A",
            topic="ai",
        ),
    ]

    await apply_memory_events(
        store=store,
        events=events,
        approved=True,
        selected_paper={"title": "Paper B"},
        human_feedback=None,
        style_llm=None,
        comp_llm=None,
    )

    topic_dict = store.topic
    assert "Paper A" in topic_dict.get("liked_topics", [])
    assert "Paper B" in topic_dict.get("liked_topics", [])
    assert any(e["message"] == "Too dense" for e in topic_dict.get("feedback_log", []))


@pytest.mark.asyncio
async def test_apply_memory_events_style_and_comp_llm():
    class DummyStore:
        def __init__(self):
            self.topic = {}
            self.comp = {"level": "intermediate"}
            self.format = {"length": "medium"}

    store = DummyStore()

    events = [
        MemoryEvent(
            kind="post_style_feedback",
            source="human_approval",
            message="Shorter please",
        ),
        MemoryEvent(
            kind="comprehension_feedback",
            source="conversation",
            message="Simplify",
        ),
    ]

    style_result = MagicMock()
    style_result.model_dump.return_value = {"length": "short"}
    comp_result = MagicMock()
    comp_result.model_dump.return_value = {"level": "beginner"}

    style_llm = MagicMock()
    comp_llm = MagicMock()
    style_llm.ainvoke = AsyncMock(return_value=style_result)
    comp_llm.ainvoke = AsyncMock(return_value=comp_result)

    with patch('src.memory.apply_events.ChatPromptTemplate') as MockPrompt:
        mock_template_style = MagicMock()
        mock_template_comp = MagicMock()
        MockPrompt.from_template.side_effect = [mock_template_style, mock_template_comp]
        mock_template_style.__or__.return_value = style_llm
        mock_template_comp.__or__.return_value = comp_llm

        await apply_memory_events(
            store=store,
            events=events,
            approved=False,
            selected_paper=None,
            human_feedback="Shorter please",
            style_llm=style_llm,
            comp_llm=comp_llm,
        )

    assert store.format == {"length": "short"}
    assert store.comp == {"level": "beginner"}


@pytest.mark.asyncio
async def test_apply_memory_events_unknown_kind_logs(caplog):
    store = MagicMock()
    store.topic = {}
    store.comp = {}
    store.format = {}

    events = [{"kind": "unknown_kind", "source": "test", "message": "n/a"}]

    await apply_memory_events(
        store=store,
        events=events,
        approved=False,
        selected_paper=None,
        human_feedback=None,
        style_llm=None,
        comp_llm=None,
    )

    assert store.topic == {}
    assert store.comp == {}
    assert store.format == {}
