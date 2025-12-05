import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.agents.memory_updater import update_memory
from src.state import AppState

@pytest.mark.asyncio
async def test_memory_updater_style_update(tmp_path):
    # Setup state with feedback
    state = AppState(
        approved=False,
        human_feedback="Make it shorter",
        memory={"post_format_preferences": {"length": "long"}},
        memory_events=[
            {"kind": "post_style_feedback", "source": "human_approval", "message": "shorter"}
        ],
    )
    
    # Mock MemoryStore to use tmp_path
    with patch('src.agents.memory_updater.MemoryStore') as MockStore, \
         patch('src.agents.memory_updater.init_chat_model') as MockInitModel, \
         patch('src.agents.memory_updater.apply_memory_events') as mock_apply_events, \
         patch('src.agents.memory_updater.settings') as mock_settings:
         
        # Configure settings
        mock_settings.openai_api_key = "test-key"
        mock_settings.llm_model = "test-model"
        
        # Configure MemoryStore
        mock_store_instance = MockStore.return_value
        mock_store_instance.topic = {}
        mock_store_instance.comp = {}
        mock_store_instance.format = {"length": "long"}

        def _get_all():
            return {
                "topic_preferences": mock_store_instance.topic,
                "comprehension_preferences": mock_store_instance.comp,
                "post_format_preferences": mock_store_instance.format,
            }
        mock_store_instance.get_all.side_effect = _get_all
        
        # Configure LLM
        # Configure LLM structured output for style update
        base_llm = MockInitModel.return_value
        structured_llm = MagicMock()
        base_llm.with_structured_output.return_value = structured_llm

        class StyleResult:
            def model_dump(self):
                return {"length": "short"}

        async def async_return(*args, **kwargs):
            return StyleResult()

        structured_llm.ainvoke.side_effect = async_return

        # Simulate apply_memory_events writing the new format
        async def fake_apply(store, events, **kwargs):
            store.format = {"length": "short"}
        mock_apply_events.side_effect = fake_apply
        
        # Run node
        updates = await update_memory(state)
        
        # Verify store updated
        assert mock_store_instance.format == {"length": "short"}
        mock_store_instance.save.assert_called_once()
        
        # Verify state update
        assert updates["memory"]["post_format_preferences"]["length"] == "short"


@pytest.mark.asyncio
async def test_memory_updater_comprehension_update():
    state = AppState(
        memory_events=[
            {"kind": "comprehension_feedback", "source": "conversation", "message": "Beginner level please."}
        ],
        memory={"comprehension_preferences": {"level": "intermediate"}},
    )

    with patch('src.agents.memory_updater.MemoryStore') as MockStore, \
         patch('src.agents.memory_updater.init_chat_model') as MockInitModel, \
         patch('src.agents.memory_updater.apply_memory_events') as mock_apply_events, \
         patch('src.agents.memory_updater.settings') as mock_settings:

        mock_settings.openai_api_key = "test-key"
        mock_settings.llm_model = "test-model"

        mock_store = MockStore.return_value
        mock_store.topic = {}
        mock_store.comp = {"level": "intermediate"}
        mock_store.format = {}

        def _get_all():
            return {
                "topic_preferences": mock_store.topic,
                "comprehension_preferences": mock_store.comp,
                "post_format_preferences": mock_store.format,
            }
        mock_store.get_all.side_effect = _get_all

        base_llm = MockInitModel.return_value
        structured_llm = MagicMock()
        base_llm.with_structured_output.return_value = structured_llm

        class CompResult:
            def model_dump(self):
                return {"level": "beginner"}

        structured_llm.ainvoke = AsyncMock(return_value=CompResult())

        async def fake_apply(store, events, **kwargs):
            store.comp = {"level": "beginner"}

        mock_apply_events.side_effect = fake_apply

        updates = await update_memory(state)

        assert updates["memory"]["comprehension_preferences"]["level"] == "beginner"
        mock_store.save.assert_called_once()


@pytest.mark.asyncio
async def test_memory_updater_liked_topics_and_feedback_log():
    state = AppState(
        approved=True,
        selected_paper={"title": "Test Paper"},
        memory_events=[
            {"kind": "paper_feedback", "source": "paper_review", "message": "Too long", "current_title": "Test Paper", "topic": "ai"}
        ],
        memory={"topic_preferences": {}},
    )

    with patch('src.agents.memory_updater.MemoryStore') as MockStore, \
         patch('src.agents.memory_updater.settings') as mock_settings:

        # Force LLM skip
        mock_settings.openai_api_key = None
        mock_settings.llm_model = "test-model"

        mock_store = MockStore.return_value
        mock_store.topic = {}
        mock_store.comp = {}
        mock_store.format = {}

        def _get_all():
            return {
                "topic_preferences": mock_store.topic,
                "comprehension_preferences": mock_store.comp,
                "post_format_preferences": mock_store.format,
            }
        mock_store.get_all.side_effect = _get_all

        updates = await update_memory(state)

        topic_prefs = updates["memory"]["topic_preferences"]
        assert "Test Paper" in topic_prefs.get("liked_topics", [])
        assert any(entry["message"] == "Too long" for entry in topic_prefs.get("feedback_log", []))
        mock_store.save.assert_called_once()
