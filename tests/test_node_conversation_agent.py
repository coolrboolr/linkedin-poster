from src.agents.conversation_agent import conversation_node
from src.state import AppState
from unittest.mock import MagicMock, patch
import pytest
from langgraph.errors import GraphInterrupt

@pytest.mark.asyncio
async def test_conversation_agent_openai():
    # Case 1: Paper selected -> model asks, user says READY
    state = AppState(selected_paper={"title": "Paper", "summary": "Summary"})
    with patch('src.agents.conversation_agent.init_chat_model') as MockInitModel, \
         patch('src.agents.conversation_agent.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.conversation_agent.interrupt', return_value={"type": "accept", "args": "READY"}) as mock_interrupt:

        mock_runnable = MagicMock()

        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "Clarifying question?"
            return mock_content

        mock_runnable.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable

        updates = await conversation_node(state)

        assert updates["user_ready"] is True
        assert updates["clarification_history"] == ["Clarifying question?"]
        mock_interrupt.assert_called_once()
    
    # Case 2: No paper -> Clarification
    state = AppState(selected_paper=None)
    
    with patch('src.agents.conversation_agent.init_chat_model') as MockInitModel, \
         patch('src.agents.conversation_agent.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.conversation_agent.interrupt', return_value=None) as mock_interrupt:
        
        mock_runnable = MagicMock()
        # Simulate LLM asking a question
        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "What specific aspect of AI?"
            return mock_content
        mock_runnable.ainvoke.side_effect = async_return
        
        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable
        
        updates = await conversation_node(state)
        
        assert updates["user_ready"] is False
        assert "clarification_history" in updates
        assert updates["clarification_history"] == ["What specific aspect of AI?"]

    # Case 3: No paper -> Clarification (LLM asks, then user interrupts)
    state = AppState(selected_paper=None)
    
    with patch('src.agents.conversation_agent.init_chat_model') as MockInitModel, \
         patch('src.agents.conversation_agent.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.conversation_agent.interrupt', return_value={"type": "accept", "args": "READY"}) as mock_interrupt:
        
        mock_runnable = MagicMock()
        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "Question?"
            return mock_content
        mock_runnable.ainvoke.side_effect = async_return
        
        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable
        
        updates = await conversation_node(state)
        assert updates["user_ready"] is True
        assert updates["clarification_history"] == ["Question?"]
        mock_interrupt.assert_called_once()

@pytest.mark.asyncio
async def test_conversation_agent_propagates_interrupt():
    state = AppState(selected_paper={"title": "Paper", "summary": "Summary"})
    with patch('src.agents.conversation_agent.init_chat_model') as MockInitModel, \
         patch('src.agents.conversation_agent.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.conversation_agent.interrupt', side_effect=GraphInterrupt("test interrupt")):

        mock_runnable = MagicMock()

        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "Clarifying question?"
            return mock_content

        mock_runnable.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable

        with pytest.raises(GraphInterrupt):
            await conversation_node(state)


@pytest.mark.asyncio
async def test_conversation_agent_response_adds_memory_and_history():
    state = AppState(selected_paper={"title": "Paper", "summary": "Summary"})
    with patch('src.agents.conversation_agent.init_chat_model') as MockInitModel, \
         patch('src.agents.conversation_agent.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.conversation_agent.interrupt', return_value={"type": "response", "args": "Can you simplify?"}):

        mock_runnable = MagicMock()

        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "Clarifying question?"
            return mock_content

        mock_runnable.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable

        updates = await conversation_node(state)

        assert updates["user_ready"] is False
        assert updates["clarification_history"][-1] == "User: Can you simplify?"
        assert any(ev["kind"] == "comprehension_feedback" for ev in updates.get("memory_events", []))


@pytest.mark.asyncio
async def test_conversation_agent_ignore_sets_exit():
    state = AppState(selected_paper={"title": "Paper", "summary": "Summary"})
    with patch('src.agents.conversation_agent.init_chat_model') as MockInitModel, \
         patch('src.agents.conversation_agent.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.conversation_agent.interrupt', return_value={"type": "ignore", "args": None}):

        mock_runnable = MagicMock()

        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "Clarifying question?"
            return mock_content

        mock_runnable.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable

        updates = await conversation_node(state)

        assert updates["exit_requested"] is True
