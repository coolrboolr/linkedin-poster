import pytest
from unittest.mock import MagicMock, patch
from src.state import AppState
from src.agents.conversation_agent import conversation_node
from src.agents.human_paper_review import human_paper_review
from src.agents.human_approval import human_approval
from src.core.constants import MEMORY_KIND_POST_STYLE_FEEDBACK

# --- Conversation Agent Tests ---

@pytest.mark.asyncio
async def test_conversation_agent_resumes_on_response():
    """Test that valid user response clears 'awaiting_user_response'."""
    state = AppState(
        chat_history=[{"role": "assistant", "message": "Hello?"}],
        awaiting_user_response=True,
        trending_keywords=["AI"],
        selected_paper={"title": "Test Paper", "summary": "Summary"},
        user_ready=False
    )
    
    # Mock settings and internal helpers to bypass LLM calls
    with patch("src.agents.conversation_agent.settings") as mock_settings, \
         patch("src.agents.conversation_agent.interrupt") as mock_interrupt, \
         patch("src.agents.conversation_agent._invoke_legacy") as mock_invoke:
        
        mock_settings.openai_api_key = "fake_key"
        mock_settings.conversation_model = "gpt-4"
        mock_invoke.return_value = ("Clarification question?", [], "Clarification question?")
        
        # Simulate user providing a response
        mock_interrupt.return_value = {"type": "response", "args": "I want more details about AI."}
        
        result = await conversation_node(state)
        
        assert result["awaiting_user_response"] is False
        assert result["user_ready"] is False
        assert len(result["chat_history"]) > 1
        assert result["chat_history"][-1]["message"] == "I want more details about AI."

@pytest.mark.asyncio
async def test_conversation_agent_accepts_and_sets_ready():
    """Test that 'accept' sets user_ready=True."""
    state = AppState(
        chat_history=[],
        trending_keywords=["AI"],
        selected_paper={"title": "Test Paper", "summary": "Summary"},
        user_ready=False
    )
    
    with patch("src.agents.conversation_agent.settings") as mock_settings, \
         patch("src.agents.conversation_agent.interrupt") as mock_interrupt, \
         patch("src.agents.conversation_agent._invoke_legacy") as mock_invoke:
        
        mock_settings.openai_api_key = "fake_key"
        mock_settings.conversation_model = "gpt-4"
        mock_invoke.return_value = ("Clarification question?", [], "Clarification question?")

        mock_interrupt.return_value = {"type": "accept", "args": "Looks good."}
        
        result = await conversation_node(state)
        
        assert result["user_ready"] is True
        assert result["awaiting_user_response"] is False


# --- Human Paper Review Tests ---

@pytest.mark.asyncio
async def test_human_paper_review_switch_paper():
    """Test that switching paper updates state correctly."""
    candidates = [
        {"title": "Paper A", "summary": "A"},
        {"title": "Paper B", "summary": "B"}
    ]
    state = AppState(
        paper_candidates=candidates,
        selected_paper=candidates[0],
        trending_keywords=["Topic"],
        user_ready=False
    )
    
    with patch("src.agents.human_paper_review.interrupt") as mock_interrupt:
        # Simulate user selecting index 1
        mock_interrupt.return_value = {"type": "edit", "args": {"Selected Paper": "1. Paper B"}}
        
        result = await human_paper_review(state)
        
        assert result["selected_paper"]["title"] == "Paper B"
        assert result["paper_approved"] is True
        # Verify it implicitly sets user_ready when selection is made
        assert result["user_ready"] is True
        assert any(e["kind"] == "paper_selection" for e in result["memory_events"])

@pytest.mark.asyncio
async def test_human_paper_review_response_routes_to_conversation():
    """Test that 'response' (feedback) sends user back to conversation logic."""
    state = AppState(
        selected_paper={"title": "A", "summary": "A"},
        paper_candidates=[],
        trending_keywords=["Topic"],
        user_ready=True 
    )
    
    with patch("src.agents.human_paper_review.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"type": "response", "args": "Is this new?"}
        
        result = await human_paper_review(state)
        
        assert result["paper_approved"] is False
        assert result["user_ready"] is False # Should be false to trigger conversation_agent
        assert result["chat_history"][-1]["message"] == "Is this new?"


# --- Human Approval Tests ---

@pytest.mark.asyncio
async def test_human_approval_edit_requests_new_draft():
    """Test that 'edit' creates a revision request."""
    state = AppState(
        post_draft="Old Draft",
        revision_history=[],
        edit_requests=[],
        post_history=[]
    )
    
    with patch("src.agents.human_approval.interrupt") as mock_interrupt:
        # User directly edits text
        mock_interrupt.return_value = {
            "type": "edit", 
            "args": {"instruction": "Make it pop", "draft": "New Draft"}
        }
        
        result = await human_approval(state)
        
        assert result["approved"] is False
        assert result["revision_requested"] is True
        assert result["post_draft"] == "New Draft"
        assert len(result["edit_requests"]) == 1
        assert result["edit_requests"][0]["instruction"] == "Make it pop"

@pytest.mark.asyncio
async def test_human_approval_response_logic():
    """
    Test 'response' behavior. 
    CURRENTLY: This might route to conversation without revision_requested=True.
    GOAL: We want to verify if we need to change this to request revision.
    """
    state = AppState(
        post_draft="Draft",
        revision_history=[],
        edit_requests=[],
        post_history=[],
        chat_history=[]
    )
    
    with patch("src.agents.human_approval.interrupt") as mock_interrupt:
        # User provides feedback but no direct edit
        mock_interrupt.return_value = {
            "type": "response",
            "args": "Make it shorter."
        }
        
        result = await human_approval(state)
        
        # We WANT this to trigger a revision so the user sees a new draft reflecting their feedback
        assert result["revision_requested"] is True
        assert result["return_to_conversation"] is False
