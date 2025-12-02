import pytest
from unittest.mock import patch
from src.agents.human_approval import human_approval
from src.state import AppState


@pytest.mark.asyncio
async def test_human_approval_accept():
    state = AppState(post_draft="Draft")
    with patch('src.agents.human_approval.interrupt', return_value={"type": "accept", "args": "Great"}):
        updates = await human_approval(state)

    assert updates["approved"] is True
    assert updates["revision_requested"] is False
    assert any(ev["kind"] == "post_style_feedback" for ev in updates.get("memory_events", []))


@pytest.mark.asyncio
async def test_human_approval_edit_triggers_revision():
    state = AppState(post_draft="Draft")
    with patch('src.agents.human_approval.interrupt', return_value={"type": "edit", "args": {"draft": "Edited Draft"}}):
        updates = await human_approval(state)

    assert updates["approved"] is False
    assert updates["revision_requested"] is True
    assert updates["post_draft"] == "Edited Draft"


@pytest.mark.asyncio
async def test_human_approval_response_returns_to_conversation():
    state = AppState(post_draft="Draft", clarification_history=[])
    with patch('src.agents.human_approval.interrupt', return_value={"type": "response", "args": "Please tweak tone"}):
        updates = await human_approval(state)

    assert updates["return_to_conversation"] is True
    assert updates["user_ready"] is False
    assert "Please tweak tone" in updates["clarification_history"][-1]


@pytest.mark.asyncio
async def test_human_approval_ignore_sets_exit():
    state = AppState(post_draft="Draft")
    with patch('src.agents.human_approval.interrupt', return_value={"type": "ignore", "args": None}):
        updates = await human_approval(state)

    assert updates["exit_requested"] is True
