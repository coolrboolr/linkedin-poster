import pytest
from unittest.mock import patch
from src.agents.human_paper_review import human_paper_review
from src.state import AppState


CANDS = [
    {"title": "Paper 1", "summary": "S1", "url": "u1", "published": "2023-01-01"},
    {"title": "Paper 2", "summary": "S2", "url": "u2", "published": "2023-02-01"},
]


@pytest.mark.asyncio
async def test_paper_review_accept():
    state = AppState(
        paper_candidates=CANDS,
        selected_paper=CANDS[0],
        trending_keywords=["ai"],
    )
    with patch('src.agents.human_paper_review.interrupt', return_value={"type": "accept", "args": None}):
        updates = await human_paper_review(state)

    assert updates["paper_approved"] is True


@pytest.mark.asyncio
async def test_paper_review_switch_index():
    state = AppState(
        paper_candidates=CANDS,
        selected_paper=CANDS[0],
        trending_keywords=["ai"],
    )
    with patch('src.agents.human_paper_review.interrupt', return_value={"type": "edit", "args": {"Selected Paper": "1. Paper 2"}}):
        updates = await human_paper_review(state)

    assert updates["paper_approved"] is True
    assert updates["selected_paper"]["title"] == "Paper 2"
    assert updates["user_ready"] is True


@pytest.mark.asyncio
async def test_paper_review_invalid_index():
    state = AppState(paper_candidates=CANDS, selected_paper=CANDS[0])
    with patch('src.agents.human_paper_review.interrupt', return_value={"type": "edit", "args": {"Selected Paper": "invalid"}}):
        updates = await human_paper_review(state)

    assert updates["paper_approved"] is False
    assert updates.get("user_ready", False) is False


@pytest.mark.asyncio
async def test_paper_review_response_logs_feedback():
    state = AppState(paper_candidates=CANDS, selected_paper=CANDS[0], trending_keywords=["ai"])
    with patch('src.agents.human_paper_review.interrupt', return_value={"type": "response", "args": "Too shallow"}):
        updates = await human_paper_review(state)

    assert updates["paper_approved"] is False
    assert any(ev["kind"] == "paper_feedback" for ev in updates.get("memory_events", []))


@pytest.mark.asyncio
async def test_paper_review_ignore_sets_exit():
    state = AppState(paper_candidates=CANDS, selected_paper=CANDS[0])
    with patch('src.agents.human_paper_review.interrupt', return_value={"type": "ignore", "args": None}):
        updates = await human_paper_review(state)

    assert updates["exit_requested"] is True
