import pytest
from src.graph import planning_router, execution_router
from src.state import AppState


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "trend_scanner"),
        ({"trending_keywords": ["ai"]}, "arxiv_fetcher"),
        ({"trending_keywords": ["ai"], "paper_candidates": [{}]}, "relevance_ranker"),
        ({"trending_keywords": ["ai"], "paper_candidates": [{}], "selected_paper": {"title": "t"}}, "conversation_agent"),
        ({"trending_keywords": ["ai"], "paper_candidates": [{}], "selected_paper": {"title": "t"}, "user_ready": True}, "human_paper_review"),
        ({"trending_keywords": ["ai"], "paper_candidates": [{}], "selected_paper": {"title": "t"}, "user_ready": True, "paper_approved": True}, "execution_router"),
        ({"exit_requested": True}, "exit"),
    ],
)
async def test_planning_router_branches(kwargs, expected):
    state = AppState(**kwargs)
    result = await planning_router(state)
    assert result["next_step"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"exit_requested": True}, "exit"),
        ({"return_to_conversation": True}, "planning_router"),
        ({"revision_requested": True}, "post_writer"),
        ({"post_draft": None}, "post_writer"),
        ({"post_draft": "Draft", "approved": False}, "human_approval"),
        ({"post_draft": "Draft", "approved": True}, "memory_updater"),
    ],
)
async def test_execution_router_branches(kwargs, expected):
    state = AppState(**kwargs)
    result = await execution_router(state)
    assert result["next_step"] == expected
