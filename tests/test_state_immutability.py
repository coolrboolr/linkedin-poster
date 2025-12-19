from copy import deepcopy

from src.graph import planning_router, execution_router
from src.state import AppState
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_kwargs",
    [
        {},
        {"trending_keywords": ["ai"]},
        {"trending_keywords": ["ai"], "paper_candidates": [{}], "selected_paper": {"title": "t"}, "user_ready": True},
        {"exit_requested": True},
    ],
)
async def test_routers_do_not_mutate_state(state_kwargs):
    state = AppState(**state_kwargs)
    before = state.model_copy(deep=True)

    await planning_router(state)
    assert state == before

    await execution_router(state)
    assert state == before
