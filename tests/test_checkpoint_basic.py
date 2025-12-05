import pytest

from src.graph import graph
from src.state import AppState

pytestmark = pytest.mark.anyio


async def test_checkpoint_runs():
    config = {"configurable": {"thread_id": "checkpoint-test"}}
    async for _ in graph.astream(AppState(), config=config):
        pass  # we only care that it does not throw
