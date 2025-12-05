import pytest
import os

from src.graph import graph

pytestmark = pytest.mark.anyio


@pytest.mark.skipif(not os.getenv("LANGSMITH_API_KEY"), reason="LANGSMITH_API_KEY not set; skipping LangSmith integration test.")
async def test_agent_simple_passthrough() -> None:
    inputs = {"changeme": "some_val"}
    res = await graph.ainvoke(inputs, config={"configurable": {"thread_id": "integration_simple_passthrough"}})
    assert res is not None
