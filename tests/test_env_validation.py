import os
import importlib

import pytest
from pydantic import BaseModel

from tests.utils import feedback


def test_run_structured_judge_raises_without_default_when_no_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    importlib.reload(feedback)  # reload to recompute llm_judge

    with pytest.raises(RuntimeError):
        feedback.llm_judge = None  # force missing judge
        class Dummy(BaseModel):
            score: float = 0.1
        feedback.run_structured_judge("q", Dummy)


def test_run_structured_judge_returns_default_when_provided(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    importlib.reload(feedback)

    class Dummy(BaseModel):
        score: float = 0.5

    default = Dummy(score=0.75)
    result = feedback.run_structured_judge("q", Dummy, default=default)
    assert result.score == 0.75
