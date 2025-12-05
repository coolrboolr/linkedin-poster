import os
from typing import Type, TypeVar

from langsmith import testing as t  # re-exported for convenience in tests
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# LangSmith doc-compliant judge client
def _init_llm_judge() -> ChatOpenAI | None:
    try:
        return ChatOpenAI(model=os.getenv("LLM_JUDGE_MODEL", "gpt-4o-mini"), temperature=0)
    except Exception:
        # Creation can fail when OPENAI_API_KEY is absent; tests will skip before use.
        return None


llm_judge = _init_llm_judge()


def require_llm_judge() -> ChatOpenAI:
    """
    Ensure we have a configured LLM judge.
    """
    if llm_judge is None:
        raise RuntimeError("LLM judge is not configured; set OPENAI_API_KEY.")
    return llm_judge


def extract_score(text: str, default: float = 0.0) -> float:
    for line in text.splitlines():
        if "score" in line.lower():
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                pass
    return default


T = TypeVar("T", bound=BaseModel)


def run_structured_judge(prompt: str, schema: Type[T], default: T | None = None) -> T:
    """
    Call the shared judge with a structured schema; optionally return a default on failure.
    """
    if llm_judge is None:
        if default is not None:
            return default
        raise RuntimeError("LLM judge is not configured; set OPENAI_API_KEY.")

    llm = llm_judge.with_structured_output(schema)
    try:
        return llm.invoke(prompt)
    except Exception:
        if default is not None:
            return default
        raise


__all__ = ["t", "llm_judge", "extract_score", "require_llm_judge", "run_structured_judge"]
