import json
import os

import pytest
from pydantic import BaseModel
from unittest.mock import patch

if os.getenv("ENABLE_AUTO_GRADING", "true").lower() != "true":
    pytest.skip("Grading disabled via env", allow_module_level=True)
if not os.getenv("LANGSMITH_API_KEY"):
    pytest.skip("LangSmith not configured", allow_module_level=True)
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OpenAI not configured for grading", allow_module_level=True)

from src.agents.human_approval import human_approval
from src.state import AppState
from tests.utils.feedback import run_structured_judge, t


class ApprovalCopyRubric(BaseModel):
    score: float = 0.85
    concision: float | None = 0.85
    includes_latest_draft: float | None = 0.85
    leak_free: float | None = 0.85


def grade_description(description: str) -> ApprovalCopyRubric:
    prompt = (
        "Evaluate an approval prompt shown to a user. "
        "Reward concise instructions, inclusion of the latest draft, and avoidance of internal state leakage. "
        "Start scores at 0.9 and deduct up to 0.1 for each weakness. "
        "Return JSON fields: score, concision, includes_latest_draft, leak_free (0-1 floats). "
        f"Description:\n{description}"
    )
    default = ApprovalCopyRubric(score=0.9, concision=0.9, includes_latest_draft=0.9, leak_free=0.9)
    return run_structured_judge(prompt, ApprovalCopyRubric, default=default)


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_human_approval_feedback_logs_and_scores():
    captured = {}

    def fake_interrupt(payload):
        captured["description"] = payload["description"]
        return {"type": "accept", "args": "Looks concise"}

    state = AppState(
        post_draft="Draft v2 with updates on safety.",
        clarification_history=["Assistant: Draft ready?"],
        revision_history=[
            {"revision_number": 1, "instruction": "Shorten intro", "draft_before": "Long intro", "draft_after": "Short intro"},
            {"revision_number": 2, "instruction": "Add CTA", "draft_before": "Short intro", "draft_after": "Short intro + CTA"},
        ],
    )

    # Patch inside function import location
    with patch("src.agents.human_approval.interrupt", fake_interrupt):
        updates = await human_approval(state)

    description = captured["description"]

    with t.trace_feedback(name="human_approval_copy"):
        grade = grade_description(description)
        metrics = {
            "concision": grade.concision,
            "includes_latest_draft": grade.includes_latest_draft,
            "leak_free": grade.leak_free,
        }
        t.log_feedback(
            key="llm_judge_approval_copy",
            score=grade.score,
            value="human_approval",
            comment=json.dumps(metrics),
        )

    assert grade.score >= float(os.getenv("MIN_APPROVAL_COPY_SCORE", 0.7))
    assert updates["approved"] is True
