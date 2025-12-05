import json
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

if os.getenv("ENABLE_AUTO_GRADING", "true").lower() != "true":
    pytest.skip("Grading disabled via env", allow_module_level=True)
if not os.getenv("LANGSMITH_API_KEY"):
    pytest.skip("LangSmith not configured", allow_module_level=True)
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OpenAI not configured for grading", allow_module_level=True)

from src.agents.relevance_ranker import rank_papers
from src.state import AppState
from tests.utils.feedback import run_structured_judge, t


class RankingRubric(BaseModel):
    score: float = 0.85
    preference_alignment: float | None = 0.85
    summary_alignment: float | None = 0.85


def grade_ranking(selected: dict, prefs: dict, topic: str) -> RankingRubric:
    prompt = (
        "You are grading whether the chosen paper aligns with the user's topic preferences. "
        "Start at 0.9 and deduct up to 0.1 for weak preference alignment and up to 0.1 for summary mismatch. "
        "Return JSON fields: score, preference_alignment, summary_alignment (0-1 floats). "
        f"Topic: {topic}\nPreferences: {prefs}\nSelected paper: {selected}"
    )
    default = RankingRubric(score=0.9, preference_alignment=0.9, summary_alignment=0.9)
    return run_structured_judge(prompt, RankingRubric, default=default)


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_relevance_ranker_feedback_logs_and_scores():
    topic = "robotics safety"
    prefs = {"interests": ["robotics", "safety"], "avoid": ["theory-only"]}
    candidates = [
        {"title": "Safe control for collaborative robots", "summary": "Applied safety layers for cobots in factories."},
        {"title": "Pure theory of RL convergence", "summary": "Math heavy discussion with no experiments."},
    ]
    state = AppState(paper_candidates=candidates, trending_keywords=[topic], memory={"topic_preferences": prefs})

    with patch("src.agents.relevance_ranker.init_chat_model") as mock_init, \
         patch("src.agents.relevance_ranker.ChatPromptTemplate") as mock_prompt:

        base_llm = mock_init.return_value
        structured_llm = MagicMock()
        base_llm.with_structured_output.return_value = structured_llm

        class RankResult:
            index = 0
            rationale = "Applied safety matches preferences."

        async def async_return(*_, **__):
            return RankResult()

        structured_llm.ainvoke.side_effect = async_return
        mock_template = mock_prompt.from_template.return_value
        mock_template.__or__.return_value = structured_llm

        updates = await rank_papers(state)

    selected = updates["selected_paper"]

    with t.trace_feedback(name="relevance_ranking"):
        grade = grade_ranking(selected, prefs, topic)
        metrics = {
            "preference_alignment": grade.preference_alignment,
            "summary_alignment": grade.summary_alignment,
            "topic": topic,
        }
        t.log_feedback(
            key="llm_judge_ranking",
            score=grade.score,
            value=selected.get("title"),
            comment=json.dumps(metrics),
        )

    assert grade.score >= float(os.getenv("MIN_RANKING_SCORE", 0.7))
