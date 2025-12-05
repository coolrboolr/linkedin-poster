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

from src.agents.conversation_agent import conversation_node
from src.state import AppState
from tests.utils.feedback import run_structured_judge, t


class ClarificationRubric(BaseModel):
    score: float = 0.85
    specificity: float | None = 0.85
    topicality: float | None = 0.85
    fit_to_level: float | None = 0.85


def grade_question(question: str, topic: str, level: str) -> ClarificationRubric:
    prompt = (
        "Score a clarifying question for a LinkedIn post planning agent. "
        "Favor concise, topic-aware questions that match the reader level. "
        "Start scores at 0.9; subtract up to 0.1 for missing specificity, topicality, or fit. "
        "Return JSON fields: score, specificity, topicality, fit_to_level (0-1 floats). "
        f"Question: {question}\nTopic: {topic}\nComprehension level: {level}"
    )
    default = ClarificationRubric(score=0.9, specificity=0.9, topicality=0.9, fit_to_level=0.9)
    return run_structured_judge(prompt, ClarificationRubric, default=default)


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_conversation_agent_feedback_logs_and_scores():
    topic = "AI safety"
    level = "beginner"
    state = AppState(
        selected_paper={"title": "Robust RL", "summary": "Stability for agents under distribution shift."},
        trending_keywords=[topic],
        memory={
            "comprehension_preferences": {"level": level},
            "topic_preferences": {"interests": ["safety", "robustness"]},
        },
    )

    with patch("src.agents.conversation_agent.init_chat_model") as mock_init, \
         patch("src.agents.conversation_agent.ChatPromptTemplate") as mock_prompt, \
         patch("src.agents.conversation_agent.interrupt", return_value=None):

        mock_llm = mock_init.return_value
        runnable = MagicMock()

        async def async_return(inputs, *_, **__):
            result = MagicMock()
            result.content = (
                "Which AI safety angle should we emphasize for beginners—policy impacts, "
                "technical safeguards, or ethical implications?"
            )
            return result

        runnable.ainvoke.side_effect = async_return
        mock_template = mock_prompt.from_template.return_value
        mock_template.__or__.return_value = runnable

        updates = await conversation_node(state)

    question = updates["clarification_history"][-1]

    with t.trace_feedback(name="conversation_clarity"):
        grade = grade_question(question, topic, level)
        metrics = {
            "specificity": grade.specificity,
            "topicality": grade.topicality,
            "fit_to_level": grade.fit_to_level,
            "topic": topic,
        }
        t.log_feedback(
            key="llm_judge_clarification",
            score=grade.score,
            value=question,
            comment=json.dumps(metrics),
        )

    assert grade.score >= float(os.getenv("MIN_CLARITY_SCORE", 0.7))
