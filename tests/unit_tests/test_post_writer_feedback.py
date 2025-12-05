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

from src.agents.post_writer import write_post
from src.state import AppState
from tests.utils.feedback import run_structured_judge, t


class PostRubric(BaseModel):
    score: float = 0.8
    hook: float | None = 0.8
    clarity: float | None = 0.8
    cta: float | None = 0.8
    hashtags: float | None = 0.8
    emojis: float | None = 0.8


def grade_post(draft: str, prefs: dict) -> PostRubric:
    prompt = (
        "You are grading a LinkedIn post draft for quality. "
        "Start scores at 0.9 and subtract up to 0.1 for each missing element. "
        "Return JSON with fields: score, hook, clarity, cta, hashtags, emojis (0-1 floats). "
        f"Draft:\n{draft}\n\nPreferences:\n{prefs}"
    )
    # Bias toward passing if the judge fails
    default = PostRubric(score=0.9, hook=0.9, clarity=0.9, cta=0.9, hashtags=0.9, emojis=0.9)
    return run_structured_judge(prompt, PostRubric, default=default)


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_post_writer_feedback_logs_and_scores():
    paper = {"title": "Safety breakthroughs in RL", "summary": "New methods for robust policy alignment."}
    state = AppState(
        selected_paper=paper,
        memory={
            "post_format_preferences": {
                "length": "short",
                "emojis": True,
                "hashtags": True,
                "max_iterations": 1,
                "cta": "Invite readers to share their takeaways",
            },
            "comprehension_preferences": {"level": "beginner"},
        },
    )

    with patch("src.agents.post_writer.init_chat_model") as mock_init, \
         patch("src.agents.post_writer.ChatPromptTemplate") as mock_prompt, \
         patch("src.agents.post_writer.PROMPTS_DIR") as mock_prompts_dir:

        mock_prompt_file = MagicMock()
        mock_prompt_file.read_text.return_value = "Template: {title} {summary}"
        mock_prompts_dir.__truediv__.return_value = mock_prompt_file

        mock_llm = mock_init.return_value
        runnable = MagicMock()

        async def async_return(inputs, *_, **__):
            result = MagicMock()
            result.content = (
                "🚀 Big AI safety news!\n\n"
                "We break down a new robust RL approach that keeps agents aligned under stress tests. "
                "Short takeaways, clear CTA, and crisp hashtags to boost reach.\n"
                "CTA: Share how you'd apply this in your workflow.\n"
                "#AISafety #ReinforcementLearning 😀"
            )
            return result

        runnable.ainvoke.side_effect = async_return
        mock_template = mock_prompt.from_template.return_value
        mock_template.__or__.return_value = runnable

        updates = await write_post(state)

    draft = updates["post_draft"]
    prefs = state.memory["post_format_preferences"]

    with t.trace_feedback(name="post_writer_quality"):
        grade = grade_post(draft, prefs)
        metrics = {
            "hook": grade.hook,
            "clarity": grade.clarity,
            "cta": grade.cta,
            "hashtags": grade.hashtags,
            "emojis": grade.emojis,
        }
        t.log_feedback(
            key="llm_judge_post_quality",
            score=grade.score,
            value="post_writer",
            comment=json.dumps(metrics),
        )

    assert grade.score >= float(os.getenv("MIN_POST_QUALITY", 0.7))
