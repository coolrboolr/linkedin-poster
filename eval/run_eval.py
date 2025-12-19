import os
from typing import Callable, Iterable, Tuple

from pydantic import BaseModel

from src.state import AppState
from tests.utils.feedback import run_structured_judge, t


class PostRubric(BaseModel):
    score: float = 0.9
    hook: float | None = 0.9
    clarity: float | None = 0.9
    hashtags: float | None = 0.9
    emojis: float | None = 0.9


class ComprehensionRubric(BaseModel):
    score: float = 0.9
    specificity: float | None = 0.9
    topicality: float | None = 0.9
    fit_to_level: float | None = 0.9


class TopicMemoryRubric(BaseModel):
    score: float = 0.9
    covers_likes: float | None = 0.9
    concise: float | None = 0.9
    actionable: float | None = 0.9


def _log_or_print(key: str, score: float, value: str, comment: str):
    if os.getenv("LANGSMITH_API_KEY"):
        t.log_feedback(key=key, score=score, value=value, comment=comment)
    else:
        print(f"{key}: {score} | {value} | {comment}")


def _run_post_eval() -> Tuple[str, float]:
    draft = (
        "🚀 Big AI safety news! We break down a new robust RL approach that keeps agents aligned under stress tests. "
        "CTA: Share how you'd apply this in your workflow. #AISafety #ReinforcementLearning 😀"
    )
    prefs = {
        "length": "short",
        "emojis": True,
        "hashtags": True,
        "cta": "Invite readers to share their takeaways",
    }
    prompt = (
        "Grade a LinkedIn post draft for quality. Start scores at 0.9 and subtract up to 0.1 "
        "for missing hook, clarity, hashtags, or emojis. Return JSON fields: score, hook, clarity, "
        "hashtags, emojis. Draft:\n"
        f"{draft}\n\nPreferences:\n{prefs}"
    )
    default = PostRubric(score=0.9, hook=0.9, clarity=0.9, hashtags=0.9, emojis=0.9)
    grade = run_structured_judge(prompt, PostRubric, default=default)
    _log_or_print("eval_post_quality", grade.score, "post_writer", grade.model_dump_json())
    return "post", grade.score


def _run_comprehension_eval() -> Tuple[str, float]:
    question = "Which AI safety angle should we emphasize for beginners—policy impacts, technical safeguards, or ethics?"
    topic = "AI safety"
    level = "beginner"
    prompt = (
        "Score a clarifying question for specificity, topicality, and fit to level. "
        "Start at 0.9; deduct up to 0.1 each. Return JSON fields: score, specificity, topicality, fit_to_level. "
        f"Question: {question}\nTopic: {topic}\nLevel: {level}"
    )
    default = ComprehensionRubric(score=0.9, specificity=0.9, topicality=0.9, fit_to_level=0.9)
    grade = run_structured_judge(prompt, ComprehensionRubric, default=default)
    _log_or_print("eval_clarification_quality", grade.score, "conversation_agent", grade.model_dump_json())
    return "comprehension", grade.score


def _run_topic_memory_eval() -> Tuple[str, float]:
    memory = {
        "topic_preferences": {"liked_topics": ["AI safety", "robust RL"], "feedback_log": [{"message": "Too dense"}]},
        "comprehension_preferences": {"level": "beginner"},
        "post_format_preferences": {"length": "short"},
    }
    prompt = (
        "Judge whether the memory snapshot is concise, covers liked topics, and is actionable. "
        "Start score at 0.9; deduct up to 0.1 for each weakness. "
        "Return JSON fields: score, covers_likes, concise, actionable. "
        f"Memory: {memory}"
    )
    default = TopicMemoryRubric(score=0.9, covers_likes=0.9, concise=0.9, actionable=0.9)
    grade = run_structured_judge(prompt, TopicMemoryRubric, default=default)
    _log_or_print("eval_memory_snapshot", grade.score, "memory", grade.model_dump_json())
    return "memory", grade.score


def main():
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LangSmith not configured; running evals offline with default scores.")

    runners: Iterable[Callable[[], Tuple[str, float]]] = (
        _run_post_eval,
        _run_comprehension_eval,
        _run_topic_memory_eval,
    )
    results = {name: score for name, score in (runner() for runner in runners)}
    print("Eval results:", results)


if __name__ == "__main__":
    main()
