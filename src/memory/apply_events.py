import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.runnables import Runnable

from langchain_core.prompts import ChatPromptTemplate

from src.memory import MemoryStore
from src.memory.models import MemoryEvent, PostFormatPreferencesUpdate, ComprehensionPreferences
from src.core.constants import (
    MEMORY_KIND_COMPREHENSION_FEEDBACK,
    MEMORY_KIND_PAPER_FEEDBACK,
    MEMORY_KIND_PAPER_SELECTION,
    MEMORY_KIND_POST_STYLE_FEEDBACK,
)
from src.core.paths import PROMPTS_DIR
from src.services.logger import get_logger

logger = get_logger(__name__)


async def apply_memory_events(
    store: MemoryStore,
    events: List[MemoryEvent] | List[Dict[str, Any]],
    *,
    approved: bool,
    selected_paper: Optional[Dict[str, Any]],
    human_feedback: Optional[str],
    style_llm: Optional[Runnable[Dict[str, Any], PostFormatPreferencesUpdate]] = None,
    comp_llm: Optional[Runnable[Dict[str, Any], ComprehensionPreferences]] = None,
) -> None:
    """
    Mutate the MemoryStore in-place based on events and state.

    - `style_llm` should be llm.with_structured_output(PostFormatPreferencesUpdate)
    - `comp_llm` should be llm.with_structured_output(ComprehensionPreferences)
    """
    norm_events: List[MemoryEvent] = []
    for ev in events:
        if isinstance(ev, MemoryEvent):
            norm_events.append(ev)
        else:
            try:
                norm_events.append(MemoryEvent(**ev))
            except Exception as exc:
                logger.warning(f"Skipping invalid memory event: {exc}")

    for ev in norm_events:
        if ev.kind not in {
            MEMORY_KIND_PAPER_SELECTION,
            MEMORY_KIND_PAPER_FEEDBACK,
            MEMORY_KIND_POST_STYLE_FEEDBACK,
            MEMORY_KIND_COMPREHENSION_FEEDBACK,
        }:
            logger.warning(f"Unhandled memory event kind: {ev.kind}")

    # 1. Topic preferences: liked topics
    topic_dict = store.topic or {}
    liked_titles = set(topic_dict.get("liked_topics", []))

    if approved and selected_paper:
        title = selected_paper.get("title")
        if title:
            liked_titles.add(title)

    for ev in norm_events:
        if ev.kind == MEMORY_KIND_PAPER_SELECTION and ev.selected_title:
            liked_titles.add(ev.selected_title)

    if liked_titles:
        topic_dict["liked_topics"] = sorted(liked_titles)
        store.topic = topic_dict

    # 2. Paper feedback log
    paper_feedback_log = topic_dict.get("feedback_log", [])
    for ev in norm_events:
        if ev.kind == MEMORY_KIND_PAPER_FEEDBACK and ev.message:
            paper_feedback_log.append(
                {
                    "message": ev.message,
                    "title": ev.current_title,
                    "topic": ev.topic,
                }
            )
    if paper_feedback_log:
        topic_dict["feedback_log"] = paper_feedback_log
        store.topic = topic_dict

    # 3. Style / format preferences
    style_feedback_chunks = [
        ev.message
        for ev in norm_events
        if ev.kind == MEMORY_KIND_POST_STYLE_FEEDBACK and ev.message
    ]
    style_feedback_text: Optional[str] = None

    if human_feedback:
        style_feedback_text = human_feedback
        extras = [
            chunk
            for chunk in style_feedback_chunks
            if chunk and chunk != human_feedback
        ]
        if extras:
            style_feedback_text += "\n" + "\n".join(extras)
    elif style_feedback_chunks:
        style_feedback_text = "\n".join(style_feedback_chunks)

    if style_feedback_text and style_llm is not None:
        try:
            prompt_path = PROMPTS_DIR / "memory_style_prompt.md"
            prompt_text = await asyncio.to_thread(prompt_path.read_text)

            chain = ChatPromptTemplate.from_template(prompt_text) | style_llm
            current_style = store.format or {}

            result = await chain.ainvoke(
                {
                    "feedback": style_feedback_text,
                    "current_style": current_style,
                }
            )
            store.format = result.model_dump()
            logger.info("Updated post format preferences from feedback.")
        except Exception as e:
            logger.error(f"Error updating style preferences: {e}")

    # 4. Comprehension preferences
    comp_feedback_chunks = [
        ev.message
        for ev in norm_events
        if ev.kind == MEMORY_KIND_COMPREHENSION_FEEDBACK and ev.message
    ]
    if comp_feedback_chunks and comp_llm is not None:
        try:
            prompt_path = PROMPTS_DIR / "comprehension_memory_prompt.md"
            prompt_text = await asyncio.to_thread(prompt_path.read_text)

            chain = ChatPromptTemplate.from_template(prompt_text) | comp_llm
            current_comp = store.comp or {}

            result = await chain.ainvoke(
                {
                    "feedback": "\n".join(comp_feedback_chunks),
                    "current_preferences": current_comp,
                }
            )
            store.comp = result.model_dump()
            logger.info("Updated comprehension preferences from feedback.")
        except Exception as e:
            logger.error(f"Error updating comprehension preferences: {e}")
