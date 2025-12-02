import asyncio
import json
from typing import List, Dict, Any
from src.state import AppState
from src.services.logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from src.config.settings import settings
from langgraph.types import interrupt
from src.core.constants import (
    MEMORY_KIND_COMPREHENSION_FEEDBACK,
)

logger = get_logger(__name__)

from src.core.paths import PROMPTS_DIR
from src.core.chat_utils import render_chat_history, summarize_revisions

from langsmith import traceable

@traceable
async def conversation_node(state: AppState) -> dict:
    """
    Manages conversation flow using OpenAI.
    Conversation continues until the human explicitly sends an "accept" action;
    all other responses keep user_ready=False and extend the chat history.
    """
    logger.info("--- NODE: Conversation Agent ---")

    def handle_user_answer(
        user_answer: dict,
        chat_history: List[Dict[str, Any]],
        clarification_history: List[str],
    ) -> dict:
        """Centralize state transitions based on user interaction."""
        answer_type = user_answer.get("type") if user_answer else None

        def append_user_message(message: str | None, source: str = "conversation"):
            nonlocal chat_history, clarification_history
            if not message:
                return
            chat_history = chat_history + [
                {"role": "user", "source": source, "message": message}
            ]
            clarification_history = clarification_history + [f"User: {message}"]

        if answer_type == "response":
            raw_args = user_answer.get("args") if user_answer else None
            feedback = raw_args if isinstance(raw_args, str) else (json.dumps(raw_args) if raw_args else "")
            append_user_message(feedback)
            if feedback:
                memory_event = {
                    "kind": MEMORY_KIND_COMPREHENSION_FEEDBACK,
                    "source": "conversation",
                    "message": feedback,
                    "extra": {"history_tail": clarification_history[-3:]},
                }
                return {
                    "user_ready": False,
                    "clarification_history": clarification_history,
                    "chat_history": chat_history,
                    "memory_events": state.memory_events + [memory_event],
                }

            return {
                "user_ready": False,
                "clarification_history": clarification_history,
                "chat_history": chat_history,
            }

        if answer_type == "accept":
            raw_args = user_answer.get("args") if user_answer else None
            normalized = raw_args if isinstance(raw_args, str) else (json.dumps(raw_args) if raw_args else None)
            append_user_message(normalized)
            confirm_event = None
            if normalized:
                confirm_event = {
                    "kind": MEMORY_KIND_COMPREHENSION_FEEDBACK,
                    "source": "conversation",
                    "message": normalized,
                    "polarity": "confirm",
                }
            patch = {
                "user_ready": True,
                "clarification_history": clarification_history,
                "chat_history": chat_history,
                "human_feedback": None,
            }
            if confirm_event:
                patch["memory_events"] = state.memory_events + [confirm_event]
            return patch

        if answer_type == "ignore":
            append_user_message("User chose to ignore/exit.")
            return {
                "user_ready": False,
                "clarification_history": clarification_history,
                "chat_history": chat_history,
                "exit_requested": True,
            }

        # No input or unknown type – keep conversation alive with updated history
        return {
            "user_ready": False,
            "clarification_history": clarification_history,
            "chat_history": chat_history,
        }

    # Build prompt and LLM once
    prompt_path = PROMPTS_DIR / "clarification_prompt.md"
    prompt_text = await asyncio.to_thread(prompt_path.read_text)

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set; cannot call LLM.")
        return {"user_ready": True}

    llm = init_chat_model(
        settings.llm_model,
        api_key=settings.openai_api_key,
    )
    chain = ChatPromptTemplate.from_template(prompt_text) | llm

    logger.info("Generating clarification question.")

    history_text = render_chat_history(state.chat_history)
    revision_summary = summarize_revisions(state.revision_history)

    inputs = {
        "paper_title": state.selected_paper['title'] if state.selected_paper else "",
        "paper_summary": state.selected_paper['summary'] if state.selected_paper else "",
        "history": history_text,
        "revision_summary": revision_summary,
        "comprehension_level": state.memory.get("comprehension_preferences", {}).get("level", "intermediate"),
        "topic": state.trending_keywords[0] if state.trending_keywords else ("" if state.selected_paper else "Unknown"),
        "preferences": state.memory.get("topic_preferences", {}),
    }

    # Only catch errors from the LLM call; allow interrupts to bubble so the UI can pause/resume.
    try:
        result = await chain.ainvoke(inputs)
    except Exception as e:
        logger.error(f"Error generating clarification question: {e}")
        return {"user_ready": True}  # Fallback to proceed

    question = result.content

    logger.info(f"Clarification question: {question}")

    new_chat_history = state.chat_history + [
        {"role": "assistant", "source": "conversation", "message": question}
    ]
    new_clarification_history = state.clarification_history + [question]

    description_lines = [
        f"{entry.get('role','assistant').capitalize()}[{entry.get('source','conversation')}]: {entry.get('message','')}"
        for entry in new_chat_history
    ]
    description_lines.append("\n---\nSelecting 'Ignore' will cancel this session and end the run.")
    description = "\n".join(description_lines)

    payload = {
        "description": description,
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": True,
        },
        "action_request": {
            "action": "conversational agent - question",
            "args": {
                "question": question
            }
        }
    }

    # Let GraphInterrupt propagate to LangGraph; tests that mock interrupt to return still pass.
    raw = interrupt(payload)
    user_answer = raw[0] if isinstance(raw, (list, tuple)) and raw else raw

    return handle_user_answer(user_answer, new_chat_history, new_clarification_history)
