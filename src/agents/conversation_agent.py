import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from langsmith import traceable
from unittest.mock import MagicMock

from src.config.settings import settings
from src.core.chat_utils import render_chat_history, summarize_revisions
from src.core.constants import MEMORY_KIND_COMPREHENSION_FEEDBACK
from src.core.paths import PROMPTS_DIR
from src.services.logger import get_logger
from src.state import AppState
from src.tools.research import expand_paper_context, search_web

logger = get_logger(__name__)

TOOLS = [search_web, expand_paper_context]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


def _parse_conversation_output(text: str) -> tuple[list[str], str]:
    """
    Extract angle suggestions and the clarifying question from model output.
    - Angles: lines starting with 'Angle' (bullet prefixes allowed).
    - Clarifying question: line starting with 'Clarifying question:'; fallback to last non-empty line.
    """
    angles: list[str] = []
    question: str = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet_stripped = stripped.lstrip("-•* ").strip()
        # Also strip simple numeric list prefixes like "1)" or "1."
        bullet_stripped = bullet_stripped.lstrip("0123456789").lstrip("). ").strip()
        if bullet_stripped.lower().startswith("angle"):
            angles.append(bullet_stripped)
        if bullet_stripped.lower().startswith("clarifying question:"):
            question = bullet_stripped.removeprefix("Clarifying question:").strip()

    if not question:
        # Fallback to the last non-empty line
        non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if non_empty:
            question = non_empty[-1]

    return angles, question


def _normalize_user_answer(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"type": "response", "args": raw}
    if isinstance(raw, dict):
        if "type" in raw:
            return raw
        if "response" in raw:
            return {"type": "response", "args": raw.get("response")}
        if "message" in raw:
            return {"type": "response", "args": raw.get("message")}
        if "text" in raw:
            return {"type": "response", "args": raw.get("text")}
        if "args" in raw and isinstance(raw.get("args"), dict):
            nested = raw["args"]
            if "response" in nested:
                return {"type": "response", "args": nested.get("response")}
            if "message" in nested:
                return {"type": "response", "args": nested.get("message")}
            if "text" in nested:
                return {"type": "response", "args": nested.get("text")}
    return None


def _find_unprocessed_user_message(state: AppState) -> str | None:
    if not state.chat_history:
        return None
    last = state.chat_history[-1]
    if last.get("role") != "user":
        return None
    message = (last.get("message") or "").strip()
    if not message:
        return None
    if f"User: {message}" in state.clarification_history:
        return None
    return message

def _should_use_tools() -> bool:
    tavily = getattr(settings, "tavily_api_key", None)
    return bool(settings.openai_api_key and isinstance(tavily, str) and tavily.strip())


async def _invoke_with_tools(
    prompt_text: str,
    inputs: Dict[str, Any],
    llm_with_tools=None,
) -> tuple[str, list[str], str]:
    """
    Run the clarification prompt with tool-calling enabled.
    Returns (assistant_content, angles, question).
    """
    if llm_with_tools is None:
        llm_model = settings.conversation_model or settings.llm_model
        llm = init_chat_model(llm_model, api_key=settings.openai_api_key)
        if not hasattr(llm, "bind_tools"):
            raise RuntimeError("LLM does not support tool binding.")
        llm_with_tools = llm.bind_tools(TOOLS)

    prompt = ChatPromptTemplate.from_template(prompt_text)
    messages = prompt.format_messages(**inputs)

    # First LLM call: may emit tool calls
    initial = await llm_with_tools.ainvoke(messages)
    tool_calls = getattr(initial, "tool_calls", None) or []

    if not tool_calls:
        angles, question = _parse_conversation_output(initial.content)
        return initial.content, angles, question

    tool_messages: List[ToolMessage] = []
    for call in tool_calls:
        tool = TOOL_MAP.get(call.get("name"))
        if not tool:
            logger.warning(f"Unknown tool requested: {call.get('name')}")
            continue
        try:
            result = await tool.ainvoke(call.get("args", {}))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Tool {tool.name} failed: {exc}")
            result = f"{tool.name} unavailable."
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call.get("id"),
            )
        )

    follow_up_messages = messages + [initial] + tool_messages
    final = await llm_with_tools.ainvoke(follow_up_messages)
    angles, question = _parse_conversation_output(final.content)
    return final.content, angles, question


async def _invoke_legacy(prompt_text: str, inputs: Dict[str, Any]) -> tuple[str, list[str], str]:
    """
    Legacy single-call clarification (no tools). Returns (content, angles, question).
    """
    model_name = settings.conversation_model or settings.llm_model
    llm = init_chat_model(model_name, api_key=settings.openai_api_key)
    chain = ChatPromptTemplate.from_template(prompt_text) | llm
    result = await chain.ainvoke(inputs)
    angles, question = _parse_conversation_output(result.content)
    return result.content, angles, question

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
        def _extract_feedback(raw_args: Any) -> str:
            if isinstance(raw_args, str):
                return raw_args
            if isinstance(raw_args, dict):
                for key in ("response", "message", "text", "comment", "feedback", "instruction"):
                    val = raw_args.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
                return json.dumps(raw_args) if raw_args else ""
            return ""

        def append_user_message(message: str | None, source: str = "conversation"):
            nonlocal chat_history, clarification_history
            if not message:
                return
            if chat_history:
                last_entry = chat_history[-1]
                if (
                    last_entry.get("role") == "user"
                    and last_entry.get("message") == message
                    and last_entry.get("source") == source
                ):
                    return
            chat_history = chat_history + [
                {"role": "user", "source": source, "message": message}
            ]
            clarification_history = clarification_history + [f"User: {message}"]

        if answer_type == "response":
            raw_args = user_answer.get("args") if user_answer else None
            feedback = _extract_feedback(raw_args)
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
                    "awaiting_user_response": False,
                    "clarification_history": clarification_history,
                    "chat_history": chat_history,
                    "memory_events": state.memory_events + [memory_event],
                }

            return {
                "user_ready": False,
                "awaiting_user_response": False,
                "clarification_history": clarification_history,
                "chat_history": chat_history,
            }

        if answer_type == "accept":
            raw_args = user_answer.get("args") if user_answer else None
            normalized = _extract_feedback(raw_args) or None
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
                "awaiting_user_response": False,
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
                "awaiting_user_response": False,
                "clarification_history": clarification_history,
                "chat_history": chat_history,
                "exit_requested": True,
            }

        # No input or unknown type – keep conversation alive with updated history
        return {
            "user_ready": False,
            "awaiting_user_response": False,
            "clarification_history": clarification_history,
            "chat_history": chat_history,
        }

    # Build prompt and LLM once
    prompt_path = PROMPTS_DIR / "clarification_prompt.md"
    prompt_text = await asyncio.to_thread(prompt_path.read_text)

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set; cannot call LLM.")
        error_message = (
            "Missing OPENAI_API_KEY. Set it in `.env` and restart the run to continue the research flow."
        )
        payload = {
            "description": error_message,
            "config": {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            },
            "action_request": {
                "action": "conversation configuration error",
                "args": {"message": error_message},
            },
        }
        interrupt(payload)
        return {
            "user_ready": False,
            "exit_requested": True,
            "chat_history": state.chat_history
            + [{"role": "assistant", "source": "conversation", "message": error_message}],
        }

    llm_model = settings.conversation_model or settings.llm_model
    llm = init_chat_model(llm_model, api_key=settings.openai_api_key)
    tool_ready = _should_use_tools() and hasattr(llm, "bind_tools") and not isinstance(llm, MagicMock)
    llm_with_tools = llm.bind_tools(TOOLS) if tool_ready else None

    logger.info("Generating clarification content (tools enabled=%s).", tool_ready)

    history_text = render_chat_history(state.chat_history)
    revision_summary = summarize_revisions(state.revision_history)

    candidates_preview = []
    for idx, candidate in enumerate(state.paper_candidates[:3]):
        title = candidate.get("title", "Untitled")
        summary = (candidate.get("summary") or "").replace("\n", " ").strip()
        if summary:
            summary = summary[:180] + ("..." if len(summary) > 180 else "")
            candidates_preview.append(f"{idx}. {title} — {summary}")
        else:
            candidates_preview.append(f"{idx}. {title}")
    candidates_text = "\n".join(candidates_preview) if candidates_preview else "None provided."

    inputs = {
        "paper_title": state.selected_paper['title'] if state.selected_paper else "",
        "paper_summary": state.selected_paper['summary'] if state.selected_paper else "",
        "paper_candidates": candidates_text,
        "history": history_text,
        "revision_summary": revision_summary,
        "comprehension_level": state.memory.get("comprehension_preferences", {}).get("level", "intermediate"),
        "topic": state.trending_keywords[0] if state.trending_keywords else ("" if state.selected_paper else "Unknown"),
        "preferences": state.memory.get("topic_preferences", {}),
    }

    # Only catch errors from the LLM call; allow interrupts to bubble so the UI can pause/resume.
    try:
        if tool_ready and llm_with_tools:
            assistant_content, angles, question = await _invoke_with_tools(prompt_text, inputs)
        else:
            assistant_content, angles, question = await _invoke_legacy(prompt_text, inputs)
    except Exception as e:
        logger.error(f"Error generating clarification question: {e}")
        error_message = (
            "Error generating the research prompt. Please retry after checking logs or model configuration."
        )
        payload = {
            "description": error_message,
            "config": {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            },
            "action_request": {
                "action": "conversation runtime error",
                "args": {"message": error_message},
            },
        }
        interrupt(payload)
        return {
            "user_ready": False,
            "exit_requested": True,
            "chat_history": state.chat_history
            + [{"role": "assistant", "source": "conversation", "message": error_message}],
        }

    logger.info(f"Clarification question: {question}")

    new_chat_history = state.chat_history + [
        {"role": "assistant", "source": "conversation", "message": assistant_content}
    ]
    new_clarification_history = state.clarification_history + [question]

    description_lines = [
        f"{entry.get('role','assistant').capitalize()}[{entry.get('source','conversation')}]: {entry.get('message','')}"
        for entry in new_chat_history
    ]
    if angles:
        description_lines.append("\nProposed angles:")
        description_lines.extend(angles)
    if candidates_preview:
        description_lines.append("\nCandidate papers:")
        description_lines.extend(candidates_preview)
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
    user_answer = _normalize_user_answer(user_answer)
    if user_answer is None:
        external_message = _find_unprocessed_user_message(state)
        if external_message:
            user_answer = {"type": "response", "args": external_message}
        else:
            return {
                "user_ready": False,
                "awaiting_user_response": True,
                "clarification_history": new_clarification_history,
                "chat_history": new_chat_history,
            }

    updates = handle_user_answer(user_answer, new_chat_history, new_clarification_history)
    if angles:
        updates["angle_suggestions"] = angles
    return updates
