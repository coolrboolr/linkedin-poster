import asyncio
from typing import Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from unittest.mock import MagicMock

from src.config.settings import settings
from src.core.chat_utils import render_chat_snippet, summarize_revisions
from src.core.paths import PROMPTS_DIR
from src.services.logger import get_logger
from src.state import AppState
from src.tools.research import expand_paper_context, search_web

logger = get_logger(__name__)

TOOLS = [search_web, expand_paper_context]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


def _should_use_tools() -> bool:
    tavily = getattr(settings, "tavily_api_key", None)
    return bool(settings.openai_api_key and isinstance(tavily, str) and tavily.strip())

@traceable
async def write_post(state: AppState) -> dict:
    """
    Generates a LinkedIn post draft.
    """
    logger.info("--- NODE: Post Writer ---")
    
    paper = state.selected_paper
    if not paper:
        return {"post_draft": "Error: No paper selected."}
        
    # Load prompt
    prompt_path = PROMPTS_DIR / "post_format_prompt.md"
    try:
        prompt_text = await asyncio.to_thread(prompt_path.read_text)
    except FileNotFoundError:
        logger.error(f"Prompt file not found at: {prompt_path}")
        return {"post_draft": "Error: Prompt missing."}
    
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set; cannot call LLM.")
        return {"post_draft": "Error: API Key missing."}

    llm = init_chat_model(
        settings.llm_model,
        api_key=settings.openai_api_key,
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    tool_ready = _should_use_tools() and hasattr(llm, "bind_tools") and not isinstance(llm, MagicMock)
    llm_with_tools = llm.bind_tools(TOOLS) if tool_ready else None
    
    # Prepare inputs
    # Extract formatting preferences and convert them into prompt-ready instructions
    format_prefs = state.memory.get("post_format_preferences", {})

    length_pref = format_prefs.get("length", "medium")
    emoji_pref = format_prefs.get("emojis", True)
    hashtag_pref = format_prefs.get("hashtags", True)
    max_variations = format_prefs.get("max_iterations") or format_prefs.get("variations")

    formatting_lines = [f"Length: {length_pref}."]
    formatting_lines.append("Use emojis." if emoji_pref else "Do NOT use emojis.")
    formatting_lines.append("Include relevant hashtags." if hashtag_pref else "Do NOT use hashtags.")

    if max_variations:
        formatting_lines.append(f"Generate {max_variations} variation(s) only.")

    # Pass through any additional custom keys to keep behavior extensible
    for key, value in format_prefs.items():
        if key in {"length", "emojis", "hashtags", "max_iterations", "variations"}:
            continue
        formatting_lines.append(f"{key}: {value}")

    formatting_instructions = "\n".join(formatting_lines)

    previous_draft = state.post_draft or (state.post_history[-1]["draft"] if state.post_history else "")
    latest_instruction = state.human_feedback or ""
    revision_summary = summarize_revisions(state.revision_history)
    chat_snippet = render_chat_snippet(state.chat_history)
    edit_requests = state.edit_requests or []
    edit_request_lines = []
    for idx, req in enumerate(edit_requests, start=1):
        instruction = (req.get("instruction") or "").strip()
        if not instruction:
            continue
        source = (req.get("source") or "").strip()
        req_type = (req.get("type") or "").strip()
        label = "/".join(part for part in (source, req_type) if part)
        prefix = f"{idx}. {label}: " if label else f"{idx}. "
        edit_request_lines.append(f"{prefix}{instruction}")
    all_edit_requests = "\n".join(edit_request_lines) if edit_request_lines else "None."

    inputs = {
        "title": paper['title'],
        "summary": paper['summary'],
        "style": state.memory.get("comprehension_preferences", {}),
        "format": formatting_instructions,
        "previous_draft": previous_draft,
        "latest_instruction": latest_instruction,
        "revision_summary": revision_summary,
        "chat_history": chat_snippet,
        "all_edit_requests": all_edit_requests,
    }
    
    try:
        if tool_ready and llm_with_tools:
            messages = prompt.format_messages(**inputs)
            initial = await llm_with_tools.ainvoke(messages)
            tool_calls = getattr(initial, "tool_calls", None) or []

            if tool_calls:
                tool_messages = []
                for call in tool_calls:
                    tool = TOOL_MAP.get(call.get("name"))
                    if not tool:
                        logger.warning(f"Unknown tool requested: {call.get('name')}")
                        continue
                    try:
                        tool_result = await tool.ainvoke(call.get("args", {}))
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error(f"Tool {tool.name} failed: {exc}")
                        tool_result = f"{tool.name} unavailable."
                    tool_messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=call.get("id"))
                    )

                final = await llm_with_tools.ainvoke(messages + [initial] + tool_messages)
                result_content = final.content
            else:
                result_content = initial.content
        else:
            chain = prompt | llm
            result = await chain.ainvoke(inputs)
            result_content = result.content

        new_draft = result_content
        new_post_history = state.post_history + [
            {
                "origin": "llm",
                "draft": new_draft,
                "revision_number": len(state.revision_history),
            }
        ]
        return {
            "post_draft": new_draft,
            "revision_requested": False,
            "post_history": new_post_history,
        }
    except Exception as e:
        logger.error(f"Error generating post: {e}")
        return {"post_draft": "Error generating post. Please check logs."}
