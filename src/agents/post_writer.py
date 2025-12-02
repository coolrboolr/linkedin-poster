import asyncio
from typing import List, Dict, Any
from src.state import AppState
from src.services.logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from src.config.settings import settings

from src.core.paths import PROMPTS_DIR
from src.core.chat_utils import render_chat_snippet, summarize_revisions

logger = get_logger(__name__)

from langsmith import traceable

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
    chain = ChatPromptTemplate.from_template(prompt_text) | llm
    
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

    inputs = {
        "title": paper['title'],
        "summary": paper['summary'],
        "style": state.memory.get("comprehension_preferences", {}),
        "format": formatting_instructions,
        "previous_draft": previous_draft,
        "latest_instruction": latest_instruction,
        "revision_summary": revision_summary,
        "chat_history": chat_snippet,
    }
    
    try:
        result = await chain.ainvoke(inputs)
        new_draft = result.content
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
