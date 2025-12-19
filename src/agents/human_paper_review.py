from src.state import AppState
from src.services.logger import get_logger
from langgraph.types import interrupt
import json
import re
from src.core.constants import (
    MEMORY_KIND_PAPER_FEEDBACK,
    MEMORY_KIND_PAPER_SELECTION,
)

logger = get_logger(__name__)

from langsmith import traceable

@traceable
async def human_paper_review(state: AppState) -> dict:
    """
    Pauses execution for human to review and approve the selected paper.
    """
    logger.info("--- NODE: Human Paper Review ---")
    
    selected_paper = state.selected_paper
    if not selected_paper:
        logger.warning("No paper selected to review.")
        return {"paper_approved": False}
    
    
    candidates = [
        {"index": i, "title": p.get("title")}
        for i, p in enumerate(state.paper_candidates)
    ]
    selected_title = selected_paper.get("title", "Unknown Title")
    selected_index = next(
        (c["index"] for c in candidates if c["title"] == selected_title),
        0,
    )
    selected_paper = {
        "title": selected_title,
        "summary": selected_paper.get("summary", "No summary available.")[:500] + "...",
        "url": selected_paper.get("url", "No URL"),
        "published": selected_paper.get("published", "Unknown Date"),
        "index": selected_index,
    }

    def append_user_feedback(message: str | None, source: str = "paper_review"):
        new_chat_history = state.chat_history
        new_clarification_history = state.clarification_history
        if message:
            new_chat_history = new_chat_history + [
                {"role": "user", "source": source, "message": message}
            ]
            new_clarification_history = new_clarification_history + [f"User: {message}"]
        return new_chat_history, new_clarification_history

    # Prepare payload for the user
    payload = {
        "description": f"""Review the selected paper. Approve to proceed, respond to ask questions or refine the research focus, or provide an index to switch.

Selecting 'Ignore' will end the current run.

--- Selected Paper ---
Title: {selected_paper['title']}
Summary: {selected_paper['summary']}
URL: {selected_paper['url']}
Published: {selected_paper['published']}
Index: {selected_paper['index']}

--- Other Candidates ---
""" + "\n".join([f"{c['index']}. {c['title']}" for c in candidates]),
        
        "config": {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            },
        "action_request": {
            "action": "human paper review - selected paper",
            "args": {
                "Selected Paper": f"{selected_paper['index']}. {selected_paper['title']}"
            }
        }
    }
    
    raw = interrupt(payload)
    response = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
    if not response or not isinstance(response, dict):
        logger.warning("No response received from paper review interrupt.")
        return {"paper_approved": False, "user_ready": False}
    feedback = None
    state_memory_events = state.memory_events
    # Keep routing entirely on interrupt type
    if response["type"] == "response":
        raw_args = response.get("args")
        feedback = raw_args if isinstance(raw_args, str) else (json.dumps(raw_args) if raw_args else None)
        new_chat_history, new_clarification_history = append_user_feedback(feedback)
        if feedback:
            memory_event = {
                "kind": MEMORY_KIND_PAPER_FEEDBACK,
                "source": "paper_review",
                "message": feedback,
                "current_title": selected_paper.get("title") if selected_paper else None,
                "topic": state.trending_keywords[0] if state.trending_keywords else None,
            }
            state_memory_events = state_memory_events + [memory_event]
        return {
            "paper_approved": False,
            "user_ready": False,
            "chat_history": new_chat_history,
            "clarification_history": new_clarification_history,
            "memory_events": state_memory_events,
        }
    
    if response["type"] == "accept":
        logger.info("User approved the paper.")
        raw_args = response.get("args")
        feedback = raw_args if isinstance(raw_args, str) else (json.dumps(raw_args) if raw_args else None)
        approval_note = feedback or "Approved selected paper."
        new_chat_history, new_clarification_history = append_user_feedback(approval_note)
        approval_event = {
            "kind": MEMORY_KIND_PAPER_SELECTION,
            "source": "paper_review",
            "selected_title": selected_paper.get("title") if selected_paper else None,
            "topic": state.trending_keywords[0] if state.trending_keywords else None,
            "polarity": "confirm",
        }
        return {
            "paper_approved": True,
            "chat_history": new_chat_history,
            "clarification_history": new_clarification_history,
            "memory_events": state_memory_events + [approval_event],
        }
    
    if response["type"] == "edit":
        idx_raw = response.get("args", {}).get("Selected Paper")
        match = re.match(r'^\d+', idx_raw or "")
        if not match:
            logger.warning(f"User provided an invalid index format: '{idx_raw}'")
            return {"paper_approved": False}

        idx = int(match.group(0))
        if 0 <= idx < len(state.paper_candidates):
            new_paper = state.paper_candidates[idx]
            logger.info(f"User switched paper to index {idx}: {new_paper.get('title')}")
            selection_note = f"Switched paper to: {idx_raw}"
            new_chat_history, new_clarification_history = append_user_feedback(selection_note)
            selection_event = {
                "kind": MEMORY_KIND_PAPER_SELECTION,
                "source": "paper_review",
                "selected_title": new_paper.get("title"),
                "previous_title": state.selected_paper.get("title") if state.selected_paper else None,
                "topic": state.trending_keywords[0] if state.trending_keywords else None,
            }
            return {
                "selected_paper": new_paper,
                # Selecting a specific paper counts as approval
                "paper_approved": True,
                "user_ready": True,
                "chat_history": new_chat_history,
                "clarification_history": new_clarification_history,
                "memory_events": state.memory_events + [selection_event],
            }

        logger.warning(f"User provided invalid index {idx}.")
        return {"paper_approved": False, "user_ready": False}

    if response["type"] == "ignore":
        logger.info("User ignored paper review prompt.")
        new_chat_history, new_clarification_history = append_user_feedback("User chose to ignore/exit.")
        return {
            "paper_approved": False,
            "user_ready": False,
            "exit_requested": True,
            "chat_history": new_chat_history,
            "clarification_history": new_clarification_history,
        }
            
    # Default reject/retry: treat as not ready so we can return to conversation
    logger.info(f"User did not approve. Feedback: {feedback}")
    result = {"paper_approved": False, "user_ready": False}
    if response.get("type") == "response" and feedback:
        result["memory_events"] = state_memory_events
    return result
