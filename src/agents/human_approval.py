import json
from datetime import datetime
from typing import List, Dict, Any
from src.state import AppState
from src.services.logger import get_logger
from langgraph.types import interrupt
from src.core.constants import MEMORY_KIND_POST_STYLE_FEEDBACK
from src.core.chat_utils import render_chat_snippet, summarize_revisions

logger = get_logger(__name__)

from langsmith import traceable

def _normalize_instruction_and_draft(raw_args, fallback_draft: str | None) -> tuple[str | None, str | None]:
    """
    Extract (instruction, edited_draft) from Agent Inbox args.
    - str args: treated as instruction only.
    - dict args: use instruction/comment/feedback keys and optional draft.
    - if only a draft is supplied, fabricate an instruction that explains the edit.
    """
    instruction = None
    edited_draft = None

    if isinstance(raw_args, str):
        instruction = raw_args
        edited_draft = fallback_draft
        return instruction, edited_draft

    if isinstance(raw_args, dict):
        instruction = raw_args.get("instruction") or raw_args.get("comment") or raw_args.get("feedback")
        edited_draft = raw_args.get("draft", fallback_draft)

        if not instruction and edited_draft and edited_draft != (fallback_draft or ""):
            instruction = "User edited the draft directly. Incorporate their changes while polishing."

        if not instruction and raw_args:
            instruction = json.dumps(raw_args)

    return instruction, edited_draft or fallback_draft

@traceable
async def human_approval(state: AppState) -> dict:
    """
    Pauses execution for human approval.
    """
    logger.info("--- NODE: Human Approval ---")

    history_snippet = render_chat_snippet(state.chat_history, max_items=5, default_source="human_approval")
    revision_summary = summarize_revisions(state.revision_history, max_items=3)

    description_lines: List[str] = []
    if history_snippet:
        description_lines.append(history_snippet)
    if revision_summary:
        description_lines.append(f"Recent revisions:\n{revision_summary}")
    description_lines.append(f"Current draft:\n\n{state.post_draft}")
    description_lines.append(
        "Approve this post? You can:\n"
        "- Accept (ready to publish)\n"
        "- Edit (change the draft and/or give instructions)\n"
        "- Respond (ask for changes without editing directly)\n"
        "Selecting 'Ignore' will cancel and end this run."
    )

    payload = {
        "description": "\n\n".join(description_lines),
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "action_request": {
            "action": "human approval - post draft",
            "args": {
                "draft": state.post_draft
            }
        },
    }
    raw = interrupt(payload)
    response = raw[0] if isinstance(raw, (list, tuple)) and raw else raw

    def append_user_chat(message: str | None, source: str) -> List[Dict[str, Any]]:
        if not message:
            return state.chat_history
        return state.chat_history + [
            {"role": "user", "source": source, "message": message}
        ]

    def append_edit_request(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        return state.edit_requests + [entry]

    # Keep interrupt type as the single driver for routing
    if response["type"] == "accept":
        raw_args = response.get("args")
        edited_draft = None
        if isinstance(raw_args, dict):
            edited_draft = raw_args.get("draft")
        feedback_text = raw_args if isinstance(raw_args, str) else (json.dumps(raw_args) if raw_args else None)
        new_chat_history = append_user_chat(feedback_text or "Approved as-is.", "human_approval")
        post_draft = edited_draft or state.post_draft
        post_history = state.post_history
        if edited_draft and edited_draft != state.post_draft:
            post_history = state.post_history + [
                {
                    "origin": "user_accept_edit",
                    "draft": edited_draft,
                    "revision_number": len(state.revision_history),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ]
        return {
            "approved": True,
            "revision_requested": False,
            "human_feedback": feedback_text,
            "chat_history": new_chat_history,
            "post_draft": post_draft,
            "post_history": post_history,
            "memory_events": state.memory_events + [
                {
                    "kind": MEMORY_KIND_POST_STYLE_FEEDBACK,
                    "source": "human_approval",
                    "polarity": "positive",
                    "message": feedback_text,
                    "draft": post_draft,
                }
            ],
        }

    if response["type"] == "edit":
        raw_args = response.get("args")
        instruction, edited_draft = _normalize_instruction_and_draft(raw_args, state.post_draft)
        revision_number = len(state.revision_history) + 1
        revision_entry = {
            "revision_number": revision_number,
            "instruction": instruction or "",
            "draft_before": state.post_draft,
            "draft_after": edited_draft,
            "source": "human_approval",
            "timestamp": datetime.utcnow().isoformat(),
        }
        new_revision_history = state.revision_history + [revision_entry]
        new_post_history = state.post_history + [
            {
                "origin": "user_edit",
                "draft": edited_draft,
                "revision_number": revision_number,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ]
        feedback_text = instruction or edited_draft
        new_chat_history = append_user_chat(feedback_text, "human_approval")
        edit_request_entry = {
            "instruction": instruction or "",
            "draft_before": state.post_draft,
            "draft_after": edited_draft,
            "source": "human_approval",
            "type": "edit",
            "revision_number": revision_number,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return {
            # User edited the draft; treat edits as guidance for regeneration
            "approved": False,
            "revision_requested": True,
            "post_draft": edited_draft,
            "human_feedback": feedback_text,
            "revision_history": new_revision_history,
            "edit_requests": append_edit_request(edit_request_entry),
            "post_history": new_post_history,
            "chat_history": new_chat_history,
            "memory_events": state.memory_events + [
                {
                    "kind": MEMORY_KIND_POST_STYLE_FEEDBACK,
                    "source": "human_approval",
                    "polarity": "adjust",
                    "message": feedback_text,
                    "draft": edited_draft,
                }
            ],
        }

    if response["type"] == "response":
        # User is requesting changes via feedback instructions; route to post_writer
        raw_args = response.get("args")
        feedback = raw_args if isinstance(raw_args, str) else (json.dumps(raw_args) if raw_args else None)
        new_chat_history = append_user_chat(feedback, "human_approval")
        edit_requests = state.edit_requests
        if feedback:
            edit_requests = append_edit_request(
                {
                    "instruction": feedback,
                    "draft_before": state.post_draft,
                    "draft_after": None,
                    "source": "human_approval",
                    "type": "response",
                    "revision_number": len(state.revision_history) + 1,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        return {
            "approved": False,
            "revision_requested": True,
            "human_feedback": feedback,
            "return_to_conversation": False,
            "edit_requests": edit_requests,
            "chat_history": new_chat_history,
            "memory_events": state.memory_events + (
                [
                    {
                        "kind": MEMORY_KIND_POST_STYLE_FEEDBACK,
                        "source": "human_approval",
                        "polarity": "neutral",
                        "message": feedback,
                    }
                ]
                if feedback
                else []
            ),
        }

    if response["type"] == "ignore":
        new_chat_history = append_user_chat("User chose to ignore/exit.", "human_approval")
        return {
            "approved": False,
            "revision_requested": False,
            "exit_requested": True,
            "chat_history": new_chat_history,
        }

    # generic "no" / reject without explicit revise
    return {"approved": False, "revision_requested": False}
