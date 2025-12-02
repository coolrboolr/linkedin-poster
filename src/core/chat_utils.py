from typing import List, Dict, Any


def render_chat_snippet(chat_history: List[Dict[str, Any]], max_items: int = 5, default_source: str = "conversation") -> str:
    """
    Render the most recent messages into a short, human-readable snippet.
    """
    if not chat_history:
        return ""
    tail = chat_history[-max_items:]
    lines: List[str] = []
    for entry in tail:
        role = entry.get("role", "assistant").capitalize()
        source = entry.get("source") or default_source
        message = entry.get("message", "")
        prefix = f"{role}[{source}]" if source else role
        lines.append(f"{prefix}: {message}")
    return "\n".join(lines)


def render_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """
    Render the full chat history for LLM consumption.
    """
    return render_chat_snippet(chat_history, max_items=len(chat_history))


def summarize_revisions(revision_history: List[Dict[str, Any]], max_items: int = 3) -> str:
    """
    Compress the tail of the revision history for prompts/UI.
    """
    if not revision_history:
        return ""
    tail = revision_history[-max_items:]
    lines: List[str] = []
    for rev in tail:
        num = rev.get("revision_number")
        instr = rev.get("instruction", "")
        lines.append(f"Rev {num}: {instr}")
    return "\n".join(lines)
