from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from src.memory.models import MemoryEvent

class AppState(BaseModel):
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
    """
    Represents the state of the LinkedIn Poster application graph.
    """
    # Trend Scanning
    trending_keywords: List[str] = Field(default_factory=list, description="List of trending keywords found.")
    
    # ArXiv Fetching
    paper_candidates: List[Dict[str, Any]] = Field(default_factory=list, description="List of paper metadata dictionaries.")
    selected_paper: Optional[Dict[str, Any]] = Field(None, description="The single paper selected for the post.")
    paper_approved: bool = Field(False, description="Flag indicating if the user has confirmed the paper selection.")
    
    # Conversation & Clarification
    clarification_history: List[str] = Field(default_factory=list, description="History of clarification questions and answers.")
    user_ready: bool = Field(False, description="Flag indicating if the user is satisfied and ready to generate the post.")
    chat_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Shared log of assistant/user messages across conversation and approvals. Entries: {role, source, message}.",
    )
    angle_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Optional list of candidate post angles proposed during the conversation.",
    )
    
    # Post Generation
    post_draft: Optional[str] = Field(None, description="The generated LinkedIn post draft.")
    human_feedback: Optional[str] = Field(None, description="Feedback provided by the human user.")
    approved: bool = Field(False, description="Flag indicating if the post has been approved.")
    revision_requested: bool = Field(False, description="Human requested a rewrite of the current draft.")
    return_to_conversation: bool = Field(
        False,
        description="Flag to hop from execution back into the conversation loop for further discussion.",
    )
    revision_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of revisions: {revision_number, instruction, draft_before, draft_after, source, timestamp}.",
    )
    post_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological list of drafts: {origin, draft, revision_number, timestamp}.",
    )

    # Hard-stop flag set by any agent when the user chooses to exit (e.g., ignore prompts)
    exit_requested: bool = Field(False, description="Signal to terminate the graph on the next router hop.")
    
    # Router hint
    next_step: Optional[str] = None # For router to signal next node
    
    # Memory Reference (Loaded at runtime)
    memory: Dict[str, Any] = Field(default_factory=dict, description="User preferences loaded from memory store.")

    # Transient, per-run collection of memory-worthy events emitted by agents
    memory_events: List[MemoryEvent] = Field(
        default_factory=list,
        description="Structured feedback events to persist in memory_updater.",
    )

    def safe(self):
        """Return a JSON-serializable snapshot of state."""
        return self.model_dump()
