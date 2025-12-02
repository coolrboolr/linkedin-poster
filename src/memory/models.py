from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class TopicPreferences(BaseModel):
    """User topic preferences and feedback for paper selection."""

    seeds: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    liked_topics: List[str] = Field(default_factory=list)
    feedback_log: List[Dict[str, Any]] = Field(default_factory=list)


class PostFormatPreferences(BaseModel):
    """Preferences for how LinkedIn posts should be formatted."""

    length: Literal["short", "medium", "long"] = "medium"
    emojis: bool = True
    hashtags: bool = True
    # Bucket for any extra/custom keys you add over time
    extra: Dict[str, Any] = Field(default_factory=dict)


class PostFormatPreferencesUpdate(PostFormatPreferences):
    """
    Schema for LLM updates to post format preferences.
    Same shape as PostFormatPreferences to keep life simple.
    """

    pass


class ComprehensionPreferences(BaseModel):
    """How explanations should be written / pitched."""

    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    tone: Optional[str] = None
    depth: Optional[str] = None
    examples: Optional[bool] = None
    math_vs_intuition: Optional[str] = None
    jargon: Optional[str] = None


class MemoryEvent(BaseModel):
    """
    Structured event describing something memory-worthy that happened
    during this run.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal[
        "comprehension_feedback",
        "paper_feedback",
        "paper_selection",
        "post_style_feedback",
    ]
    source: str
    message: Optional[str] = None
    topic: Optional[str] = None
    current_title: Optional[str] = None
    selected_title: Optional[str] = None
    previous_title: Optional[str] = None
    polarity: Optional[str] = None
    # Anything else you want to stuff in
    extra: Dict[str, Any] = Field(default_factory=dict)
