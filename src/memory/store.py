import json
from typing import Dict, Any

from src.core.paths import MEMORY_DIR as MEMORY_PATH
from src.memory.models import (
    TopicPreferences,
    PostFormatPreferences,
    ComprehensionPreferences,
)

class MemoryStore:
    """
    Manages persistent user preferences for topics, comprehension style, and post formatting.
    """
    def __init__(self):
        self._ensure_memory_dir()
        self.topic = self._load("topic_preferences.json")
        self.comp = self._load("comprehension_preferences.json")
        self.format = self._load("post_format_preferences.json")

    def _ensure_memory_dir(self):
        """Ensures the memory directory exists."""
        if not MEMORY_PATH.exists():
            MEMORY_PATH.mkdir(parents=True, exist_ok=True)

    def _load(self, filename: str) -> Dict[str, Any]:
        """Loads a JSON file from the memory directory, creating it if it doesn't exist."""
        path = MEMORY_PATH / filename
        if not path.exists():
            path.write_text("{}")
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    def save(self):
        """Saves all current memory states to disk."""
        self._ensure_memory_dir()
        (MEMORY_PATH / "topic_preferences.json").write_text(json.dumps(self.topic, indent=2))
        (MEMORY_PATH / "comprehension_preferences.json").write_text(json.dumps(self.comp, indent=2))
        (MEMORY_PATH / "post_format_preferences.json").write_text(json.dumps(self.format, indent=2))

    @property
    def topic_model(self) -> TopicPreferences:
        return TopicPreferences(**(self.topic or {}))

    @property
    def format_model(self) -> PostFormatPreferences:
        return PostFormatPreferences(**(self.format or {}))

    @property
    def comp_model(self) -> ComprehensionPreferences:
        return ComprehensionPreferences(**(self.comp or {}))

    def get_all(self) -> Dict[str, Any]:
        """Returns a consolidated dictionary of all memory."""
        return {
            "topic_preferences": self.topic,
            "comprehension_preferences": self.comp,
            "post_format_preferences": self.format
        }
