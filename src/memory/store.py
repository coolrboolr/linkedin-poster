import json
import asyncio
from typing import Dict, Any

from src.core.paths import MEMORY_DIR as MEMORY_PATH
from src.memory.models import (
    TopicPreferences,
    PostFormatPreferences,
    ComprehensionPreferences,
)
from src.services.logger import get_logger

logger = get_logger(__name__)

class MemoryStore:
    """
    Manages persistent user preferences for topics, comprehension style, and post formatting.
    """
    def __init__(self):
        self._ensure_memory_dir()
        self.topic = {}
        self.comp = {}
        self.format = {}

    def _ensure_memory_dir(self):
        """Ensures the memory directory exists."""
        if not MEMORY_PATH.exists():
            MEMORY_PATH.mkdir(parents=True, exist_ok=True)
            
    async def load(self):
        """Loads preferences from disk asynchronously."""
        self.topic = await self._load("topic_preferences.json")
        self.comp = await self._load("comprehension_preferences.json")
        self.format = await self._load("post_format_preferences.json")

    async def _load(self, filename: str) -> Dict[str, Any]:
        """Loads a JSON file from the memory directory asynchronously."""
        path = MEMORY_PATH / filename
        
        def _read():
            if not path.exists():
                try:
                    path.write_text("{}")
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Unable to initialize memory file {filename}: {exc}")
                    return {}
                return {}
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.error(f"Corrupted memory file detected; resetting: {filename}")
                try:
                    path.write_text("{}")
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Unable to reset corrupted memory file {filename}: {exc}")
                    return {}
                return {}

        return await asyncio.to_thread(_read)

    async def save(self):
        """Saves all current memory states to disk asynchronously."""
        self._ensure_memory_dir()
        
        def _write():
            try:
                (MEMORY_PATH / "topic_preferences.json").write_text(json.dumps(self.topic, indent=2))
                (MEMORY_PATH / "comprehension_preferences.json").write_text(json.dumps(self.comp, indent=2))
                (MEMORY_PATH / "post_format_preferences.json").write_text(json.dumps(self.format, indent=2))
                logger.info("[MemoryStore] Saving memory", extra={"memory": self.get_all()})
                return True
            except Exception as exc:  # pragma: no cover
                logger.error(f"[MemoryStore] Failed to save memory: {exc}")
                return False

        return await asyncio.to_thread(_write)

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
        """Returns a consolidated, normalized dictionary of all memory."""
        return {
            "topic_preferences": self.topic_model.model_dump(),
            "comprehension_preferences": self.comp_model.model_dump(),
            "post_format_preferences": self.format_model.model_dump(),
        }
