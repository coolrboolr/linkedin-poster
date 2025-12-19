import json
from pathlib import Path
from unittest.mock import patch

from src.memory import MemoryStore


async def test_memory_store_loads_default_files(tmp_path: Path):
    # Seed temp memory directory with current default content
    (tmp_path / "topic_preferences.json").write_text(
        json.dumps(
            {
                "seeds": ["AI", "LLM", "Machine Learning"],
                "avoid": [],
                "liked_topics": [],
                "feedback_log": [],
            }
        )
    )
    (tmp_path / "post_format_preferences.json").write_text(
        json.dumps(
            {
                "length": "short",
                "emojis": True,
                "hashtags": True,
                "max_iterations": 1,
                "cta": "Invite readers to share their takeaways",
            }
        )
    )
    (tmp_path / "comprehension_preferences.json").write_text(
        json.dumps({"level": "intermediate"})
    )

    with patch("src.memory.store.MEMORY_PATH", tmp_path):
        store = MemoryStore()
        await store.load()
        data = store.get_all()

    assert data["topic_preferences"]["seeds"] == ["AI", "LLM", "Machine Learning"]
    assert data["topic_preferences"]["liked_topics"] == []
    assert data["post_format_preferences"]["length"] == "short"
    assert data["post_format_preferences"]["cta"] == "Invite readers to share their takeaways"
    assert data["comprehension_preferences"]["level"] == "intermediate"
