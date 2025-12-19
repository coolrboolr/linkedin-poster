import json
import pathlib
import pytest
from unittest.mock import patch

from src.memory import MemoryStore

@pytest.mark.asyncio
async def test_memory_load_save(tmp_path):
    # Patch the MEMORY_PATH in src.memory to use tmp_path
    with patch('src.memory.store.MEMORY_PATH', tmp_path):
        store = MemoryStore()
        await store.load()
        store.topic = {"test": "data"}
        await store.save()
        
        assert (tmp_path / "topic_preferences.json").exists()
        content = json.loads((tmp_path / "topic_preferences.json").read_text())
        assert content == {"test": "data"}


@pytest.mark.asyncio
async def test_memory_save_handles_write_failure(monkeypatch, tmp_path):
    with patch('src.memory.store.MEMORY_PATH', tmp_path):
        store = MemoryStore()
        await store.load()
        store.topic = {"ok": True}

        original_write = pathlib.Path.write_text

        def failing_write(self, *args, **kwargs):
            if self.name == "topic_preferences.json":
                raise OSError("disk full")
            return original_write(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "write_text", failing_write)

        result = await store.save()

        assert result is False
