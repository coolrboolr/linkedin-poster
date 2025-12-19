import json
from unittest.mock import patch

import pytest

from src.agents.memory_loader import load_memory
from src.memory.store import MEMORY_PATH
from src.state import AppState


@pytest.mark.asyncio
async def test_load_memory_creates_missing_files(tmp_path):
    with patch("src.memory.store.MEMORY_PATH", tmp_path):
        result = await load_memory(AppState())
        assert result["memory"]["topic_preferences"].get("liked_topics") == []
        assert (tmp_path / "topic_preferences.json").exists()


@pytest.mark.asyncio
async def test_load_memory_recovers_from_corruption(tmp_path):
    bad_file = tmp_path / "topic_preferences.json"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("{not-json")

    with patch("src.memory.store.MEMORY_PATH", tmp_path):
        result = await load_memory(AppState())
        assert result["memory"]["topic_preferences"].get("liked_topics") == []
        assert json.loads(bad_file.read_text()) == {}


@pytest.mark.asyncio
async def test_load_memory_handles_read_only(tmp_path, monkeypatch, caplog):
    with patch("src.memory.store.MEMORY_PATH", tmp_path):
        def fail_write(*_args, **_kwargs):
            raise PermissionError("readonly")

        monkeypatch.setattr("pathlib.Path.write_text", fail_write)

        with caplog.at_level("WARNING"):
            result = await load_memory(AppState())

        # Returns empty defaults despite inability to write seed files
        assert result["memory"]["topic_preferences"].get("liked_topics") == []
