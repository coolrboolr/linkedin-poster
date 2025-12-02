from src.memory import MemoryStore
import json
from unittest.mock import patch

def test_memory_load_save(tmp_path):
    # Patch the MEMORY_PATH in src.memory to use tmp_path
    with patch('src.memory.store.MEMORY_PATH', tmp_path):
        store = MemoryStore()
        store.topic = {"test": "data"}
        store.save()
        
        assert (tmp_path / "topic_preferences.json").exists()
        content = json.loads((tmp_path / "topic_preferences.json").read_text())
        assert content == {"test": "data"}
