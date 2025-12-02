import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from src.core.paths import CACHE_DIR

def get_cache_path(filename: str) -> Path:
    """Returns the path to a cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / filename

def load_cache(filename: str) -> Dict[str, Any]:
    """Loads a JSON cache file."""
    path = get_cache_path(filename)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}

def save_cache(filename: str, data: Dict[str, Any]):
    """Saves data to a JSON cache file using atomic write."""
    path = get_cache_path(filename)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)

def hash_text(text: str) -> str:
    """Returns an MD5 hash of the given text."""
    return hashlib.md5(text.encode()).hexdigest()
