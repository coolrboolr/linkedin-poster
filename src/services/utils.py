import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from src.core.paths import CACHE_DIR

import asyncio

def get_cache_path(filename: str) -> Path:
    """Returns the path to a cache file."""
    # This is a fast directory check/create, acceptable to be sync usually, 
    # but strictly speaking mkdir can block. For safety in high performance 
    # contexts, we could wrap it, but it's often negligible. 
    # Let's keep it sync for simplicity unless we want to wrap everything.
    # Actually, the plan said "Use asyncio.to_thread for file I/O inside these functions."
    # We'll leave this helper sync but wrap calls to it if needed, or just wrap the IO.
    # mkdir is usually fast enough, but read/write is the main blocker.
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / filename

async def load_cache(filename: str) -> Dict[str, Any]:
    """Loads a JSON cache file."""
    path = get_cache_path(filename)
    if not path.exists():
        return {}
    
    def _read():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    return await asyncio.to_thread(_read)

async def save_cache(filename: str, data: Dict[str, Any]):
    """Saves data to a JSON cache file using atomic write."""
    path = get_cache_path(filename)
    
    def _write():
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(path)

    await asyncio.to_thread(_write)

def hash_text(text: str) -> str:
    """Returns an MD5 hash of the given text."""
    return hashlib.md5(text.encode()).hexdigest()
