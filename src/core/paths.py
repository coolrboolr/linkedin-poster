from pathlib import Path

# Define project root relative to this file: src/core/paths.py -> src/core -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MEMORY_DIR = DATA_DIR / "memory"
PROMPTS_DIR = PROJECT_ROOT / "src" / "config" / "prompts"
