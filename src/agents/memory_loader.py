from src.state import AppState
from src.memory import MemoryStore
from src.services.logger import get_logger

logger = get_logger(__name__)

from langsmith import traceable

@traceable
async def load_memory(state: AppState) -> dict:
    """
    Loads persistent memory into the state at the start of execution.
    """
    logger.info("--- NODE: Load Memory ---")
    store = MemoryStore()
    
    # Load all preferences into a dictionary
    memory_snapshot = store.get_all()
    
    logger.info("Memory loaded successfully.")
    return {"memory": memory_snapshot}
