import asyncio
import json
import copy
from src.state import AppState
from src.memory import MemoryStore
from src.services.logger import get_logger
from src.config.settings import settings
from langchain.chat_models import init_chat_model
from src.memory.models import (
    PostFormatPreferencesUpdate,
    ComprehensionPreferences,
)
from src.memory.apply_events import apply_memory_events

logger = get_logger(__name__)

from langsmith import traceable

@traceable
async def update_memory(state: AppState) -> dict:
    """
    Updates persistent memory based on the interaction.
    """
    logger.info("--- NODE: Memory Updater ---")

    if state.exit_requested:
        logger.info("Exit requested; skipping memory writes.")
        return {"memory_events": []}
    
    store = MemoryStore()
    baseline_memory = copy.deepcopy(store.get_all())
    
    events = state.memory_events or []

    # Fast path: nothing to do — return normalized disk memory to avoid stale state merges
    if not events and not (state.approved and state.selected_paper):
        normalized_memory = store.get_all()
        state.memory_events.clear()
        return {"memory": normalized_memory, "memory_events": []}

    style_llm = None
    comp_llm = None

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set; skipping LLM-based memory updates.")
    else:
        base_llm = init_chat_model(
            settings.llm_model,
            api_key=settings.openai_api_key,
        )
        style_llm = base_llm.with_structured_output(PostFormatPreferencesUpdate, method="function_calling")
        comp_llm = base_llm.with_structured_output(ComprehensionPreferences, method="function_calling")

    await apply_memory_events(
        store=store,
        events=events,
        approved=state.approved,
        selected_paper=state.selected_paper,
        human_feedback=state.human_feedback,
        style_llm=style_llm,
        comp_llm=comp_llm,
    )

    await asyncio.to_thread(store.save)

    normalized_memory = store.get_all()

    logger.info(
        "Memory diff after update",
        extra={
            "before": baseline_memory,
            "after": normalized_memory,
        },
    )

    try:
        json.dumps(normalized_memory)
    except Exception as e:
        logger.error(f"Memory not JSON-serializable: {e}")

    logger.info("Memory Updater completed; checkpoint boundary reached.")

    state.memory_events.clear()

    return {"memory": normalized_memory, "memory_events": []}
