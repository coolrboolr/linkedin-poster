import asyncio
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
    
    events = state.memory_events or []

    # Fast path: nothing to do
    if not events and not (state.approved and state.selected_paper):
        updated_memory = dict(state.memory or {})
        updated_memory.update(store.get_all())
        return {"memory": updated_memory, "memory_events": []}

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

    updated_memory = dict(state.memory or {})
    updated_memory.update(store.get_all())

    return {"memory": updated_memory, "memory_events": []}
