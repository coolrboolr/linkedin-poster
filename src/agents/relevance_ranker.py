import asyncio
from src.state import AppState
from src.services.logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from src.config.settings import settings
from src.core.paths import PROMPTS_DIR

import json
from pydantic import BaseModel

logger = get_logger(__name__)

from langsmith import traceable


class RankingChoice(BaseModel):
    index: int
    rationale: str | None = None

@traceable
async def rank_papers(state: AppState) -> dict:
    """
    Ranks papers and selects the best one using OpenAI.
    """
    logger.info("--- NODE: Relevance Ranker ---")
    
    candidates = state.paper_candidates
    if not candidates:
        logger.warning("No papers to rank.")
        return {"selected_paper": None}
        
    # Load prompt
    prompt_path = PROMPTS_DIR / "ranking_prompt.md"
    prompt_text = await asyncio.to_thread(prompt_path.read_text)
    
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set; cannot call LLM.")
        return {"selected_paper": None}

    llm = init_chat_model(
        settings.llm_model,
        api_key=settings.openai_api_key,
    )
    structured_llm = llm.with_structured_output(RankingChoice, method="function_calling")
    chain = ChatPromptTemplate.from_template(prompt_text) | structured_llm
    
    # Prepare inputs
    # Format papers for the prompt
    papers_str = json.dumps([{ "title": p["title"], "summary": p["summary"][:200] } for p in candidates], indent=2)
    
    # Use memory for interests
    interests = (state.memory or {}).get("topic_preferences", {})
    
    inputs = {
        "topic": state.trending_keywords[0] if state.trending_keywords else "General AI",
        "interests": interests,
        "papers": papers_str
    }
    
    try:
        result = await chain.ainvoke(inputs)
        index = result.index

        if 0 <= index < len(candidates):
            selected_paper = candidates[index]
            logger.info(f"Selected paper index: {index}")
            return {"selected_paper": selected_paper}
        else:
            logger.warning(f"Ranker returned out-of-bounds index: {index}. Defaulting to 0.")
            return {"selected_paper": candidates[0]}
            
    except Exception as e:
        logger.error(f"Error ranking papers: {e}")
        # Fallback to first paper
        return {"selected_paper": candidates[0]} # Fallback
