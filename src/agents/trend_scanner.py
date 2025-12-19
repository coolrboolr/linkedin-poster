import asyncio
from src.state import AppState
from src.services.google_trends import GoogleTrendsService
from src.services.logger import get_logger

logger = get_logger(__name__)

from langsmith import traceable

@traceable
async def scan_trending_topics(state: AppState) -> dict:
    """
    Scans for trending topics using Google Trends.
    """
    logger.info("--- NODE: Trend Scanner ---")
    trends_service = GoogleTrendsService()
    
    # Get seed keywords from memory or use defaults
    seeds = state.memory.get("topic_preferences", {}).get("seeds", [])
    if not seeds:
        logger.info("No seeds found in memory. Using defaults.")
        seeds = ["AI", "LLM", "Machine Learning"]
    
    # Use top 3 seeds to get broader trends
    # PyTrends supports up to 5 keywords
    search_seeds = seeds[:3]
    
    logger.info(f"Scanning trends for seeds: {search_seeds}")
    
    logger.info(f"Scanning trends for seeds: {search_seeds}")
    
    # Run async service call which now handles its own thread offloading if needed
    trends = await trends_service.get_trending_topics(keywords=search_seeds)
    
    # Filter out avoided topics
    avoid_list = state.memory.get("topic_preferences", {}).get("avoid", [])
    
    # Lowercase for deduplication and sort for determinism
    trends = sorted(list(set([t.lower() for t in trends])))
    
    if avoid_list:
        filtered_trends = [t for t in trends if not any(avoid.lower() in t.lower() for avoid in avoid_list)]
        if len(filtered_trends) < len(trends):
            logger.info(f"Filtered out {len(trends) - len(filtered_trends)} avoided topics.")
        trends = filtered_trends
    
    return {"trending_keywords": trends}
