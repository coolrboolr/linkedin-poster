import asyncio
from pytrends.request import TrendReq
from langsmith import traceable
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from src.services.logger import get_logger
from src.services.utils import load_cache, save_cache
import datetime

logger = get_logger(__name__)

class GoogleTrendsService:
    def __init__(self):
        self.cache_file = "trends_cache.json"

    def _fetch_pytrends_sync(self, keywords: List[str]) -> List[str]:
        """Encapsulated blocking logic for pytrends to run in a thread."""
        trending_list = []
        try:
            # Initialize pytrends here to avoid blocking __init__
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # This is a simplified usage. Real usage might involve related_queries or trending_searches
            # For stability, we'll try to get related queries for our seed keywords
            pytrends.build_payload(kw_list=keywords[:1], timeframe='now 7-d') # Limit to 1 for stability
            related = pytrends.related_queries()
            
            for kw in keywords[:1]:
                if related and kw in related and "top" in related[kw]:
                    top = related[kw]['top']
                    if top is not None:
                        trending_list.extend(top['query'].head(5).tolist())
            return trending_list
        except Exception as e:
            logger.error(f"Error executing synchronous pytrends call: {e}")
            return []

    @traceable
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_trending_topics(self, keywords: List[str] = None) -> List[str]:
        """
        Fetches trending topics related to the provided keywords or defaults to ML/AI.
        Uses caching to avoid excessive API calls.
        """
        # Check cache first (12-hour cache)
        now = datetime.datetime.now()
        cache = await load_cache(self.cache_file)
        
        if cache.get("timestamp"):
            cached_time = datetime.datetime.fromisoformat(cache["timestamp"])
            if (now - cached_time) < datetime.timedelta(hours=12) and cache.get("topics"):
                logger.info("Returning cached trending topics.")
                return cache["topics"]

        logger.info("Fetching new trending topics from Google Trends.")
        
        # Default keywords if none provided
        if not keywords:
            keywords = ["Machine Learning", "Artificial Intelligence", "Generative AI", "LLM"]

        trending_list = []
        try:
            # Run blocking pytrends logic in thread
            fetched = await asyncio.to_thread(self._fetch_pytrends_sync, keywords)
            trending_list.extend(fetched)
            
            # Fallback if empty or API issues
            if not trending_list:
                logger.warning("No trending data found, using defaults.")
                trending_list = keywords

            # Deduplicate and normalize
            trending_list = list(set([t.lower() for t in trending_list]))
            
            # Update cache
            await save_cache(self.cache_file, {"timestamp": now.isoformat(), "topics": trending_list})
            return trending_list

        except Exception as e:
            logger.error(f"Error fetching trends: {e}")
            return keywords # Return seeds on failure
