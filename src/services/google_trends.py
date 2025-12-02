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
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.cache_file = "trends_cache.json"

    @traceable
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_trending_topics(self, keywords: List[str] = None) -> List[str]:
        """
        Fetches trending topics related to the provided keywords or defaults to ML/AI.
        Uses caching to avoid excessive API calls.
        """
        # Check cache first (12-hour cache)
        now = datetime.datetime.now()
        cache = load_cache(self.cache_file)
        
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
            # This is a simplified usage. Real usage might involve related_queries or trending_searches
            # For stability, we'll try to get related queries for our seed keywords
            self.pytrends.build_payload(kw_list=keywords[:1], timeframe='now 7-d') # Limit to 1 for stability
            related = self.pytrends.related_queries()
            
            for kw in keywords[:1]:
                if related and kw in related and "top" in related[kw]:
                    top = related[kw]['top']
                    if top is not None:
                        trending_list.extend(top['query'].head(5).tolist())
            
            # Fallback if empty or API issues
            if not trending_list:
                logger.warning("No trending data found, using defaults.")
                trending_list = keywords

            # Deduplicate and normalize
            trending_list = list(set([t.lower() for t in trending_list]))
            
            # Update cache
            save_cache(self.cache_file, {"timestamp": now.isoformat(), "topics": trending_list})
            return trending_list

        except Exception as e:
            logger.error(f"Error fetching trends: {e}")
            return keywords # Return seeds on failure
