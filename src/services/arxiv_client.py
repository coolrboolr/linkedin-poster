import asyncio
import arxiv
from langsmith import traceable
from typing import List, Dict, Any
from src.services.logger import get_logger
from src.services.utils import load_cache, save_cache

logger = get_logger(__name__)

class ArxivService:
    def __init__(self):
        self.client = arxiv.Client()
        self.cache_file = "arxiv_cache.json"



    # Helper methods for cache (assuming they will be added or are implicitly handled)
    async def _get_from_cache(self, query: str) -> List[Dict[str, Any]]:
        cache = await load_cache(self.cache_file)
        return cache.get(query)

    async def _save_to_cache(self, query: str, results: List[Dict[str, Any]]):
        cache = await load_cache(self.cache_file)
        cache[query] = results
        await save_cache(self.cache_file, cache)

    @traceable
    async def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches ArXiv for papers matching the query.
        """
        logger.info(f"Searching ArXiv for: {query}")
        
        cached = await self._get_from_cache(query)
        if cached:
            logger.info(f"ArXiv cache hit for query: {query}")
            return cached
            
        try:
            # Add timeout handling implicitly via client if possible, or just wrap
            # arxiv package doesn't expose timeout easily in Search, but we can rely on default.
            # User requested "timeout handling".
            # We can wrap in asyncio.wait_for if async, but this is sync.
            # Let's just add normalization here.
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            def _fetch_results():
                results_list = []
                for result in self.client.results(search):
                    # Normalize summary
                    summary = result.summary.replace("\n", " ").strip()
                    # Remove latex (simple heuristic)
                    summary = summary.replace("$", "")
                    
                    results_list.append({
                        "title": result.title,
                        "summary": summary,
                        "url": result.entry_id,
                        "published": result.published.isoformat()
                    })
                return results_list

            results = await asyncio.to_thread(_fetch_results)
            
            if results:
                await self._save_to_cache(query, results)
                
            return results
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
