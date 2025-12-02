import asyncio
from src.state import AppState
from src.services.arxiv_client import ArxivService
from src.services.logger import get_logger

logger = get_logger(__name__)

from langsmith import traceable

@traceable
async def fetch_arxiv_papers(state: AppState) -> dict:
    """
    Fetches papers from ArXiv based on trending keywords.
    """
    logger.info("--- NODE: ArXiv Fetcher ---")
    arxiv_service = ArxivService()
    
    keywords = state.trending_keywords
    if not keywords:
        logger.warning("No trending keywords found. Using default.")
        keywords = ["Machine Learning"]
        
    # Quote keywords for Arxiv search (e.g. all:"Machine Learning")
    # Arxiv API supports boolean operators AND, OR, ANDNOT
    quoted_keywords = [f'all:"{k.strip().replace(chr(34), "")}"' for k in keywords[:3]]
    query = " OR ".join(quoted_keywords)
    loop = asyncio.get_running_loop()
    papers = await loop.run_in_executor(None, arxiv_service.search_papers, query)
    
    return {"paper_candidates": papers}
