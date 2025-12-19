from datetime import datetime
from src.state import AppState
from src.services.logger import get_logger
from src.services.linkedin_api import LinkedInService
from src.config.settings import settings
from langsmith import traceable

logger = get_logger(__name__)

@traceable
async def publisher_node(state: AppState) -> dict:
    """
    Publishes the approved post to LinkedIn.
    """
    logger.info("--- NODE: Publisher ---")

    if not state.approved:
        logger.warning("Post not approved. Skipping publication.")
        return {}

    if not state.post_draft:
        logger.warning("No post draft to publish.")
        return {}

    # Initialize service with settings
    service = LinkedInService(
        client_id=settings.linkedin_client_id or "",
        client_secret=settings.linkedin_client_secret or "",
        access_token=settings.linkedin_access_token or "",
        author_urn=settings.linkedin_author_urn or ""
    )

    success = await service.post_update(state.post_draft)
    
    if success:
        logger.info("Content published successfully.")
        # We could add a memory event here or update a 'published' flag in state if it existed.
        # For now, we just proceed.
    else:
        logger.error("Failed to publish content.")

    return {}
