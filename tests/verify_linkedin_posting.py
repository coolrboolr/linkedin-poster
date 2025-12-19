import asyncio
from src.services.linkedin_api import LinkedInService
from src.config.settings import settings
from src.services.logger import get_logger

logger = get_logger(__name__)

def test_linkedin_posting():
    print("Testing LinkedIn Posting Logic...")
    
    # Check if access token is present
    if settings.linkedin_access_token:
        print(f"Access Token Found: {settings.linkedin_access_token[:5]}...")
    else:
        print("No Access Token Found. Expecting Mock Post.")

    service = LinkedInService(
        client_id=settings.linkedin_client_id or "",
        client_secret=settings.linkedin_client_secret or "",
        access_token=settings.linkedin_access_token or ""
    )

    test_content = "This is a test post to verify the footer integration."
    
    result = service.post_update(test_content)
    
    if result:
        print("SUCCESS: Post method returned True.")
    else:
        print("FAILURE: Post method returned False.")

if __name__ == "__main__":
    test_linkedin_posting()
