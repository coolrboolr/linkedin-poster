import pytest
from src.config.settings import settings
from src.services.linkedin_api import LinkedInService

def test_linkedin_credentials_present():
    """
    Ensure that LinkedIn client ID and secret are present in the settings.
    This validates that the .env file is correctly loaded and contains the necessary keys.
    """
    assert settings.linkedin_client_id is not None, "LinkedIn Client ID is missing in settings"
    assert settings.linkedin_client_id != "", "LinkedIn Client ID is empty"
    
    assert settings.linkedin_client_secret is not None, "LinkedIn Client Secret is missing in settings"
    assert settings.linkedin_client_secret != "", "LinkedIn Client Secret is empty"

def test_linkedin_service_initialization():
    """
    Ensure that the LinkedInService initializes correctly with the settings.
    """
    service = LinkedInService(
        client_id=settings.linkedin_client_id,
        client_secret=settings.linkedin_client_secret
    )
    
    assert service.client_id == settings.linkedin_client_id
    assert service.client_secret == settings.linkedin_client_secret

def test_oauth_url_generation():
    """
    Ensure that the OAuth URL is generated correctly with the client ID.
    """
    service = LinkedInService(
        client_id=settings.linkedin_client_id,
        client_secret=settings.linkedin_client_secret
    )
    
    url = service.get_oauth_url()
    assert settings.linkedin_client_id in url
    assert "https://www.linkedin.com/oauth/v2/authorization" in url
