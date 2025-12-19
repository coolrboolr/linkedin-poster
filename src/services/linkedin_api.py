import urllib.parse
import requests
from src.services.logger import get_logger

logger = get_logger(__name__)

class LinkedInService:
    def __init__(self, client_id: str = "", client_secret: str = "", access_token: str = "", author_urn: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.author_urn = author_urn

    def post_update(self, text: str) -> bool:
        """
        Posts an update to LinkedIn.
        """
        # Append footer
        footer = "\n\nbrought to you by langgraph and agent inbox\nhttps://github.com/coolrboolr/linkedin-poster"
        full_text = text + footer

        if not self.access_token:
            logger.warning("LinkedIn access token not set. Skipping actual API call.")
            logger.info(f"--- MOCK LINKEDIN POST ---\n{full_text}\n--------------------------")
            return True

        url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        }
        
        # We need the user's Person URN to post.
        # Check if explicitly provided first
        author_urn = self.author_urn

        # Prefer OpenID Connect userinfo (recommended), with a legacy /me fallback.
        # Method 1: /userinfo (requires openid profile)
        try:
            userinfo_response = requests.get(
                "https://api.linkedin.com/v2/userinfo",
                headers=headers,
            )
            if userinfo_response.status_code == 200:
                subject = userinfo_response.json().get("sub")
                if subject:
                    author_urn = f"urn:li:person:{subject}"
            else:
                 logger.warning(f"Userinfo failed: {userinfo_response.status_code} {userinfo_response.text}")
        except Exception as e:
            logger.warning(f"Userinfo exception: {e}")

        # Method 2: /me (legacy; requires r_liteprofile or r_basicprofile)
        if not author_urn:
            try:
                me_response = requests.get(
                    "https://api.linkedin.com/v2/me",
                    headers=headers,
                )
                if me_response.status_code == 200:
                    member_id = me_response.json().get("id")
                    if member_id:
                        author_urn = f"urn:li:person:{member_id}"
                else:
                    logger.warning(f"Me endpoint failed: {me_response.status_code} {me_response.text}")
            except Exception as e:
                logger.warning(f"Me endpoint exception: {e}")

        if not author_urn:
            logger.error(
                "Failed to fetch LinkedIn profile ID. Ensure token has 'openid profile' (or legacy r_liteprofile) scope."
            )
            return False

        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": full_text
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully posted to LinkedIn: {response.json().get('id')}")
            return True
        except Exception as e:
            logger.error(f"Failed to post to LinkedIn: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
            return False

    def get_oauth_url(self) -> str:
        """Returns the OAuth authorization URL."""
        scopes = urllib.parse.quote("openid profile email w_member_social")
        redirect_uri = urllib.parse.quote("http://localhost:8000/callback")
        return (
            "https://www.linkedin.com/oauth/v2/authorization"
            f"?response_type=code&client_id={self.client_id}&redirect_uri={redirect_uri}&scope={scopes}"
        )
