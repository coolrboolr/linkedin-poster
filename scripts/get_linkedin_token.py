import requests
from src.config.settings import settings
import urllib.parse

def get_linkedin_token():
    client_id = settings.linkedin_client_id
    client_secret = settings.linkedin_client_secret
    redirect_uri = "http://localhost:8000/callback" 
    
    if not client_id or not client_secret:
        print("Error: LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET must be set in .env")
        return

    # 1. Generate Auth URL
    # Retry Legacy Scopes
    scopes = "r_liteprofile r_emailaddress w_member_social"
    encoded_scopes = urllib.parse.quote(scopes)
    encoded_redirect = urllib.parse.quote(redirect_uri)
    
    auth_url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={client_id}&redirect_uri={encoded_redirect}&scope={encoded_scopes}"
    
    print("\n--- LinkedIn OAuth Helper ---\n")
    print(f"1. Ensure your LinkedIn App has '{redirect_uri}' in the 'Authorized Redirect URLs' under the 'Auth' tab.")
    print(f"2. Visit this URL in your browser:\n")
    print(f"{auth_url}\n")
    print("3. You will be redirected to a page (that might fail to load if you don't have a server running, that's fine).")
    print("4. Copy the FULL URL from your browser address bar and paste it here:")
    
    redirected_url = input("\nPaste Redirected URL: ").strip()
    
    # Extract code
    parsed = urllib.parse.urlparse(redirected_url)
    params = urllib.parse.parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    
    if not code:
        print("Error: Could not find 'code' in the URL.")
        return

    # 2. Exchange for Token
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    print("\nExchanging code for access token...")
    response = requests.post(token_url, data=payload)
    
    if response.status_code == 200:
        data = response.json()
        access_token = data.get("access_token")
        expires_in = data.get("expires_in")
        print(f"\nSUCCESS! Here is your Access Token (valid for {expires_in} seconds):\n")
        print(f"{access_token}\n")
        print("Add this to your .env file as:")
        print(f"LINKEDIN_ACCESS_TOKEN={access_token}")
    else:
        print(f"\nError exchanging token: {response.text}")

if __name__ == "__main__":
    get_linkedin_token()
