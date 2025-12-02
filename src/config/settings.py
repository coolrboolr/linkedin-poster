from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    linkedin_client_id: Optional[str] = None
    linkedin_client_secret: Optional[str] = None
    llm_model: str = "openai:gpt-4o"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
