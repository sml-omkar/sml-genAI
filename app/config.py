"""
Application Configuration
Loads settings from .env file using pydantic-settings.
All config is environment-variable driven for container readiness.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Main application settings.
    Values are loaded from .env file or environment variables.
    """

    # --- Application ---
    APP_NAME: str = "AI-Bot"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = "change-me"

    # --- PostgreSQL ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "aibot_user"
    POSTGRES_PASSWORD: str = "aibot_pass"
    POSTGRES_DB: str = "aibot_db"

    # --- OpenAI ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # --- Embeddings (Ollama local) ---
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIMENSIONS: int = 768

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "policy_chunks"

    # --- JWT ---
    JWT_SECRET_KEY: str = "jwt-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # --- Upload ---
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # --- Microsoft Teams Bot ---
    MicrosoftAppType: str = "MultiTenant"
    MicrosoftAppId: str = ""
    MicrosoftAppPassword: str = ""
    MicrosoftAppTenantId: str = ""

    # --- RAG ---
    CHUNK_SIZE: int = 640
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5
    LLM_TEMPERATURE: float = 0.1

    # --- Memory ---
    CONVERSATION_TTL_HOURS: int = 24
    MEMORY_MAX_MESSAGES: int = 10

    # --- Cache ---
    CACHE_TTL_SECONDS: int = 3600

    @property
    def DATABASE_URL(self) -> str:
        """Build async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """Build sync PostgreSQL connection string (for init scripts)."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Returns the same Settings instance across the app.
    """
    return Settings()
