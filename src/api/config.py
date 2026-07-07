from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "default"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    database_url: str = "sqlite+aiosqlite:///./perf_logs.db"

    tavily_api_key: str = ""

    prompts_path: str = "src/agents/prompts.yaml"
    default_prompt_variant: str = "default"
    default_uncertainty_threshold: float = 0.75
    max_expand_attempts: int = 2
    max_generation_attempts: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
