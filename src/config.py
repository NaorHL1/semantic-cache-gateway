"""Gateway settings, loaded from environment variables or `.env` (ADR-0004 §4.13)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"

    cache_schema_version: str = Field("v1", pattern=r"^v\d+$")
    cache_similarity_threshold: float = Field(0.94, gt=0.0, le=1.0)
    cache_ttl_seconds: int = Field(604800, gt=0)
    cache_lookup_timeout_ms: int = Field(50, gt=0)
    cache_allow_tools: bool = True
    cache_debug_headers: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
