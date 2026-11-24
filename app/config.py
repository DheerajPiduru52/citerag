"""Application settings, overridable via environment or .env."""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CiteRAG configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://100milionseek.top/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "deepseek-v4.1-flash-free"
    llm_temperature: float = 0.0
    llm_timeout: float = 60.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    vector_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60

    index_path: str = "index"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
