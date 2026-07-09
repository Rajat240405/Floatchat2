"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """FloatChat runtime settings.

    All values can be overridden via environment variables with the prefix
    ``FLOATCHAT_``. For example: ``FLOATCHAT_GDAC_BASE_URL=...``.
    """

    # GDAC / data
    gdac_base_url: str = "https://data-argo.ifremer.fr"
    metadata_index_path: str = "/argo_bio-profile_index.txt.gz"
    metadata_cache_ttl_hours: int = 24

    # HTTP
    http_timeout: int = 30
    http_max_retries: int = 3
    http_max_connections: int = 20
    http_max_keepalive: int = 10

    # Query limits
    max_profiles_per_query: int = 5

    # Conversation memory
    conversation_max_turns: int = 10

    # LLM / Ollama
    llm_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b "
    ollama_timeout: float = 60.0
    ollama_classifier_timeout: float = 10.0

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = "FLOATCHAT_"
        case_sensitive = False


# Global settings singleton. Override in tests via dependency injection.
settings = Settings()
