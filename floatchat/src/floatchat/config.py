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
    enable_synthetic_index: bool = False

    # HTTP
    http_timeout: int = 30
    http_max_retries: int = 3
    http_max_connections: int = 20
    http_max_keepalive: int = 10

    # Query limits
    max_profiles_per_query: int = 5
    deployment_mode: str = "GLOBAL"  # Options: "GLOBAL", "INDIA_ONLY"

    # Conversation memory
    conversation_max_turns: int = 10

    # NetCDF cache
    netcdf_cache_ttl_days: int = 7

    # LLM / Ollama
    llm_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b "
    ollama_timeout: float = 60.0
    ollama_classifier_timeout: float = 10.0

    # Scientific Narrator – LLM-driven explanation (Phase 26+)
    sci_narrator_enabled: bool = True
    sci_narrator_model: str = "qwen3:8b"
    sci_narrator_temperature: float = 0.25
    sci_narrator_top_p: float = 0.9
    sci_narrator_timeout: float = 8.0
    sci_narrator_max_retries: int = 1
    sci_narrator_thinking: bool = False  # Qwen3 – disable /no_think for fast narration
    sci_narrator_max_tokens: int = 450
    sci_narrator_max_payload_bytes: int = 4096  # configurable ScientificFacts JSON cap
    sci_narrator_prompt_version: str = "sci_narrator_v1_2026-07"
    sci_narrator_fallback: str = "template"  # template | kb

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = "FLOATCHAT_"
        case_sensitive = False


# Global settings singleton. Override in tests via dependency injection.
settings = Settings()
