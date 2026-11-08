from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM settings
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    writer_model: str = "gpt-4o"
    temperature: float = 0.1

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    tavily_api_key: str = ""

    # Research Loop & Scraper Constraints
    max_iterations: int = 2
    min_subquestions: int = 3
    max_subquestions: int = 5
    search_results_per_query: int = 3
    scraper_timeout_sec: float = 8.0
    max_scraped_chars: int = 4500

    # Arize Phoenix Observability
    phoenix_enabled: bool = True
    phoenix_project_name: str = "deep-research-rag-agent"
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"


settings = Settings()
