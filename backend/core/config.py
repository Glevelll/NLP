"""Настройки приложения (pydantic-settings, читаются из переменных окружения и .env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Общие ---
    app_name: str = "Генератор описаний товаров"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    # --- База данных ---
    database_url: str = f"sqlite+aiosqlite:///{(BASE_DIR / 'data' / 'app.db').as_posix()}"
    db_echo: bool = False
    auto_import_data: bool = True
    data_dir: Path = BASE_DIR / "data" / "import"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    use_redis_cache: bool = False
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_always_eager: bool = True

    # --- Кэш ---
    cache_enabled: bool = True
    cache_ttl_hours: int = 72

    # --- Qdrant / эмбеддинги ---
    qdrant_url: str | None = None  # None -> локальный in-memory Qdrant
    qdrant_collection: str = "product_descriptions"
    embedding_backend: str = "sentence-transformers"  # sentence-transformers | hashing
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    # --- RAG ---
    rag_top_k: int = 4
    rrf_k: int = 60
    retrieval_candidates: int = 30

    # --- LLM ---
    llm_default_provider: str = "auto"  # auto | ollama | hf | openai | anthropic | openrouter | groq | mock
    llm_fallback_order: list[str] = ["ollama", "openai", "anthropic", "openrouter", "groq", "hf", "mock"]
    llm_allow_fallback: bool = True
    llm_timeout: float = 180.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 900

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_models: list[str] = ["qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b"]

    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_device: int = -1

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    openrouter_api_key: str | None = None
    openrouter_model: str = "qwen/qwen-2.5-7b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Температуры агентов
    temperature_generator: float = 0.8
    temperature_validator: float = 0.0
    temperature_critic: float = 0.1
    temperature_editor: float = 0.5

    # --- Агенты ---
    max_iterations: int = 3
    critic_threshold: float = 7.0
    uncertain_threshold: float = 6.0
    validator_use_llm: bool = False
    ensemble_models: list[str] = ["ollama:qwen2.5:7b", "ollama:qwen2.5:3b", "mock"]

    # --- Оценка ---
    bertscore_backend: str = "auto"  # auto | bert-score | embedding
    bertscore_model: str = "bert-base-multilingual-cased"
    results_dir: Path = BASE_DIR / "results"

    # --- Безопасность ---
    jwt_secret: str = Field(default="change-me-in-production-please-32b", min_length=16)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # --- Логирование ---
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"
    log_json: bool = False
    log_levels: dict[str, str] = {"httpx": "WARNING", "sqlalchemy.engine": "WARNING", "backend.llm": "INFO"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
