"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Illustrative placeholder pricing (USD per 1M tokens) -- NOT verified current
# vendor pricing. Override via MODEL_PRICING or edit this table before using
# cost figures for anything real.
_DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input_per_million": 3.0, "output_per_million": 15.0},
    "claude-opus-5": {"input_per_million": 15.0, "output_per_million": 75.0},
    "claude-haiku-4-5-20251001": {"input_per_million": 0.8, "output_per_million": 4.0},
    "gpt-5": {"input_per_million": 5.0, "output_per_million": 15.0},
    "_default": {"input_per_million": 1.0, "output_per_million": 3.0},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None  # for model names prefixed "groq:", e.g. "groq:openai/gpt-oss-120b"

    # Model routing: which model each agent role uses by default.
    supervisor_model: str = "claude-sonnet-5"
    reviewer_model: str = "claude-sonnet-5"
    specialist_model: str = "gpt-5"

    # Escalation / quality thresholds (Phase 3 wires these to the approval queue).
    plan_confidence_threshold: float = 0.6
    review_score_threshold: float = 0.7
    max_specialist_retries: int = 2
    max_review_cycles: int = 2

    # Sandbox for file + code-execution tools.
    sandbox_workdir: str = "./sandbox"
    code_execution_timeout_seconds: int = 10

    # Short-term working memory (Redis), scoped to a single task run.
    redis_url: str = "redis://localhost:6379/0"
    working_memory_ttl_seconds: int = 3600

    # Long-term semantic memory (ChromaDB). "persistent" writes to a local
    # directory (no server -- the default for local dev/tests); "http" talks
    # to a Chroma server container (used by docker-compose).
    chroma_mode: str = "persistent"
    chroma_persist_dir: str = "./chroma_data"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    memory_collection_name: str = "task_memories"
    memory_top_k: int = 3
    memory_importance_half_life_days: float = 30.0
    memory_min_importance: float = 0.05
    memory_min_age_days_before_expiry: float = 7.0
    memory_consolidation_similarity_threshold: float = 0.93

    # Observability (Phase 4): tracing, cost tracking, persistence for replay.
    trace_db_path: str = "./traces.db"
    checkpoint_db_path: str = "./checkpoints.db"
    otel_service_name: str = "agent-orchestrator"
    otel_console_export: bool = True
    otel_exporter_otlp_endpoint: str | None = None
    model_pricing: dict[str, dict[str, float]] = _DEFAULT_MODEL_PRICING

    # Phase 5: containerized, horizontally-scalable deployment. All optional
    # and off by default so local dev/tests stay simple (SQLite checkpoints,
    # in-process specialist execution).
    postgres_url: str | None = None  # e.g. postgresql://user:pass@host:5432/db
    use_celery_for_specialists: bool = False
    celery_broker_url: str | None = None  # defaults to redis_url if unset
    celery_result_backend: str | None = None  # defaults to redis_url if unset


settings = Settings()
