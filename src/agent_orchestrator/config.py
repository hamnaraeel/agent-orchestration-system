"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

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


settings = Settings()
