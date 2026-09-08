"""Data model for a single long-term memory: one past task, what approach
worked, and what was learned along the way.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExtractedMemory(BaseModel):
    """What the MemoryExtractorAgent pulls out of a finished task."""

    approach_summary: str = Field(
        description="What approach was taken and why it worked (or didn't)."
    )
    facts: list[str] = Field(
        default_factory=list, description="Domain-specific facts discovered while doing the task."
    )
    preferences: list[str] = Field(
        default_factory=list, description="User preferences observed (tone, format, constraints, etc.)."
    )
    importance: float = Field(
        ge=0.0, le=1.0,
        description="How useful this memory is likely to be for future similar tasks.",
    )


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    task: str
    approach_summary: str
    tools_used: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    success: bool = True
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    access_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    last_accessed_at: datetime = Field(default_factory=_utcnow)

    def embedding_text(self) -> str:
        parts = [self.task, self.approach_summary, *self.facts, *self.preferences]
        return "\n".join(p for p in parts if p)
