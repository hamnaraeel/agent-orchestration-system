"""Data model for an entry in the human approval queue."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import EscalationRequest, HumanDecision


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    NOTIFIED = "notified"  # non-blocking: informational only, already "resolved"


class ChatMessage(BaseModel):
    role: Literal["human", "agent"]
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)


class PendingApproval(BaseModel):
    task_id: str
    escalation: EscalationRequest
    source: str = Field(
        description=(
            "Which trigger caused this escalation, e.g. 'plan_low_confidence', "
            "'plan_sensitive', 'plan_requested', 'specialist_retry', "
            "'review_sensitive', 'review_exhausted'."
        )
    )
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision: HumanDecision | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    resolved_at: datetime | None = None
