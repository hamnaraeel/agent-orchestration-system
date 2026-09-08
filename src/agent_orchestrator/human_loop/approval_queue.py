"""The human approval queue: a Redis-backed store of pending/resolved
escalations, decoupled from the graph itself.

The graph only knows how to `interrupt()` and wait for a resume value -- it
has no idea a queue exists. A *runner* (see `run.py`'s `run_task`, or a future
Celery worker) is what bridges the two: on interrupt, it submits a
`PendingApproval` here and blocks on `wait_for_decision`; a human resolves it
through the review API/UI by calling `resolve`, which is what unblocks the
runner to resume the graph with `Command(resume=decision)`.
"""
from __future__ import annotations

import time

from ..schemas import EscalationRequest, HumanDecision
from .models import ApprovalStatus, ChatMessage, PendingApproval, _utcnow


class ApprovalQueue:
    _KEY = "approvals"
    _PENDING_SET = "approvals:pending"

    def __init__(self, redis_client, poll_interval: float = 1.0) -> None:
        self.redis = redis_client
        self.poll_interval = poll_interval

    @classmethod
    def from_url(cls, url: str, poll_interval: float = 1.0) -> "ApprovalQueue":
        import redis

        return cls(redis.from_url(url, decode_responses=True), poll_interval=poll_interval)

    def submit(self, approval: PendingApproval) -> None:
        self.redis.hset(self._KEY, approval.task_id, approval.model_dump_json())
        self.redis.sadd(self._PENDING_SET, approval.task_id)

    def get(self, task_id: str) -> PendingApproval | None:
        raw = self.redis.hget(self._KEY, task_id)
        return PendingApproval.model_validate_json(raw) if raw else None

    def list_pending(self) -> list[PendingApproval]:
        ids = self.redis.smembers(self._PENDING_SET)
        approvals = (self.get(task_id) for task_id in ids)
        return [a for a in approvals if a is not None]

    def list_all(self) -> list[PendingApproval]:
        return [
            PendingApproval.model_validate_json(raw)
            for raw in self.redis.hgetall(self._KEY).values()
        ]

    def resolve(self, task_id: str, decision: HumanDecision) -> PendingApproval:
        approval = self.get(task_id)
        if approval is None:
            raise KeyError(f"No pending approval for task '{task_id}'.")
        approval.decision = decision
        approval.status = ApprovalStatus.RESOLVED
        approval.resolved_at = _utcnow()
        self.redis.hset(self._KEY, task_id, approval.model_dump_json())
        self.redis.srem(self._PENDING_SET, task_id)
        return approval

    def notify(self, task_id: str, escalation: EscalationRequest) -> PendingApproval:
        """Record a non-blocking NOTIFY-level escalation; nothing waits on this."""
        approval = PendingApproval(
            task_id=task_id,
            escalation=escalation,
            source="notify",
            status=ApprovalStatus.NOTIFIED,
            resolved_at=_utcnow(),
        )
        self.redis.hset(self._KEY, task_id, approval.model_dump_json())
        return approval

    def append_message(self, task_id: str, role: str, content: str) -> PendingApproval:
        approval = self.get(task_id)
        if approval is None:
            raise KeyError(f"No pending approval for task '{task_id}'.")
        approval.messages.append(ChatMessage(role=role, content=content))
        self.redis.hset(self._KEY, task_id, approval.model_dump_json())
        return approval

    def wait_for_decision(
        self, task_id: str, timeout: float | None = None
    ) -> HumanDecision:
        start = time.monotonic()
        while True:
            approval = self.get(task_id)
            if (
                approval is not None
                and approval.status == ApprovalStatus.RESOLVED
                and approval.decision is not None
            ):
                return approval.decision
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(f"Timed out waiting for a decision on '{task_id}'.")
            time.sleep(self.poll_interval)
