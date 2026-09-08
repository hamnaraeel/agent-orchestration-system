"""Minimal HTTP API over long-term memory and the human approval queue:
- a dashboard feed of what the system remembers about a user, plus
  maintenance and data-deletion endpoints.
- the review interface's backend: list/inspect pending escalations, resolve
  them, and ask the paused task's context a clarifying question.

This process only touches the *queue* -- it never holds a reference to a
running graph. Resolving an approval here does not itself resume anything;
whichever process is blocked in `ApprovalQueue.wait_for_decision` (the CLI's
`run_task`, or eventually a Celery worker) notices the resolution and is what
actually calls `Command(resume=...)`. That decoupling is what lets this API
run in a separate process from task execution.

Run with: uvicorn agent_orchestrator.api:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agents.clarifier import ClarificationAgent
from .config import settings
from .human_loop.approval_queue import ApprovalQueue
from .human_loop.models import ChatMessage, PendingApproval
from .memory.long_term import LongTermMemory
from .schemas import HumanDecision

app = FastAPI(title="Agent Orchestrator Memory API")
_memory = LongTermMemory()
_approval_queue = ApprovalQueue.from_url(settings.redis_url)
_clarifier = ClarificationAgent(settings.reviewer_model)


class MemoryDashboardEntry(BaseModel):
    id: str
    task: str
    approach_summary: str
    facts: list[str]
    preferences: list[str]
    tools_used: list[str]
    success: bool
    importance: float
    effective_importance: float
    access_count: int
    created_at: str
    last_accessed_at: str


def _to_entry(record) -> MemoryDashboardEntry:
    return MemoryDashboardEntry(
        id=record.id,
        task=record.task,
        approach_summary=record.approach_summary,
        facts=record.facts,
        preferences=record.preferences,
        tools_used=record.tools_used,
        success=record.success,
        importance=record.importance,
        effective_importance=_memory.effective_importance(record),
        access_count=record.access_count,
        created_at=record.created_at.isoformat(),
        last_accessed_at=record.last_accessed_at.isoformat(),
    )


@app.get("/memory/{user_id}", response_model=list[MemoryDashboardEntry])
def get_memory_dashboard(user_id: str) -> list[MemoryDashboardEntry]:
    entries = [_to_entry(r) for r in _memory.list_user_memories(user_id)]
    entries.sort(key=lambda e: e.effective_importance, reverse=True)
    return entries


@app.delete("/memory/{user_id}")
def delete_user_memory(user_id: str) -> dict:
    """Data-deletion endpoint: wipes every memory the system holds about this user."""
    deleted = _memory.delete_user_memories(user_id)
    return {"user_id": user_id, "deleted_count": deleted}


@app.delete("/memory/{user_id}/{memory_id}")
def delete_single_memory(user_id: str, memory_id: str) -> dict:
    _memory.delete(memory_id)
    return {"user_id": user_id, "memory_id": memory_id, "deleted": True}


@app.post("/memory/{user_id}/consolidate")
def consolidate_user_memory(user_id: str) -> dict:
    merged = _memory.consolidate(user_id)
    return {"user_id": user_id, "merged_count": merged}


@app.post("/memory/expire")
def expire_stale_memory() -> dict:
    expired = _memory.expire()
    return {"expired_count": expired}


class AskRequest(BaseModel):
    question: str


def _get_or_404(task_id: str) -> PendingApproval:
    approval = _approval_queue.get(task_id)
    if approval is None:
        raise HTTPException(status_code=404, detail=f"No approval found for '{task_id}'.")
    return approval


@app.get("/approvals", response_model=list[PendingApproval])
def list_pending_approvals() -> list[PendingApproval]:
    return _approval_queue.list_pending()


@app.get("/approvals/{task_id}", response_model=PendingApproval)
def get_approval(task_id: str) -> PendingApproval:
    return _get_or_404(task_id)


@app.post("/approvals/{task_id}/decide", response_model=PendingApproval)
def decide_approval(task_id: str, decision: HumanDecision) -> PendingApproval:
    _get_or_404(task_id)
    return _approval_queue.resolve(task_id, decision)


@app.post("/approvals/{task_id}/ask", response_model=ChatMessage)
def ask_about_approval(task_id: str, request: AskRequest) -> ChatMessage:
    """Clarifying-question chat: answers from the packaged context only, and
    is recorded on the approval's transcript for the eventual reviewer to see."""
    approval = _get_or_404(task_id)
    _approval_queue.append_message(task_id, "human", request.question)
    answer = _clarifier.answer(approval.escalation.context, request.question)
    updated = _approval_queue.append_message(task_id, "agent", answer)
    return updated.messages[-1]
