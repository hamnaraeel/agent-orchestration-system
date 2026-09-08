"""Minimal HTTP API over long-term memory, the human approval queue, and
observability:
- a dashboard feed of what the system remembers about a user, plus
  maintenance and data-deletion endpoints.
- the review interface's backend: list/inspect pending escalations, resolve
  them, and ask the paused task's context a clarifying question.
- the trace explorer's backend: list/inspect past tasks and their spans, plus
  cost/performance aggregates.
- the replay/time-travel debugger's backend: list a task's checkpoints and
  fork+resume one with a modified value.

The memory/approval endpoints only touch their respective stores -- this
process never holds a reference to a *running* graph, and resolving an
approval here does not itself resume anything; whichever process is blocked
in `ApprovalQueue.wait_for_decision` (the CLI's `run_task`) notices the
resolution and is what actually calls `Command(resume=...)`. The replay
endpoints are the one exception: they need a live graph (see
`_get_replay_app`) because forking a checkpoint and resuming it are graph
operations, not just queue/store reads.

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
from .tracing.replay import list_checkpoints, replay_from_checkpoint
from .tracing.store import TraceStore

app = FastAPI(title="Agent Orchestrator Memory API")
_memory = LongTermMemory()
_approval_queue = ApprovalQueue.from_url(settings.redis_url)
_clarifier = ClarificationAgent(settings.reviewer_model)
_trace_store = TraceStore(settings.trace_db_path)
_replay_app = None


def _get_replay_app():
    """Lazily builds the same graph `run.py` uses, pointed at the same
    persistent checkpoint file, so replay can fork/resume real past runs.
    Lazy so `/traces` and `/analytics/*` work even without LLM credentials."""
    global _replay_app
    if _replay_app is None:
        from .run import build_app

        _replay_app, _, _ = build_app(enable_memory=True)
    return _replay_app


def _jsonable(value):
    """Recursively converts pydantic models (ExecutionPlan, SubtaskResult,
    etc. -- our node functions return these directly into graph state) into
    plain JSON-able data, for endpoints that expose raw checkpoint state."""
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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


# -- trace explorer + cost/performance dashboards -----------------------


@app.get("/traces")
def list_traces(limit: int = 50, user_id: str | None = None) -> list[dict]:
    return [_jsonable(vars(t)) for t in _trace_store.list_tasks(limit=limit, user_id=user_id)]


@app.get("/traces/{task_id}")
def get_trace(task_id: str) -> dict:
    task = _trace_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No trace found for '{task_id}'.")
    spans = _trace_store.get_spans(task_id)
    return {"task": _jsonable(vars(task)), "spans": [_jsonable(vars(s)) for s in spans]}


@app.get("/analytics/cost-by-task-type")
def cost_by_task_type() -> list[dict]:
    return _trace_store.cost_by_task_type()


@app.get("/analytics/expensive-agents")
def expensive_agents(limit: int = 10) -> list[dict]:
    return _trace_store.most_expensive_agents(limit=limit)


@app.get("/analytics/tool-usage")
def tool_usage_patterns() -> list[dict]:
    return _trace_store.tool_usage_patterns()


@app.get("/analytics/escalation-trends")
def escalation_rate_trends() -> list[dict]:
    return _trace_store.escalation_rate_trends()


# -- replay / time-travel debugger ---------------------------------------


@app.get("/traces/{task_id}/checkpoints")
def get_checkpoints(task_id: str) -> list[dict]:
    app_ = _get_replay_app()
    checkpoints = list_checkpoints(app_, task_id)
    if not checkpoints:
        raise HTTPException(status_code=404, detail=f"No checkpoints found for '{task_id}'.")
    return [
        {
            "checkpoint_id": c.checkpoint_id,
            "next_nodes": list(c.next_nodes),
            "values": _jsonable(c.values),
        }
        for c in checkpoints
    ]


class ReplayRequest(BaseModel):
    checkpoint_id: str
    state_update: dict


@app.post("/traces/{task_id}/replay")
def replay_task(task_id: str, request: ReplayRequest) -> dict:
    app_ = _get_replay_app()
    checkpoints = list_checkpoints(app_, task_id)
    checkpoint = next(
        (c for c in checkpoints if c.checkpoint_id == request.checkpoint_id), None
    )
    if checkpoint is None:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint '{request.checkpoint_id}' for task '{task_id}'.",
        )
    result = replay_from_checkpoint(app_, checkpoint, request.state_update)
    return {
        "interrupted": "__interrupt__" in result,
        "state": _jsonable({k: v for k, v in result.items() if k != "__interrupt__"}),
    }
