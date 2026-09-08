"""Shared state for the orchestrator graph.

`completed` and `retry_counts` are updated from parallel `run_specialist`
branches (via `Send`), so they need merge reducers rather than LangGraph's
default overwrite-on-write behavior. Setting a subtask's entry to `None` in an
update (see `apply_review_feedback`) deletes it, which is how a rejected
subtask gets put back into the "pending" pool.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..schemas import EscalationRequest, ExecutionPlan, HumanDecision, ReviewResult, SubtaskResult


def _merge_optional_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


class OrchestratorState(TypedDict, total=False):
    task: str
    user_id: str
    task_id: str
    memory_context: str
    require_human_approval: bool
    plan: ExecutionPlan | None
    completed: Annotated[dict[str, SubtaskResult], _merge_optional_dict]
    retry_counts: Annotated[dict[str, int], _merge_optional_dict]
    subtask_feedback: Annotated[dict[str, str], _merge_optional_dict]
    review: ReviewResult | None
    review_cycles: int
    status: str
    escalation: EscalationRequest | None
    escalation_source: str | None
    human_decision: HumanDecision | None
    final_output: str | None
    errors: Annotated[list[str], operator.add]
    # One entry per completed node (see graph/build.py's `_trace_event`), used
    # by the tracing recorder to persist spans and compute cost/latency --
    # never touched by the nodes' own business logic.
    trace_events: Annotated[list[dict], operator.add]

    # Only present in the per-branch state handed to `run_specialist` via Send.
    subtask_id: str
