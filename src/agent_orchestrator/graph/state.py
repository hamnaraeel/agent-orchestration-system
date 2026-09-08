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

from ..schemas import EscalationRequest, ExecutionPlan, ReviewResult, SubtaskResult


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
    memory_context: str
    plan: ExecutionPlan | None
    completed: Annotated[dict[str, SubtaskResult], _merge_optional_dict]
    retry_counts: Annotated[dict[str, int], _merge_optional_dict]
    review: ReviewResult | None
    review_cycles: int
    status: str
    escalation: EscalationRequest | None
    final_output: str | None
    errors: Annotated[list[str], operator.add]

    # Only present in the per-branch state handed to `run_specialist` via Send.
    subtask_id: str
