"""Drives a task through the graph, turning each node's `trace_events` entry
(see `graph/build.py`) into both a persisted `Span` (for the trace explorer
and cost dashboards) and a real OpenTelemetry span (for external observability
tooling). This is the traced counterpart of `run.run_task` -- same
interrupt/approval-queue handling, plus recording.
"""
from __future__ import annotations

import uuid

from langgraph.types import Command
from opentelemetry.trace import Status, StatusCode

from ..human_loop.approval_queue import ApprovalQueue
from ..human_loop.models import PendingApproval
from ..schemas import EscalationRequest, HumanDecision
from .otel import get_tracer
from .pricing import estimate_cost
from .store import TraceStore

_FAILURE_STATUSES = {"failure", "escalated"}


def _persist_and_export(
    trace_store: TraceStore, tracer, task_id: str, event: dict
) -> None:
    usage = event.get("usage", [])
    input_tokens = sum(u["input_tokens"] for u in usage)
    output_tokens = sum(u["output_tokens"] for u in usage)
    cost_usd = sum(estimate_cost(u["model"], u["input_tokens"], u["output_tokens"]) for u in usage)

    trace_store.add_span(
        task_id,
        node_name=event["node"],
        agent=event["agent"],
        status=event["status"],
        started_at=event["started_at"],
        ended_at=event["ended_at"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        attributes=event["attributes"],
    )
    for call in event["attributes"].get("tool_calls", []):
        trace_store.add_span(
            task_id,
            node_name=f"tool:{call['tool_name']}",
            agent=event["agent"],
            status="success" if call["success"] else "failure",
            started_at=event["ended_at"] - call["latency_ms"] / 1000,
            ended_at=event["ended_at"],
            tool_name=call["tool_name"],
            attributes={"inputs": call["inputs"], "output": call.get("output"), "error": call.get("error")},
        )

    span = tracer.start_span(
        event["node"],
        start_time=int(event["started_at"] * 1e9),
        attributes={
            "task_id": task_id,
            "agent": event["agent"],
            "status": event["status"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        },
    )
    span.set_status(
        Status(StatusCode.ERROR if event["status"] in _FAILURE_STATUSES else StatusCode.OK)
    )
    span.end(end_time=int(event["ended_at"] * 1e9))


def _prompt_cli_for_decision(payload: dict) -> HumanDecision:
    from ..schemas import DecisionAction

    escalation = EscalationRequest(**payload["escalation"])
    print("\n--- HUMAN REVIEW REQUIRED " + "-" * 40)
    print(f"Level:  {escalation.level.value}")
    print(f"Reason: {escalation.reason}")
    print(f"Task:   {escalation.context.get('task')}")
    print("-" * 66)
    action = input("Decision [approve/reject/modify/take_over]: ").strip().lower()
    feedback = None
    output = None
    if action == "modify":
        feedback = input("Guidance for the agents: ").strip()
    elif action == "take_over":
        output = input("Your output: ").strip()
    return HumanDecision(action=DecisionAction(action), feedback=feedback, output=output)


def run_traced_task(
    app,
    task: str,
    trace_store: TraceStore,
    user_id: str = "anonymous",
    require_human_approval: bool = False,
    approval_queue: ApprovalQueue | None = None,
) -> dict:
    task_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": task_id}}
    tracer = get_tracer()

    trace_store.start_task(task_id, user_id, task)

    result = app.invoke(
        {
            "task": task,
            "user_id": user_id,
            "task_id": task_id,
            "memory_context": "",
            "require_human_approval": require_human_approval,
        },
        config,
    )
    recorded = 0

    def _record_new_events(result: dict) -> None:
        nonlocal recorded
        events = result.get("trace_events", [])
        for event in events[recorded:]:
            _persist_and_export(trace_store, tracer, task_id, event)
        recorded = len(events)

    _record_new_events(result)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        if approval_queue is not None:
            approval_queue.submit(
                PendingApproval(
                    task_id=task_id,
                    escalation=EscalationRequest(**payload["escalation"]),
                    source=payload["source"],
                )
            )
            print(f"Escalation pushed to the approval queue (task_id={task_id}). Waiting...")
            decision = approval_queue.wait_for_decision(task_id)
        else:
            decision = _prompt_cli_for_decision(payload)
        result = app.invoke(Command(resume=decision.model_dump()), config)
        _record_new_events(result)

    human_review_ms = 0.0
    if approval_queue is not None:
        approval = approval_queue.get(task_id)
        if approval is not None and approval.resolved_at is not None:
            human_review_ms = (
                approval.resolved_at - approval.created_at
            ).total_seconds() * 1000
    if approval_queue is not None and result.get("escalation") and result["escalation"].level.value == "notify":
        approval_queue.notify(task_id, result["escalation"])

    plan = result.get("plan")
    task_type = (
        "+".join(sorted({st.specialist.value for st in plan.subtasks})) if plan else "unknown"
    )
    escalation_count = sum(
        1 for e in result.get("trace_events", []) if e["node"] == "human_escalation"
    )

    trace_store.finish_task(
        task_id,
        status=result.get("status", "unknown"),
        task_type=task_type,
        human_review_ms=human_review_ms,
        escalation_count=escalation_count,
    )
    return result
