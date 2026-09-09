"""CLI entry point: runs a single task through the full orchestrator graph,
handling any human-in-the-loop escalation along the way, and recording a full
trace (spans, tokens, cost) to `settings.trace_db_path`.

When an escalation queue is available (Redis reachable), a paused run is
pushed there and this process blocks on `ApprovalQueue.wait_for_decision` --
so resolving it from a *separate* process (the review API, or `ui/app.py`'s
Review Queue page) is exactly what unblocks this one. Without a queue
(`--no-memory`), the decision is instead prompted for right here on the
terminal, which is the fastest way to exercise escalation locally.

The graph is checkpointed to `settings.checkpoint_db_path` (not just kept in
memory), so a task's full history survives this process exiting -- which is
what makes `ui/app.py`'s Replay Debugger page possible against a run made
from a previous CLI invocation.
"""
from __future__ import annotations

import argparse
import json
import uuid

from langgraph.types import Command

from .agents.reviewer import ReviewerAgent
from .agents.specialists import build_specialists
from .agents.supervisor import SupervisorAgent
from .config import settings
from .graph.build import build_graph
from .graph.checkpointing import postgres_checkpointer, sqlite_checkpointer
from .human_loop.approval_queue import ApprovalQueue
from .human_loop.models import PendingApproval
from .memory.extractor import MemoryExtractorAgent
from .memory.long_term import LongTermMemory
from .memory.working import WorkingMemory
from .schemas import DecisionAction, EscalationRequest, HumanDecision, SpecialistType
from .tools.builtin import register_builtin_tools
from .tools.registry import ToolRegistry
from .tracing.recorder import run_traced_task
from .tracing.store import TraceStore


def build_app(enable_memory: bool = True):
    registry = ToolRegistry()
    register_builtin_tools(registry)

    specialist_models = {
        specialist_type: settings.specialist_model for specialist_type in SpecialistType
    }
    specialists = build_specialists(registry, specialist_models)
    supervisor = SupervisorAgent(settings.supervisor_model)
    reviewer = ReviewerAgent(settings.reviewer_model)

    working_memory = WorkingMemory.from_url(settings.redis_url) if enable_memory else None
    long_term_memory = LongTermMemory() if enable_memory else None
    memory_extractor = MemoryExtractorAgent(settings.supervisor_model) if enable_memory else None
    approval_queue = ApprovalQueue.from_url(settings.redis_url) if enable_memory else None

    # docker-compose sets POSTGRES_URL so every API/worker container shares
    # one checkpoint store; local/single-machine use falls back to SQLite.
    checkpointer = (
        postgres_checkpointer(settings.postgres_url)
        if settings.postgres_url
        else sqlite_checkpointer(settings.checkpoint_db_path)
    )

    graph = build_graph(
        supervisor,
        reviewer,
        specialists,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        memory_extractor=memory_extractor,
        checkpointer=checkpointer,
        use_celery=settings.use_celery_for_specialists,
    )
    return graph, registry, approval_queue


def _prompt_cli_for_decision(payload: dict) -> HumanDecision:
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


def run_task(
    app,
    task: str,
    user_id: str = "anonymous",
    require_human_approval: bool = False,
    approval_queue: ApprovalQueue | None = None,
) -> dict:
    task_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": task_id}}
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

    if approval_queue is not None and result.get("escalation") and result["escalation"].level.value == "notify":
        approval_queue.notify(task_id, result["escalation"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a task through the agent orchestrator.")
    parser.add_argument("task", help="The task to run.")
    parser.add_argument("--user-id", default="anonymous", help="User id to scope memory to.")
    parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Force human review of the plan before any work begins.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable Redis working memory, ChromaDB long-term memory, and the "
        "approval queue -- escalations are prompted for directly on this terminal.",
    )
    args = parser.parse_args()

    app, registry, approval_queue = build_app(enable_memory=not args.no_memory)
    trace_store = TraceStore(settings.trace_db_path)
    final_state = run_traced_task(
        app,
        args.task,
        trace_store,
        user_id=args.user_id,
        require_human_approval=args.require_approval,
        approval_queue=approval_queue,
    )

    print(
        json.dumps(
            {
                "task_id": final_state.get("task_id"),
                "status": final_state.get("status"),
                "final_output": final_state.get("final_output"),
                "escalation": (
                    final_state["escalation"].model_dump()
                    if final_state.get("escalation")
                    else None
                ),
                "tool_calls": len(registry.call_log),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
