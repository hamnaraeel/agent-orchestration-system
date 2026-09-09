"""End-to-end showcase: a multi-part research task that exercises the full
system live -- supervisor decomposition, specialists (working in parallel
when the plan allows it) actually calling tools, the reviewer evaluating the
result, long-term memory informing a second, similar task, and a human
approving a plan before any work begins on a sensitive-sounding request.

This runs the *real* pipeline against whichever LLMs are configured
(OPENAI_API_KEY / ANTHROPIC_API_KEY) -- it is not scripted or faked, so
exactly which subtasks the supervisor creates, and whether the reviewer
accepts them first try, is genuine model behavior and may vary between runs.

Needs real API keys, and ideally the docker-compose stack up (`docker compose
up -d redis postgres chroma`) so working memory, the approval queue, and
long-term memory are real services rather than local files.

Run with: python demo.py
"""
from __future__ import annotations

import threading
import time

from agent_orchestrator.config import settings
from agent_orchestrator.run import build_app
from agent_orchestrator.schemas import DecisionAction, HumanDecision
from agent_orchestrator.tracing.recorder import run_traced_task
from agent_orchestrator.tracing.store import TraceStore


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _auto_approve_watcher(approval_queue, stop_event: threading.Event) -> None:
    """Stands in for a human reviewer so this script can run unattended: on
    every pending escalation, prints it (what a real reviewer would read) and
    approves it. This still exercises the real approval-queue round trip --
    submit, a separate 'reviewer', resolve, resume -- just without an actual
    person clicking a button.

    No "already handled" set here on purpose: `list_pending()` only ever
    returns entries that are *currently* pending, so once we resolve one it
    naturally stops appearing -- and a task can escalate more than once (e.g.
    a low-confidence plan, then later a stuck specialist), each time with a
    fresh pending entry under the same task_id that must be caught again."""
    while not stop_event.is_set():
        for approval in approval_queue.list_pending():
            _banner("HUMAN REVIEW REQUIRED (auto-deciding for this unattended demo)")
            print(f"Level:  {approval.escalation.level.value}")
            print(f"Reason: {approval.escalation.reason}")
            print("(In production, a human reviews the full packaged context")
            print(" here -- see ui/app.py's Review Queue page -- before deciding.)")
            if approval.source == "specialist_retry":
                # Blindly re-approving a stuck specialist just retries the
                # exact same failing call -- a real reviewer would step in
                # rather than click "approve" forever, so this does too.
                decision = HumanDecision(
                    action=DecisionAction.TAKE_OVER,
                    output="[Provided by a human reviewer standing in for the stuck specialist.]",
                )
            else:
                decision = HumanDecision(action=DecisionAction.APPROVE)
            approval_queue.resolve(approval.task_id, decision)
        time.sleep(0.3)


def _run(app, trace_store, approval_queue, task: str, **kwargs) -> dict:
    print(f"\nTask: {task}")
    result = run_traced_task(app, task, trace_store, approval_queue=approval_queue, **kwargs)
    print(f"Status: {result.get('status')}")
    if result.get("final_output"):
        print(f"Output:\n{result['final_output']}")
    print(f"(task_id: {result.get('task_id')})")
    return result


def main() -> None:
    app, _registry, approval_queue = build_app(enable_memory=True)
    trace_store = TraceStore(settings.trace_db_path)

    stop_event = threading.Event()
    watcher = None
    if approval_queue is not None:
        watcher = threading.Thread(
            target=_auto_approve_watcher, args=(approval_queue, stop_event), daemon=True
        )
        watcher.start()

    try:
        _banner("STEP 1 -- a multi-part research task")
        _run(
            app,
            trace_store,
            approval_queue,
            "Research AcmeCo's Q3 2025 revenue growth, extract the key figures, "
            "and write a two-paragraph summary for an investor update.",
            user_id="demo-investor",
        )

        _banner("STEP 2 -- a similar task: memory should inform planning this time")
        _run(
            app,
            trace_store,
            approval_queue,
            "Research GlobexCorp's Q3 2025 revenue growth, extract the key figures, "
            "and write a two-paragraph summary for an investor update.",
            user_id="demo-investor",
        )
        print(
            "\n(Open ui/app.py's Trace Explorer page and check this task's plan_task "
            "span -- its prompt includes the memory retrieved from step 1.)"
        )

        _banner("STEP 3 -- a sensitive-sounding request: human approval required first")
        _run(
            app,
            trace_store,
            approval_queue,
            "Draft and send a payment confirmation email to a vendor for last "
            "month's invoice.",
            user_id="demo-investor",
            require_human_approval=True,
        )
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=2)

    _banner("DONE")
    print(f"{len(trace_store.list_tasks())} task(s) recorded this run.")
    print("Explore what happened:")
    print("  streamlit run ui/app.py")
    print("    -> Trace Explorer: spans, cost, latency, prompts/responses")
    print("    -> Review Queue: the escalations that were resolved")
    print("    -> Memory Dashboard: what step 1 taught step 2's planning")
    print("    -> Replay Debugger: time-travel through any of these runs")


if __name__ == "__main__":
    main()
