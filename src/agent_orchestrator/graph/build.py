"""Wires the supervisor/specialist/reviewer agents into the LangGraph state
machine:

    intake -> planning -> specialist execution (parallel where possible)
    -> review -> synthesis -> delivery

with conditional edges for specialist retry, reviewer rejection, and human
escalation.

Escalation is real, not a placeholder: `human_escalation` calls LangGraph's
`interrupt()`, which suspends the graph (durably, via the configured
checkpointer) until a caller resumes it with `Command(resume=decision)`. The
graph itself has no idea a queue exists -- see `human_loop/approval_queue.py`
and `run.py`'s `run_task` for the runner that bridges an interrupt to a
Redis-backed approval queue a human resolves through the review API/UI.

Escalation triggers map to levels like this:
  - low plan confidence / a sensitive subtask in the plan / an explicit
    human-review request -> APPROVE_PLAN (before any work begins)
  - reviewer flags a deliverable as sensitive/risky -> APPROVE_ACTION
  - reviewer never approves after `max_review_cycles` -> TAKE_OVER
  - a specialist fails the same subtask past `max_specialist_retries` -> TAKE_OVER
  - reviewer approves but the quality score is still below threshold -> NOTIFY
    (non-blocking: logged, execution proceeds automatically)

Whatever the level, a human resolving a *blocking* escalation always chooses
one of approve / reject / modify / take_over (see `schemas.DecisionAction`);
the level is a severity/urgency hint for the UI, not a restriction on which
decisions are available.
"""
from __future__ import annotations

import time
import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from ..agents.reviewer import ReviewerAgent
from ..agents.specialists import SpecialistAgent
from ..agents.supervisor import SupervisorAgent
from ..config import settings
from ..memory.extractor import MemoryExtractorAgent
from ..memory.long_term import LongTermMemory
from ..memory.models import MemoryRecord
from ..memory.working import WorkingMemory
from ..schemas import (
    DecisionAction,
    EscalationLevel,
    EscalationRequest,
    HumanDecision,
    SpecialistType,
    SubtaskResult,
)
from .checkpointing import default_checkpointer
from .state import OrchestratorState

_PLAN_SOURCES = {"plan_low_confidence", "plan_sensitive", "plan_requested"}


def _usage_entry(node_name: str, agent) -> list[dict]:
    """The token usage an agent recorded on its most recent call, as a single-
    entry list ready to append to `trace_events[-1]["usage"]` -- a list so a
    caller can always `+=` it in without a None check."""
    if agent.last_usage is None:
        return []
    return [
        {
            "node": node_name,
            "model": agent.model_name,
            "input_tokens": agent.last_usage.input_tokens,
            "output_tokens": agent.last_usage.output_tokens,
        }
    ]


def _trace_event(
    node: str,
    agent: str,
    status: str,
    started_at: float,
    attributes: dict | None = None,
    usage: list[dict] | None = None,
) -> dict:
    return {
        "node": node,
        "agent": agent,
        "status": status,
        "started_at": started_at,
        "ended_at": time.time(),
        "attributes": attributes or {},
        "usage": usage or [],
    }


def _format_memories(memories: list[MemoryRecord]) -> str:
    blocks = []
    for memory in memories:
        block = f"- Past task: {memory.task}\n  Approach that worked: {memory.approach_summary}"
        if memory.facts:
            block += f"\n  Known facts: {'; '.join(memory.facts)}"
        if memory.preferences:
            block += f"\n  User preferences: {'; '.join(memory.preferences)}"
        blocks.append(block)
    return "\n".join(blocks)


def build_graph(
    supervisor: SupervisorAgent,
    reviewer: ReviewerAgent,
    specialists: dict[SpecialistType, SpecialistAgent],
    working_memory: WorkingMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    memory_extractor: MemoryExtractorAgent | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    use_celery: bool = False,
) -> CompiledStateGraph:
    def intake(state: OrchestratorState) -> dict:
        start = time.time()
        # A caller that wants to correlate its own thread_id (for resuming an
        # interrupted run) with working memory / the approval queue can pass
        # task_id in the initial state; otherwise one is generated here.
        task_id = state.get("task_id") or str(uuid.uuid4())
        user_id = state.get("user_id") or "anonymous"
        memory_context = state.get("memory_context", "")

        memory_hits = 0
        if long_term_memory is not None:
            memories = long_term_memory.query(state["task"], user_id=user_id)
            memory_hits = len(memories)
            if memories:
                memory_context = _format_memories(memories)

        if working_memory is not None:
            working_memory.save(task_id, "task", state["task"])
            working_memory.save(task_id, "user_id", user_id)

        return {
            "task_id": task_id,
            "user_id": user_id,
            "memory_context": memory_context,
            "trace_events": [
                _trace_event(
                    "intake", "system", "success", start,
                    attributes={"memory_hits": memory_hits},
                )
            ],
        }

    def plan_task(state: OrchestratorState) -> dict:
        start = time.time()
        plan = supervisor.create_plan(state["task"], state.get("memory_context", ""))
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], "plan", plan.model_dump())
        return {
            "plan": plan,
            "completed": {},
            "retry_counts": {},
            "review_cycles": 0,
            "trace_events": [
                _trace_event(
                    "plan_task", "supervisor", "success", start,
                    attributes={
                        "confidence": plan.confidence,
                        "subtask_count": len(plan.subtasks),
                        "reasoning": plan.reasoning,
                        "sensitive": any(st.sensitive for st in plan.subtasks),
                        "prompt": supervisor.last_prompt,
                        "response": supervisor.last_response_text,
                    },
                    usage=_usage_entry("plan_task", supervisor),
                )
            ],
        }

    def route_after_plan(state: OrchestratorState) -> str:
        plan = state["plan"]
        if (
            plan.confidence < settings.plan_confidence_threshold
            or any(st.sensitive for st in plan.subtasks)
            or state.get("require_human_approval")
        ):
            return "human_escalation"
        return "route_batch"

    def route_batch(_state: OrchestratorState) -> dict:
        return {}

    def decide_next_batch(state: OrchestratorState) -> str | list[Send]:
        plan = state["plan"]
        completed = state.get("completed", {})
        retry_counts = state.get("retry_counts", {})
        by_id = {st.id: st for st in plan.subtasks}

        ready: list[str] = []
        stuck: list[str] = []
        for sid, subtask in by_id.items():
            result = completed.get(sid)
            if result is not None and result.success:
                continue  # already done
            if result is not None and not result.success:
                if retry_counts.get(sid, 0) >= settings.max_specialist_retries:
                    stuck.append(sid)
                    continue
            deps_done = all(
                completed.get(dep) is not None and completed[dep].success
                for dep in subtask.depends_on
            )
            if deps_done:
                ready.append(sid)

        if stuck:
            return "human_escalation"
        if ready:
            return [
                Send(
                    "run_specialist",
                    {
                        "plan": plan,
                        "completed": completed,
                        "retry_counts": retry_counts,
                        "subtask_feedback": state.get("subtask_feedback", {}),
                        "subtask_id": sid,
                        "task_id": state.get("task_id"),
                    },
                )
                for sid in ready
            ]
        # Nothing ready and nothing stuck: either every subtask is done, or
        # some subtasks can never become ready (an unresolvable dependency).
        unresolved = [sid for sid, r in completed.items() if not r.success]
        if unresolved or len(completed) < len(by_id):
            return "human_escalation"
        return "review"

    def run_specialist(state: OrchestratorState) -> dict:
        start = time.time()
        subtask_id = state["subtask_id"]
        plan = state["plan"]
        subtask = next(st for st in plan.subtasks if st.id == subtask_id)
        completed = state.get("completed", {})
        feedback = state.get("subtask_feedback", {}).get(subtask_id)
        specialist = specialists[subtask.specialist]

        if use_celery:
            from ..tasks import run_specialist_task

            payload = run_specialist_task.delay(
                subtask.specialist.value,
                specialist.model_name,
                subtask.model_dump(),
                {sid: r.model_dump() for sid, r in completed.items()},
                feedback,
            ).get(timeout=180)
            result = SubtaskResult.model_validate(payload["subtask_result"])
            prompt, response = payload["prompt"], payload["response"]
            usage_list = (
                [
                    {
                        "node": f"run_specialist:{subtask_id}",
                        "model": payload["model"],
                        **payload["usage"],
                    }
                ]
                if payload["usage"]
                else []
            )
        else:
            result = specialist.run(subtask, completed, feedback=feedback)
            prompt, response = specialist.last_prompt, specialist.last_response_text
            usage_list = _usage_entry(f"run_specialist:{subtask_id}", specialist)

        retry_counts = state.get("retry_counts", {})
        next_retry_count = retry_counts.get(subtask_id, 0) + (0 if result.success else 1)
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], f"subtask:{subtask_id}", result.model_dump())
        return {
            "completed": {subtask_id: result},
            "retry_counts": {subtask_id: next_retry_count},
            "trace_events": [
                _trace_event(
                    f"run_specialist:{subtask_id}",
                    subtask.specialist.value,
                    "success" if result.success else "failure",
                    start,
                    attributes={
                        "subtask_id": subtask_id,
                        "output": result.output,
                        "error": result.error,
                        "retried_with_feedback": feedback is not None,
                        "tool_calls": [c.model_dump() for c in result.tool_calls],
                        "prompt": prompt,
                        "response": response,
                        "executed_via": "celery" if use_celery else "in_process",
                    },
                    usage=usage_list,
                )
            ],
        }

    def review(state: OrchestratorState) -> dict:
        start = time.time()
        result = reviewer.review(state["task"], state["completed"])
        review_cycles = state.get("review_cycles", 0) + 1
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], f"review:{review_cycles}", result.model_dump())
        return {
            "review": result,
            "review_cycles": review_cycles,
            "trace_events": [
                _trace_event(
                    "review",
                    "reviewer",
                    "success" if result.approved else "warning",
                    start,
                    attributes={
                        "score": result.score,
                        "approved": result.approved,
                        "feedback": result.feedback,
                        "subtasks_to_redo": result.subtasks_to_redo,
                        "requires_human_review": result.requires_human_review,
                        "prompt": reviewer.last_prompt,
                        "response": reviewer.last_response_text,
                    },
                    usage=_usage_entry("review", reviewer),
                )
            ],
        }

    def route_after_review(state: OrchestratorState) -> str:
        review_result = state["review"]
        if review_result.requires_human_review:
            return "human_escalation"
        if review_result.approved:
            if review_result.score < settings.review_score_threshold:
                return "notify_human"
            return "synthesize"
        if state.get("review_cycles", 0) >= settings.max_review_cycles:
            return "human_escalation"
        return "apply_review_feedback"

    def apply_review_feedback(state: OrchestratorState) -> dict:
        start = time.time()
        # Deleting these entries (via the None sentinel) puts them back in the
        # "pending" pool so decide_next_batch re-dispatches them, this time
        # with the reviewer's feedback attached.
        redo_ids = state["review"].subtasks_to_redo
        return {
            "completed": {sid: None for sid in redo_ids},
            "subtask_feedback": {sid: state["review"].feedback for sid in redo_ids},
            "trace_events": [
                _trace_event(
                    "apply_review_feedback", "system", "warning", start,
                    attributes={"redo_ids": redo_ids},
                )
            ],
        }

    def notify_human(state: OrchestratorState) -> dict:
        start = time.time()
        review_result = state["review"]
        escalation = EscalationRequest(
            level=EscalationLevel.NOTIFY,
            reason=(
                f"Reviewer approved with a quality score ({review_result.score:.2f}) "
                f"below the notify threshold ({settings.review_score_threshold})."
            ),
            context={"task": state.get("task"), "review": review_result.model_dump()},
        )
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], "notification", escalation.model_dump())
        return {
            "escalation": escalation,
            "escalation_source": "review_low_score_notify",
            "trace_events": [
                _trace_event(
                    "notify_human", "system", "warning", start,
                    attributes={"reason": escalation.reason},
                )
            ],
        }

    def synthesize(state: OrchestratorState) -> dict:
        start = time.time()
        final_output = supervisor.synthesize(state["task"], state["completed"])
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], "final_output", final_output)
        return {
            "final_output": final_output,
            "trace_events": [
                _trace_event(
                    "synthesize", "supervisor", "success", start,
                    attributes={
                        "final_output_preview": final_output[:200],
                        "prompt": supervisor.last_prompt,
                        "response": supervisor.last_response_text,
                    },
                    usage=_usage_entry("synthesize", supervisor),
                )
            ],
        }

    def deliver(state: OrchestratorState) -> dict:
        start = time.time()
        if long_term_memory is not None and memory_extractor is not None:
            record = memory_extractor.extract(
                state.get("user_id", "anonymous"),
                state["task"],
                state["plan"],
                state.get("completed", {}),
                state.get("final_output", ""),
            )
            long_term_memory.add(record)
        if working_memory is not None and state.get("task_id"):
            working_memory.clear(state["task_id"])
        return {
            "status": "completed",
            "trace_events": [_trace_event("deliver", "system", "success", start)],
        }

    def reject_task(state: OrchestratorState) -> dict:
        start = time.time()
        if working_memory is not None and state.get("task_id"):
            working_memory.clear(state["task_id"])
        return {
            "status": "rejected",
            "trace_events": [_trace_event("reject_task", "system", "failure", start)],
        }

    def human_escalation(state: OrchestratorState) -> dict:
        # Note on timing: this function re-runs from the top on resume (that's
        # how `interrupt()` replay works), so `start` here reflects when the
        # resume was processed, not when the pause began -- the real human
        # wait time is tracked separately, from the approval queue's own
        # created_at/resolved_at timestamps (see tracing/recorder.py).
        start = time.time()
        completed = state.get("completed", {})
        review_result = state.get("review")
        stuck_ids: list[str] = []

        if not completed and review_result is None:
            plan = state["plan"]
            if plan.confidence < settings.plan_confidence_threshold:
                level, reason, source = (
                    EscalationLevel.APPROVE_PLAN,
                    "Supervisor's plan confidence is below threshold.",
                    "plan_low_confidence",
                )
            elif any(st.sensitive for st in plan.subtasks):
                level, reason, source = (
                    EscalationLevel.APPROVE_PLAN,
                    "This plan includes a sensitive operation (financial, "
                    "data-destructive, or external communication).",
                    "plan_sensitive",
                )
            else:
                level, reason, source = (
                    EscalationLevel.APPROVE_PLAN,
                    "Human review of the plan was explicitly requested.",
                    "plan_requested",
                )
        elif review_result is not None and review_result.requires_human_review:
            level, reason, source = (
                EscalationLevel.APPROVE_ACTION,
                "Reviewer flagged this deliverable as sensitive or high-risk.",
                "review_sensitive",
            )
        elif review_result is not None:
            level, reason, source = (
                EscalationLevel.TAKE_OVER,
                "Reviewer did not approve the output after "
                f"{state.get('review_cycles', 0)} attempt(s).",
                "review_exhausted",
            )
        else:
            stuck_ids = [
                sid
                for sid, count in state.get("retry_counts", {}).items()
                if count >= settings.max_specialist_retries
                and sid in completed
                and not completed[sid].success
            ]
            level, reason, source = (
                EscalationLevel.TAKE_OVER,
                "A specialist could not complete its subtask after the retry limit.",
                "specialist_retry",
            )

        escalation = EscalationRequest(
            level=level,
            reason=reason,
            context={
                "task": state.get("task"),
                "user_id": state.get("user_id"),
                "plan": state["plan"].model_dump() if state.get("plan") else None,
                "completed": {sid: r.model_dump() for sid, r in completed.items()},
                "review": review_result.model_dump() if review_result else None,
                "memory_context": state.get("memory_context", ""),
                "stuck_subtask_ids": stuck_ids,
            },
        )

        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], f"escalation:{source}", escalation.model_dump())

        decision_payload = interrupt(
            {
                "task_id": state.get("task_id"),
                "source": source,
                "escalation": escalation.model_dump(),
            }
        )
        decision = HumanDecision.model_validate(decision_payload)

        return {
            "escalation": escalation,
            "escalation_source": source,
            "human_decision": decision,
            "trace_events": [
                _trace_event(
                    "human_escalation", "human", "escalated", start,
                    attributes={
                        "level": escalation.level.value,
                        "reason": escalation.reason,
                        "source": source,
                        "decision": decision.action.value,
                    },
                )
            ],
        }

    def apply_human_decision(state: OrchestratorState) -> dict:
        start = time.time()
        decision = state["human_decision"]
        source = state["escalation_source"]
        escalation = state["escalation"]
        updates: dict = {}

        if decision.action == DecisionAction.TAKE_OVER:
            if source == "specialist_retry":
                stuck_ids = escalation.context.get("stuck_subtask_ids", [])
                by_id = {st.id: st for st in state["plan"].subtasks}
                updates["completed"] = {
                    sid: SubtaskResult(
                        subtask_id=sid,
                        specialist=by_id[sid].specialist,
                        success=True,
                        output=decision.output or "",
                        confidence=1.0,
                    )
                    for sid in stuck_ids
                }
            else:
                updates["final_output"] = decision.output or ""
        elif decision.action == DecisionAction.MODIFY:
            guidance = decision.feedback or ""
            if source in _PLAN_SOURCES:
                updates["memory_context"] = (
                    f"{state.get('memory_context', '')}\n\nHuman guidance: {guidance}".strip()
                )
            elif source == "specialist_retry":
                stuck_ids = escalation.context.get("stuck_subtask_ids", [])
                updates["subtask_feedback"] = {sid: guidance for sid in stuck_ids}
                updates["completed"] = {sid: None for sid in stuck_ids}
                updates["retry_counts"] = {sid: 0 for sid in stuck_ids}
            else:
                redo_ids = state["review"].subtasks_to_redo or [
                    st.id for st in state["plan"].subtasks
                ]
                updates["subtask_feedback"] = {sid: guidance for sid in redo_ids}
                updates["completed"] = {sid: None for sid in redo_ids}
        elif decision.action == DecisionAction.APPROVE and source == "specialist_retry":
            stuck_ids = escalation.context.get("stuck_subtask_ids", [])
            updates["completed"] = {sid: None for sid in stuck_ids}
            updates["retry_counts"] = {sid: 0 for sid in stuck_ids}

        updates["trace_events"] = [
            _trace_event(
                "apply_human_decision", "human", "success", start,
                attributes={
                    "action": decision.action.value,
                    "source": source,
                    "feedback": decision.feedback,
                    "output": decision.output,
                },
            )
        ]
        return updates

    def route_after_decision(state: OrchestratorState) -> str:
        decision = state["human_decision"]
        source = state["escalation_source"]

        if decision.action == DecisionAction.REJECT:
            return "reject_task"
        if decision.action == DecisionAction.TAKE_OVER:
            return "route_batch" if source == "specialist_retry" else "deliver"
        if decision.action == DecisionAction.MODIFY:
            if source in _PLAN_SOURCES:
                return "plan_task"
            if source == "specialist_retry":
                return "route_batch"
            return "apply_review_feedback"
        # approve
        if source in _PLAN_SOURCES or source == "specialist_retry":
            return "route_batch"
        return "synthesize"

    graph = StateGraph(OrchestratorState)
    graph.add_node("intake", intake)
    graph.add_node("plan_task", plan_task)
    graph.add_node("route_batch", route_batch)
    graph.add_node("run_specialist", run_specialist)
    graph.add_node("review", review)
    graph.add_node("apply_review_feedback", apply_review_feedback)
    graph.add_node("notify_human", notify_human)
    graph.add_node("synthesize", synthesize)
    graph.add_node("deliver", deliver)
    graph.add_node("reject_task", reject_task)
    graph.add_node("human_escalation", human_escalation)
    graph.add_node("apply_human_decision", apply_human_decision)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan_task")
    graph.add_conditional_edges(
        "plan_task", route_after_plan, ["human_escalation", "route_batch"]
    )
    graph.add_conditional_edges(
        "route_batch", decide_next_batch, ["human_escalation", "review", "run_specialist"]
    )
    graph.add_edge("run_specialist", "route_batch")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        ["human_escalation", "notify_human", "synthesize", "apply_review_feedback"],
    )
    graph.add_edge("apply_review_feedback", "route_batch")
    graph.add_edge("notify_human", "synthesize")
    graph.add_edge("synthesize", "deliver")
    graph.add_edge("deliver", END)
    graph.add_edge("reject_task", END)

    graph.add_edge("human_escalation", "apply_human_decision")
    graph.add_conditional_edges(
        "apply_human_decision",
        route_after_decision,
        ["route_batch", "plan_task", "apply_review_feedback", "synthesize", "deliver", "reject_task"],
    )

    return graph.compile(checkpointer=checkpointer or default_checkpointer())
