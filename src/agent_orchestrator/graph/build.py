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

import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from .. import schemas
from ..agents.reviewer import ReviewerAgent
from ..agents.specialists import SpecialistAgent
from ..agents.supervisor import SupervisorAgent
from ..config import settings
from ..memory import models as memory_models
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
from .state import OrchestratorState

_PLAN_SOURCES = {"plan_low_confidence", "plan_sensitive", "plan_requested"}


def _msgpack_allowlist_for(*modules) -> list[tuple[str, str]]:
    """Every class defined in `modules`, as (module, name) pairs -- lets the
    checkpointer's msgpack serializer round-trip our schema/memory types
    without the "unregistered type" deprecation warning (and without falling
    back to pickle, which `LANGGRAPH_STRICT_MSGPACK=true` would otherwise
    block for anything not on this list)."""
    return [
        (module.__name__, name)
        for module in modules
        for name, obj in vars(module).items()
        if isinstance(obj, type) and obj.__module__ == module.__name__
    ]


def _default_checkpointer() -> MemorySaver:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=_msgpack_allowlist_for(schemas, memory_models)
    )
    return MemorySaver(serde=serde)


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
) -> CompiledStateGraph:
    def intake(state: OrchestratorState) -> dict:
        # A caller that wants to correlate its own thread_id (for resuming an
        # interrupted run) with working memory / the approval queue can pass
        # task_id in the initial state; otherwise one is generated here.
        task_id = state.get("task_id") or str(uuid.uuid4())
        user_id = state.get("user_id") or "anonymous"
        memory_context = state.get("memory_context", "")

        if long_term_memory is not None:
            memories = long_term_memory.query(state["task"], user_id=user_id)
            if memories:
                memory_context = _format_memories(memories)

        if working_memory is not None:
            working_memory.save(task_id, "task", state["task"])
            working_memory.save(task_id, "user_id", user_id)

        return {"task_id": task_id, "user_id": user_id, "memory_context": memory_context}

    def plan_task(state: OrchestratorState) -> dict:
        plan = supervisor.create_plan(state["task"], state.get("memory_context", ""))
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], "plan", plan.model_dump())
        return {"plan": plan, "completed": {}, "retry_counts": {}, "review_cycles": 0}

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
        subtask_id = state["subtask_id"]
        plan = state["plan"]
        subtask = next(st for st in plan.subtasks if st.id == subtask_id)
        completed = state.get("completed", {})
        feedback = state.get("subtask_feedback", {}).get(subtask_id)
        result = specialists[subtask.specialist].run(subtask, completed, feedback=feedback)
        retry_counts = state.get("retry_counts", {})
        next_retry_count = retry_counts.get(subtask_id, 0) + (0 if result.success else 1)
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], f"subtask:{subtask_id}", result.model_dump())
        return {
            "completed": {subtask_id: result},
            "retry_counts": {subtask_id: next_retry_count},
        }

    def review(state: OrchestratorState) -> dict:
        result = reviewer.review(state["task"], state["completed"])
        review_cycles = state.get("review_cycles", 0) + 1
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], f"review:{review_cycles}", result.model_dump())
        return {"review": result, "review_cycles": review_cycles}

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
        # Deleting these entries (via the None sentinel) puts them back in the
        # "pending" pool so decide_next_batch re-dispatches them, this time
        # with the reviewer's feedback attached.
        redo_ids = state["review"].subtasks_to_redo
        return {
            "completed": {sid: None for sid in redo_ids},
            "subtask_feedback": {sid: state["review"].feedback for sid in redo_ids},
        }

    def notify_human(state: OrchestratorState) -> dict:
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
        return {"escalation": escalation, "escalation_source": "review_low_score_notify"}

    def synthesize(state: OrchestratorState) -> dict:
        final_output = supervisor.synthesize(state["task"], state["completed"])
        if working_memory is not None and state.get("task_id"):
            working_memory.save(state["task_id"], "final_output", final_output)
        return {"final_output": final_output}

    def deliver(state: OrchestratorState) -> dict:
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
        return {"status": "completed"}

    def reject_task(state: OrchestratorState) -> dict:
        if working_memory is not None and state.get("task_id"):
            working_memory.clear(state["task_id"])
        return {"status": "rejected"}

    def human_escalation(state: OrchestratorState) -> dict:
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

        return {"escalation": escalation, "escalation_source": source, "human_decision": decision}

    def apply_human_decision(state: OrchestratorState) -> dict:
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

    return graph.compile(checkpointer=checkpointer or _default_checkpointer())
