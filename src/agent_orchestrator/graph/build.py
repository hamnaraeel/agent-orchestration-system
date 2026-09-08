"""Wires the supervisor/specialist/reviewer agents into the LangGraph state
machine described in the Phase 1 build plan:

    intake -> planning -> specialist execution (parallel where possible)
    -> review -> synthesis -> delivery

with conditional edges for specialist retry, reviewer rejection, and human
escalation. Escalation itself is a placeholder node here — the real approval
queue lands in Phase 3 (`human-in-the-loop`); for now it just records an
`EscalationRequest` on the state and the graph ends, so a caller can inspect
`state["escalation"]` and decide what to do.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from ..agents.reviewer import ReviewerAgent
from ..agents.specialists import SpecialistAgent
from ..agents.supervisor import SupervisorAgent
from ..config import settings
from ..schemas import EscalationLevel, EscalationRequest, SpecialistType
from .state import OrchestratorState


def build_graph(
    supervisor: SupervisorAgent,
    reviewer: ReviewerAgent,
    specialists: dict[SpecialistType, SpecialistAgent],
) -> CompiledStateGraph:
    def plan_task(state: OrchestratorState) -> dict:
        plan = supervisor.create_plan(state["task"], state.get("memory_context", ""))
        return {"plan": plan, "completed": {}, "retry_counts": {}, "review_cycles": 0}

    def route_after_plan(state: OrchestratorState) -> str:
        if state["plan"].confidence < settings.plan_confidence_threshold:
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
                        "subtask_id": sid,
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
        result = specialists[subtask.specialist].run(subtask, completed)
        retry_counts = state.get("retry_counts", {})
        next_retry_count = retry_counts.get(subtask_id, 0) + (0 if result.success else 1)
        return {
            "completed": {subtask_id: result},
            "retry_counts": {subtask_id: next_retry_count},
        }

    def review(state: OrchestratorState) -> dict:
        result = reviewer.review(state["task"], state["completed"])
        return {"review": result, "review_cycles": state.get("review_cycles", 0) + 1}

    def route_after_review(state: OrchestratorState) -> str:
        review_result = state["review"]
        if review_result.requires_human_review:
            return "human_escalation"
        if review_result.approved:
            return "synthesize"
        if state.get("review_cycles", 0) >= settings.max_review_cycles:
            return "human_escalation"
        return "apply_review_feedback"

    def apply_review_feedback(state: OrchestratorState) -> dict:
        # Deleting these entries (via the None sentinel) puts them back in the
        # "pending" pool so decide_next_batch re-dispatches them.
        return {"completed": {sid: None for sid in state["review"].subtasks_to_redo}}

    def synthesize(state: OrchestratorState) -> dict:
        final_output = supervisor.synthesize(state["task"], state["completed"])
        return {"final_output": final_output}

    def deliver(_state: OrchestratorState) -> dict:
        return {"status": "completed"}

    def human_escalation(state: OrchestratorState) -> dict:
        if not state.get("completed"):
            level, reason = (
                EscalationLevel.APPROVE_PLAN,
                "Supervisor's plan confidence is below threshold.",
            )
        elif state.get("review") is not None and state["review"].requires_human_review:
            level, reason = (
                EscalationLevel.APPROVE_ACTION,
                "Reviewer flagged this deliverable as sensitive or high-risk.",
            )
        else:
            level, reason = (
                EscalationLevel.APPROVE_ACTION,
                "A specialist could not complete its subtask after the retry limit.",
            )
        escalation = EscalationRequest(
            level=level,
            reason=reason,
            context={
                "task": state.get("task"),
                "plan": state["plan"].model_dump() if state.get("plan") else None,
                "completed": {
                    sid: r.model_dump() for sid, r in state.get("completed", {}).items()
                },
                "review": state["review"].model_dump() if state.get("review") else None,
            },
        )
        return {"escalation": escalation, "status": "escalated"}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan_task", plan_task)
    graph.add_node("route_batch", route_batch)
    graph.add_node("run_specialist", run_specialist)
    graph.add_node("review", review)
    graph.add_node("apply_review_feedback", apply_review_feedback)
    graph.add_node("synthesize", synthesize)
    graph.add_node("deliver", deliver)
    graph.add_node("human_escalation", human_escalation)

    graph.add_edge(START, "plan_task")
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
        ["human_escalation", "synthesize", "apply_review_feedback"],
    )
    graph.add_edge("apply_review_feedback", "route_batch")
    graph.add_edge("synthesize", "deliver")
    graph.add_edge("deliver", END)
    graph.add_edge("human_escalation", END)

    return graph.compile()
