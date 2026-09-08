"""End-to-end smoke tests for the LangGraph state machine, using FakeChatModel
so no LLM provider or API key is needed. These exercise the conditional edges:
low-confidence plan -> escalation -> resume, specialist failure -> retry ->
escalation -> resume, and reviewer rejection -> redo -> re-review -> synthesis.
"""
from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage
from langgraph.types import Command

from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.schemas import (
    DecisionAction,
    ExecutionPlan,
    HumanDecision,
    ReviewResult,
    SpecialistOutput,
    SpecialistType,
    SubTask,
)
from agent_orchestrator.tools.builtin import register_builtin_tools
from agent_orchestrator.tools.registry import ToolRegistry
from tests.fakes import FakeChatModel


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _two_subtask_plan(confidence: float = 0.9, sensitive: bool = False) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Research and write a summary",
        confidence=confidence,
        reasoning="Research first, then write.",
        subtasks=[
            SubTask(
                id="st-1",
                description="Research the topic",
                specialist=SpecialistType.RESEARCH,
                expected_output_format="bullet points",
                estimated_complexity=2,
                sensitive=sensitive,
            ),
            SubTask(
                id="st-2",
                description="Write the summary",
                specialist=SpecialistType.WRITING,
                depends_on=["st-1"],
                expected_output_format="markdown",
                estimated_complexity=2,
            ),
        ],
    )


def _build_test_app(
    *,
    plan_confidence: float,
    specialist_outputs: dict[SpecialistType, SpecialistOutput | list],
    review_results: list[ReviewResult],
    sensitive_plan: bool = False,
):
    registry = ToolRegistry()
    register_builtin_tools(registry)

    supervisor_llm = FakeChatModel(
        structured_response=_two_subtask_plan(plan_confidence, sensitive=sensitive_plan),
        plain_response="Final synthesized answer.",
    )
    supervisor = SupervisorAgent("gpt-5", llm=supervisor_llm)

    reviewer_llm = FakeChatModel()
    reviewer = ReviewerAgent("gpt-5", llm=reviewer_llm)
    review_queue = list(review_results)

    def scripted_review(task, results):
        return review_queue.pop(0)

    reviewer.review = scripted_review  # type: ignore[assignment]

    specialists = {}
    for specialist_type in SpecialistType:
        output = specialist_outputs.get(
            specialist_type, SpecialistOutput(output="ok", confidence=0.9)
        )
        llm = FakeChatModel(structured_response=output, plain_response="")
        specialists[specialist_type] = SpecialistAgent(
            specialist_type, "gpt-5", registry, llm=llm
        )

    app = build_graph(supervisor, reviewer, specialists)
    return app


def test_happy_path_completes_and_synthesizes(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "max_review_cycles", 2)
    app = _build_test_app(
        plan_confidence=0.9,
        specialist_outputs={
            SpecialistType.RESEARCH: SpecialistOutput(output="facts found", confidence=0.9),
            SpecialistType.WRITING: SpecialistOutput(output="summary written", confidence=0.9),
        },
        review_results=[ReviewResult(approved=True, score=0.9, feedback="Looks good.")],
    )

    final_state = app.invoke({"task": "Summarize topic X", "memory_context": ""}, _config())

    assert final_state["status"] == "completed"
    assert final_state["final_output"] == "Final synthesized answer."
    assert final_state["completed"]["st-1"].success is True
    assert final_state["completed"]["st-2"].success is True
    assert "__interrupt__" not in final_state


def test_low_confidence_plan_pauses_then_resumes_on_approve(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "plan_confidence_threshold", 0.6)
    app = _build_test_app(
        plan_confidence=0.3,
        specialist_outputs={
            SpecialistType.RESEARCH: SpecialistOutput(output="facts found", confidence=0.9),
            SpecialistType.WRITING: SpecialistOutput(output="summary written", confidence=0.9),
        },
        review_results=[ReviewResult(approved=True, score=0.9, feedback="Looks good.")],
    )
    cfg = _config()

    paused_state = app.invoke({"task": "Do something risky", "memory_context": ""}, cfg)

    assert "__interrupt__" in paused_state
    interrupt_payload = paused_state["__interrupt__"][0].value
    assert interrupt_payload["source"] == "plan_low_confidence"
    assert interrupt_payload["escalation"]["level"] == "approve_plan"
    assert paused_state.get("completed", {}) == {}

    final_state = app.invoke(
        Command(resume=HumanDecision(action=DecisionAction.APPROVE).model_dump()), cfg
    )

    assert final_state["status"] == "completed"
    assert final_state["completed"]["st-1"].success is True


def test_low_confidence_plan_rejected_stops_the_task(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "plan_confidence_threshold", 0.6)
    app = _build_test_app(plan_confidence=0.3, specialist_outputs={}, review_results=[])
    cfg = _config()

    app.invoke({"task": "Do something risky", "memory_context": ""}, cfg)
    final_state = app.invoke(
        Command(resume=HumanDecision(action=DecisionAction.REJECT).model_dump()), cfg
    )

    assert final_state["status"] == "rejected"
    assert final_state.get("completed", {}) == {}


def test_sensitive_plan_escalates_before_any_work():
    app = _build_test_app(
        plan_confidence=0.9,
        sensitive_plan=True,
        specialist_outputs={
            SpecialistType.RESEARCH: SpecialistOutput(output="facts found", confidence=0.9),
            SpecialistType.WRITING: SpecialistOutput(output="summary written", confidence=0.9),
        },
        review_results=[ReviewResult(approved=True, score=0.9, feedback="Looks good.")],
    )
    cfg = _config()

    paused_state = app.invoke({"task": "Wire money to a vendor", "memory_context": ""}, cfg)

    assert "__interrupt__" in paused_state
    payload = paused_state["__interrupt__"][0].value
    assert payload["source"] == "plan_sensitive"
    assert payload["escalation"]["level"] == "approve_plan"


def test_reviewer_rejection_routes_back_for_redo_then_approves(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "max_review_cycles", 3)
    app = _build_test_app(
        plan_confidence=0.9,
        specialist_outputs={
            SpecialistType.RESEARCH: SpecialistOutput(output="facts found", confidence=0.9),
            SpecialistType.WRITING: SpecialistOutput(output="summary written", confidence=0.9),
        },
        review_results=[
            ReviewResult(
                approved=False,
                score=0.4,
                feedback="Summary is too short.",
                subtasks_to_redo=["st-2"],
            ),
            ReviewResult(approved=True, score=0.85, feedback="Better now."),
        ],
    )

    final_state = app.invoke({"task": "Summarize topic X", "memory_context": ""}, _config())

    assert final_state["status"] == "completed"
    assert final_state["review_cycles"] == 2


def test_reviewer_low_score_notifies_without_blocking(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "review_score_threshold", 0.7)
    app = _build_test_app(
        plan_confidence=0.9,
        specialist_outputs={
            SpecialistType.RESEARCH: SpecialistOutput(output="facts found", confidence=0.9),
            SpecialistType.WRITING: SpecialistOutput(output="summary written", confidence=0.9),
        },
        review_results=[ReviewResult(approved=True, score=0.5, feedback="Passable.")],
    )

    final_state = app.invoke({"task": "Summarize topic X", "memory_context": ""}, _config())

    assert "__interrupt__" not in final_state
    assert final_state["status"] == "completed"
    assert final_state["escalation"].level.value == "notify"


def _build_stuck_specialist_app(max_retries: int):
    registry = ToolRegistry()
    register_builtin_tools(registry)

    supervisor = SupervisorAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=_two_subtask_plan(0.9), plain_response="Final answer."
        ),
    )
    reviewer = ReviewerAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=ReviewResult(approved=True, score=0.9, feedback="Looks fine.")
        ),
    )

    # Research always raises inside the tool-calling loop by requesting an
    # unregistered tool, so SpecialistAgent.run catches it as a failure.
    failing_llm = FakeChatModel(
        tool_call_turns=[
            AIMessage(
                content="",
                tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "call-1"}],
            )
        ],
    )
    ok_llm = FakeChatModel(structured_response=SpecialistOutput(output="ok", confidence=0.9))

    specialists = {
        SpecialistType.RESEARCH: SpecialistAgent(
            SpecialistType.RESEARCH, "gpt-5", registry, llm=failing_llm
        ),
        SpecialistType.WRITING: SpecialistAgent(
            SpecialistType.WRITING, "gpt-5", registry, llm=ok_llm
        ),
        SpecialistType.DATA_ANALYSIS: SpecialistAgent(
            SpecialistType.DATA_ANALYSIS, "gpt-5", registry, llm=ok_llm
        ),
        SpecialistType.CODE_EXECUTION: SpecialistAgent(
            SpecialistType.CODE_EXECUTION, "gpt-5", registry, llm=ok_llm
        ),
    }
    return build_graph(supervisor, reviewer, specialists)


def test_specialist_failure_exhausts_retries_then_take_over_resumes(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "max_specialist_retries", 1)
    app = _build_stuck_specialist_app(max_retries=1)
    cfg = _config()

    paused_state = app.invoke({"task": "Summarize topic X", "memory_context": ""}, cfg)

    assert "__interrupt__" in paused_state
    payload = paused_state["__interrupt__"][0].value
    assert payload["source"] == "specialist_retry"
    assert payload["escalation"]["level"] == "take_over"
    assert payload["escalation"]["context"]["stuck_subtask_ids"] == ["st-1"]

    final_state = app.invoke(
        Command(
            resume=HumanDecision(
                action=DecisionAction.TAKE_OVER, output="Human-provided research notes."
            ).model_dump()
        ),
        cfg,
    )

    assert final_state["status"] == "completed"
    assert final_state["completed"]["st-1"].output == "Human-provided research notes."
    assert final_state["completed"]["st-2"].success is True
