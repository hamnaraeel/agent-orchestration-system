"""End-to-end smoke tests for the LangGraph state machine, using FakeChatModel
so no LLM provider or API key is needed. These exercise the conditional edges
described in the Phase 1 plan: low-confidence plan -> escalation, specialist
failure -> retry -> escalation once exhausted, and reviewer rejection ->
redo -> re-review -> synthesis.
"""
from __future__ import annotations

from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.schemas import (
    ExecutionPlan,
    ReviewResult,
    SpecialistOutput,
    SpecialistType,
    SubTask,
)
from agent_orchestrator.tools.builtin import register_builtin_tools
from agent_orchestrator.tools.registry import ToolRegistry
from tests.fakes import FakeChatModel


def _two_subtask_plan(confidence: float = 0.9) -> ExecutionPlan:
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
):
    registry = ToolRegistry()
    register_builtin_tools(registry)

    supervisor_llm = FakeChatModel(
        structured_response=_two_subtask_plan(plan_confidence),
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

    final_state = app.invoke({"task": "Summarize topic X", "memory_context": ""})

    assert final_state["status"] == "completed"
    assert final_state["final_output"] == "Final synthesized answer."
    assert final_state["completed"]["st-1"].success is True
    assert final_state["completed"]["st-2"].success is True


def test_low_confidence_plan_escalates_before_any_work(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "plan_confidence_threshold", 0.6)
    app = _build_test_app(
        plan_confidence=0.3,
        specialist_outputs={},
        review_results=[],
    )

    final_state = app.invoke({"task": "Do something risky", "memory_context": ""})

    assert final_state["status"] == "escalated"
    assert final_state["escalation"].level.value == "approve_plan"
    assert final_state.get("completed", {}) == {}


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

    final_state = app.invoke({"task": "Summarize topic X", "memory_context": ""})

    assert final_state["status"] == "completed"
    assert final_state["review_cycles"] == 2


def test_specialist_failure_exhausts_retries_and_escalates(monkeypatch):
    from agent_orchestrator import config

    monkeypatch.setattr(config.settings, "max_specialist_retries", 1)

    registry = ToolRegistry()
    register_builtin_tools(registry)

    supervisor = SupervisorAgent(
        "gpt-5",
        llm=FakeChatModel(structured_response=_two_subtask_plan(0.9)),
    )
    reviewer = ReviewerAgent("gpt-5", llm=FakeChatModel())

    # Research always raises inside the tool-calling loop by requesting an
    # unregistered tool, so SpecialistAgent.run catches it as a failure.
    from langchain_core.messages import AIMessage

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

    app = build_graph(supervisor, reviewer, specialists)
    final_state = app.invoke({"task": "Summarize topic X", "memory_context": ""})

    assert final_state["status"] == "escalated"
    assert final_state["completed"]["st-1"].success is False
