"""Verifies `run_traced_task` actually persists a usable trace: a task row
with aggregated cost/tokens/tool-calls, and a span per node (plus one per
tool call), correctly ordered and attributed.
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
from agent_orchestrator.tracing.recorder import run_traced_task
from agent_orchestrator.tracing.store import TraceStore
from tests.fakes import FakeChatModel


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Research and write a summary",
        confidence=0.9,
        reasoning="Research first, then write.",
        subtasks=[
            SubTask(
                id="st-1",
                description="Research the topic",
                specialist=SpecialistType.RESEARCH,
                expected_output_format="bullet points",
                estimated_complexity=2,
            ),
        ],
    )


def _build_app():
    registry = ToolRegistry()
    register_builtin_tools(registry)

    supervisor = SupervisorAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=_plan(),
            plain_response="Final synthesized answer.",
            usage={"input_tokens": 100, "output_tokens": 50},
        ),
    )
    reviewer = ReviewerAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=ReviewResult(approved=True, score=0.9, feedback="Fine."),
            usage={"input_tokens": 20, "output_tokens": 10},
        ),
    )
    specialists = {
        specialist_type: SpecialistAgent(
            specialist_type,
            "gpt-5",
            registry,
            llm=FakeChatModel(
                structured_response=SpecialistOutput(output="facts found", confidence=0.9),
                usage={"input_tokens": 30, "output_tokens": 15},
            ),
        )
        for specialist_type in SpecialistType
    }
    return build_graph(supervisor, reviewer, specialists)


def test_traced_run_persists_a_full_task_and_span_trace(tmp_path):
    app = _build_app()
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"))

    final_state = run_traced_task(app, "Summarize topic X", trace_store, user_id="alice")

    assert final_state["status"] == "completed"

    task_id = final_state["task_id"]
    task = trace_store.get_task(task_id)
    assert task is not None
    assert task.status == "completed"
    assert task.user_id == "alice"
    assert task.task_type == "research"
    # plan(100) + specialist(30*2: one tool-loop turn + one final structured
    # call, both against the same fake usage) + review(20) + synthesize(100)
    assert task.total_input_tokens == 100 + 30 * 2 + 20 + 100
    assert task.total_output_tokens == 50 + 15 * 2 + 10 + 50
    assert task.total_cost_usd > 0
    assert task.total_tool_calls == 0  # this plan's specialist doesn't call any tool
    assert task.wall_clock_ms is not None and task.wall_clock_ms >= 0

    spans = trace_store.get_spans(task_id)
    node_names = [s.node_name for s in spans]
    assert node_names == [
        "intake",
        "plan_task",
        "run_specialist:st-1",
        "review",
        "synthesize",
        "deliver",
    ]
    plan_span = next(s for s in spans if s.node_name == "plan_task")
    assert plan_span.agent == "supervisor"
    assert plan_span.status == "success"
    assert plan_span.input_tokens == 100
    assert plan_span.attributes["subtask_count"] == 1

    specialist_span = next(s for s in spans if s.node_name == "run_specialist:st-1")
    assert specialist_span.agent == "research"
    assert specialist_span.attributes["output"] == "facts found"


def test_cost_and_agent_aggregates_reflect_the_run(tmp_path):
    app = _build_app()
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"))
    run_traced_task(app, "Summarize topic X", trace_store, user_id="alice")

    by_type = trace_store.cost_by_task_type()
    assert len(by_type) == 1
    assert by_type[0]["task_type"] == "research"
    assert by_type[0]["task_count"] == 1

    agents = {row["agent"]: row for row in trace_store.most_expensive_agents()}
    assert "supervisor" in agents
    assert "research" in agents
    assert "reviewer" in agents
    assert agents["supervisor"]["call_count"] == 2  # plan_task + synthesize
