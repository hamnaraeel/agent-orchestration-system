"""Verifies the Celery task in isolation (reconstructs a specialist and runs
it), and that the graph's `run_specialist` node correctly dispatches through
Celery -- in "always eager" mode, so no real broker/Redis is needed; Celery
runs the task function directly in-process but through its normal
dispatch/serialization path, unlike calling the function directly.
"""
from __future__ import annotations

import uuid

from agent_orchestrator import tasks as tasks_module
from agent_orchestrator.agents import base as base_module
from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.schemas import (
    ExecutionPlan,
    ReviewResult,
    SpecialistOutput,
    SpecialistType,
    SubTask,
)
from tests.fakes import FakeChatModel


def _fake_get_llm(model_name, temperature=0.0):
    return FakeChatModel(
        structured_response=SpecialistOutput(output="task done via celery", confidence=0.8),
        usage={"input_tokens": 5, "output_tokens": 3},
    )


def test_run_specialist_task_reconstructs_and_runs_a_specialist(monkeypatch):
    monkeypatch.setattr(base_module, "get_llm", _fake_get_llm)

    subtask = SubTask(
        id="st-1",
        description="Do a thing",
        specialist=SpecialistType.RESEARCH,
        expected_output_format="text",
        estimated_complexity=1,
    )
    payload = tasks_module.run_specialist_task.apply(
        args=(SpecialistType.RESEARCH.value, "gpt-5", subtask.model_dump(), {}, None)
    ).get()

    assert payload["subtask_result"]["success"] is True
    assert payload["subtask_result"]["output"] == "task done via celery"
    assert payload["model"] == "gpt-5"
    # one tool-loop turn (no tools called) + one final structured call, both
    # against the same fake usage of (5, 3) -> doubled
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 6}
    assert payload["prompt"] is not None


def test_graph_dispatches_specialist_work_through_celery_when_enabled(monkeypatch):
    monkeypatch.setattr(base_module, "get_llm", _fake_get_llm)
    monkeypatch.setattr(tasks_module.celery_app.conf, "task_always_eager", True)

    plan = ExecutionPlan(
        goal="Answer a question",
        confidence=0.9,
        reasoning="One step is enough.",
        subtasks=[
            SubTask(
                id="st-1",
                description="Look something up",
                specialist=SpecialistType.RESEARCH,
                expected_output_format="text",
                estimated_complexity=1,
            )
        ],
    )
    supervisor = SupervisorAgent(
        "gpt-5",
        llm=FakeChatModel(structured_response=plan, plain_response="Final answer."),
    )
    reviewer = ReviewerAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=ReviewResult(approved=True, score=0.9, feedback="Fine.")
        ),
    )
    # `specialists` still needs real SpecialistAgent instances (build_graph
    # reads .model_name off them to pass to the Celery task), but their `llm`
    # is never used directly when use_celery=True -- the task reconstructs
    # its own via the (monkeypatched) get_llm.
    from agent_orchestrator.agents.specialists import SpecialistAgent
    from agent_orchestrator.tools.builtin import register_builtin_tools
    from agent_orchestrator.tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_builtin_tools(registry)
    specialists = {
        specialist_type: SpecialistAgent(specialist_type, "gpt-5", registry)
        for specialist_type in SpecialistType
    }

    app = build_graph(supervisor, reviewer, specialists, use_celery=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = app.invoke({"task": "What is the capital of France?", "memory_context": ""}, config)

    assert final_state["status"] == "completed"
    assert final_state["completed"]["st-1"].output == "task done via celery"
    specialist_span = next(
        e for e in final_state["trace_events"] if e["node"] == "run_specialist:st-1"
    )
    assert specialist_span["attributes"]["executed_via"] == "celery"
    assert specialist_span["usage"][0]["input_tokens"] == 10
