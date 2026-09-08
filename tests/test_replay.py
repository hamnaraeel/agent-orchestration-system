"""Verifies time-travel replay against a real (persistent) checkpointer:
list every checkpoint a run passed through, fork one with a modified value,
resume, and confirm execution actually diverged while the original run's
checkpoints are still intact.
"""
from __future__ import annotations

from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.graph.checkpointing import sqlite_checkpointer
from agent_orchestrator.schemas import (
    ExecutionPlan,
    ReviewResult,
    SpecialistOutput,
    SpecialistType,
    SubTask,
)
from agent_orchestrator.tools.builtin import register_builtin_tools
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tracing.replay import list_checkpoints, replay_from_checkpoint
from tests.fakes import FakeChatModel


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Answer a simple question",
        confidence=0.9,
        reasoning="One research step is enough.",
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


def _build_app(checkpointer):
    registry = ToolRegistry()
    register_builtin_tools(registry)
    supervisor = SupervisorAgent(
        "gpt-5",
        llm=FakeChatModel(structured_response=_plan(), plain_response="Original final answer."),
    )
    reviewer = ReviewerAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=ReviewResult(approved=True, score=0.9, feedback="Fine.")
        ),
    )
    specialists = {
        specialist_type: SpecialistAgent(
            specialist_type,
            "gpt-5",
            registry,
            llm=FakeChatModel(structured_response=SpecialistOutput(output="ok", confidence=0.9)),
        )
        for specialist_type in SpecialistType
    }
    return build_graph(supervisor, reviewer, specialists, checkpointer=checkpointer)


def test_replay_forks_from_a_checkpoint_and_diverges():
    checkpointer = sqlite_checkpointer(":memory:")
    app = _build_app(checkpointer)
    task_id = "task-replay-1"
    config = {"configurable": {"thread_id": task_id}}

    original_result = app.invoke(
        {"task": "What is the capital of France?", "memory_context": ""}, config
    )
    assert original_result["final_output"] == "Original final answer."

    checkpoints = list_checkpoints(app, task_id)
    assert len(checkpoints) > 1
    # find the checkpoint about to run "deliver" -- final_output is already
    # decided by then, so overwriting it here is a clean, deterministic way
    # to prove the replayed branch actually diverges.
    pre_deliver = next(c for c in checkpoints if c.next_nodes == ("deliver",))
    assert pre_deliver.values["final_output"] == "Original final answer."

    replayed_result = replay_from_checkpoint(
        app, pre_deliver, {"final_output": "REPLACED DURING REPLAY"}
    )

    assert replayed_result["final_output"] == "REPLACED DURING REPLAY"
    assert replayed_result["status"] == "completed"

    # the original checkpoint history is untouched by the fork
    checkpoints_after_fork = list_checkpoints(app, task_id)
    original_pre_deliver = next(
        c for c in checkpoints_after_fork if c.checkpoint_id == pre_deliver.checkpoint_id
    )
    assert original_pre_deliver.values["final_output"] == "Original final answer."


def test_list_checkpoints_reflects_node_progression():
    checkpointer = sqlite_checkpointer(":memory:")
    app = _build_app(checkpointer)
    task_id = "task-replay-2"
    config = {"configurable": {"thread_id": task_id}}

    app.invoke({"task": "What is the capital of France?", "memory_context": ""}, config)

    checkpoints = list_checkpoints(app, task_id)
    next_sequences = [c.next_nodes for c in checkpoints]
    assert ("intake",) in next_sequences
    assert ("plan_task",) in next_sequences
    assert () in next_sequences  # the terminal checkpoint has nothing left to run
