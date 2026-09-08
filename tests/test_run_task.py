"""Exercises the actual runner loop (`run.run_task`): on escalation it submits
to the approval queue and blocks; resolving the queue from another thread
(standing in for a separate review API/UI process) is what unblocks it.
"""
from __future__ import annotations

import threading
import time

import fakeredis

from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.human_loop.approval_queue import ApprovalQueue
from agent_orchestrator.run import run_task
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


def _low_confidence_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Do a risky thing",
        confidence=0.2,
        reasoning="Ambiguous request.",
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
            structured_response=_low_confidence_plan(), plain_response="Final answer."
        ),
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
    return build_graph(supervisor, reviewer, specialists)


def test_run_task_blocks_on_approval_queue_until_resolved():
    app = _build_app()
    redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
    approval_queue = ApprovalQueue(redis_client, poll_interval=0.05)

    result_holder: dict = {}

    def runner():
        result_holder["final_state"] = run_task(
            app, "Do a risky thing", user_id="alice", approval_queue=approval_queue
        )

    thread = threading.Thread(target=runner)
    thread.start()

    deadline = time.monotonic() + 5
    pending = []
    while time.monotonic() < deadline and not pending:
        pending = approval_queue.list_pending()
        time.sleep(0.02)

    assert len(pending) == 1
    approval = pending[0]
    assert approval.source == "plan_low_confidence"

    approval_queue.resolve(approval.task_id, HumanDecision(action=DecisionAction.APPROVE))
    thread.join(timeout=5)

    assert not thread.is_alive()
    final_state = result_holder["final_state"]
    assert final_state["status"] == "completed"
    assert final_state["completed"]["st-1"].success is True
