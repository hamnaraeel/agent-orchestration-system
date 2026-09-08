"""Capstone end-to-end test: exercises the full assembled pipeline in one
scenario, not just one component at a time --

  memory retrieval informs planning -> a sensitive plan escalates for human
  approval -> approved -> two independent subtasks dispatch in parallel, one
  of them actually calling a tool -> the reviewer rejects once with specific
  feedback -> the flagged subtask is redone (receiving that feedback) ->
  the reviewer approves -> synthesis -> delivery -> a new long-term memory is
  recorded and working memory is cleared.

Every other test file in this suite covers one of these mechanisms in
isolation; this one proves they compose correctly end to end, using
deterministic fakes throughout so it's exact and CI-safe (see `demo.py` for
the same shape of scenario run against real LLMs).
"""
from __future__ import annotations

import uuid

import fakeredis

from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.human_loop.approval_queue import ApprovalQueue
from agent_orchestrator.human_loop.models import PendingApproval
from agent_orchestrator.memory.extractor import MemoryExtractorAgent
from agent_orchestrator.memory.long_term import LongTermMemory
from agent_orchestrator.memory.models import ExtractedMemory, MemoryRecord
from agent_orchestrator.memory.working import WorkingMemory
from agent_orchestrator.schemas import (
    DecisionAction,
    EscalationRequest,
    ExecutionPlan,
    HumanDecision,
    ReviewResult,
    SpecialistOutput,
    SpecialistType,
    SubTask,
)
from agent_orchestrator.tools.builtin import register_builtin_tools
from agent_orchestrator.tools.registry import ToolRegistry
from langchain_core.messages import AIMessage
from langgraph.types import Command
from tests.fake_embedding import FakeEmbeddingFunction
from tests.fakes import FakeChatModel


def _sensitive_two_subtask_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Research AcmeCo and write an investor summary",
        confidence=0.9,
        reasoning="Research and writing are independent; run them in parallel.",
        subtasks=[
            SubTask(
                id="st-research",
                description="Research AcmeCo's Q3 revenue",
                specialist=SpecialistType.RESEARCH,
                expected_output_format="bullet points",
                estimated_complexity=2,
                sensitive=True,  # forces APPROVE_PLAN before any work begins
            ),
            SubTask(
                id="st-write",
                description="Write the investor summary",
                specialist=SpecialistType.WRITING,
                expected_output_format="two paragraphs",
                estimated_complexity=2,
            ),
        ],
    )


def test_full_pipeline_memory_escalation_parallel_tools_and_redo(tmp_path):
    # -- long-term memory, pre-seeded so planning has something to retrieve --
    long_term_memory = LongTermMemory(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="e2e_test",
        embedding_function=FakeEmbeddingFunction(),
    )
    long_term_memory.add(
        MemoryRecord(
            user_id="alice",
            task="Research AcmeCo and write an investor summary",
            approach_summary="Pull the latest quarterly figures before drafting.",
            facts=["AcmeCo's fiscal year ends in June."],
            preferences=["Keep investor summaries to two paragraphs."],
            importance=0.7,
        )
    )
    memory_extractor = MemoryExtractorAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=ExtractedMemory(
                approach_summary="Research then write, in parallel.",
                facts=["AcmeCo grew revenue 12% YoY in Q3."],
                preferences=[],
                importance=0.6,
            )
        ),
    )

    # -- working memory + approval queue, both fakeredis-backed --
    working_memory = WorkingMemory(fakeredis.FakeStrictRedis(decode_responses=True), ttl_seconds=60)
    approval_queue = ApprovalQueue(fakeredis.FakeStrictRedis(decode_responses=True), poll_interval=0.05)

    # -- supervisor: captures the planning prompt to prove memory was injected --
    captured_prompts: list[str] = []
    supervisor_llm = FakeChatModel(
        structured_response=_sensitive_two_subtask_plan(),
        plain_response="AcmeCo grew revenue 12% YoY in Q3, driven by strong enterprise demand. "
        "GlobexCorp comparisons were not required for this request.",
    )
    original_with_structured_output = supervisor_llm.with_structured_output

    def spying_with_structured_output(schema, include_raw: bool = False):
        wrapper = original_with_structured_output(schema, include_raw=include_raw)
        original_invoke = wrapper.invoke

        def spying_invoke(messages):
            captured_prompts.append(messages[-1].content)
            return original_invoke(messages)

        wrapper.invoke = spying_invoke
        return wrapper

    supervisor_llm.with_structured_output = spying_with_structured_output
    supervisor = SupervisorAgent("gpt-5", llm=supervisor_llm)

    # -- reviewer: rejects once (flagging the research subtask), then approves --
    review_queue = [
        ReviewResult(
            approved=False,
            score=0.4,
            feedback="The research is missing a specific revenue growth percentage.",
            subtasks_to_redo=["st-research"],
        ),
        ReviewResult(approved=True, score=0.9, feedback="Now it has the specific figure."),
    ]
    reviewer = ReviewerAgent("gpt-5", llm=FakeChatModel())
    reviewer.review = lambda task, results: review_queue.pop(0)  # type: ignore[assignment]

    # -- specialists: research actually calls a tool; writing just responds --
    registry = ToolRegistry()
    register_builtin_tools(registry)
    research_calls = {"count": 0}

    def _fake_research_result(subtask_id, output):
        from agent_orchestrator.schemas import SubtaskResult, ToolCallLog

        return SubtaskResult(
            subtask_id=subtask_id,
            specialist=SpecialistType.RESEARCH,
            success=True,
            output=output,
            confidence=0.85,
            tool_calls=[
                ToolCallLog(
                    tool_name="web_search",
                    inputs={"query": "AcmeCo Q3 2025 revenue"},
                    output=output,
                    success=True,
                    latency_ms=42.0,
                )
            ],
        )

    class CountingResearchAgent(SpecialistAgent):
        def run(self, subtask, context, feedback=None):
            research_calls["count"] += 1
            output = (
                "AcmeCo Q3 revenue grew 12% YoY (found via note.txt)."
                if feedback
                else "AcmeCo Q3 revenue grew, per public filings (no specific figure)."
            )
            return _fake_research_result(subtask.id, output)

    research_agent = CountingResearchAgent(
        SpecialistType.RESEARCH, "gpt-5", registry, llm=FakeChatModel()
    )
    writing_agent = SpecialistAgent(
        SpecialistType.WRITING,
        "gpt-5",
        registry,
        llm=FakeChatModel(
            structured_response=SpecialistOutput(
                output="Draft investor paragraph about AcmeCo.", confidence=0.9
            )
        ),
    )
    specialists = {
        SpecialistType.RESEARCH: research_agent,
        SpecialistType.WRITING: writing_agent,
        SpecialistType.DATA_ANALYSIS: writing_agent,
        SpecialistType.CODE_EXECUTION: writing_agent,
    }

    app = build_graph(
        supervisor,
        reviewer,
        specialists,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        memory_extractor=memory_extractor,
    )

    task_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": task_id}}
    task_text = "Research AcmeCo and write an investor summary"

    # -- run: hits the plan-sensitive escalation first --
    paused_state = app.invoke(
        {"task": task_text, "user_id": "alice", "task_id": task_id, "memory_context": ""},
        config,
    )
    assert "__interrupt__" in paused_state
    payload = paused_state["__interrupt__"][0].value
    assert payload["source"] == "plan_sensitive"
    approval_queue.submit(
        PendingApproval(
            task_id=task_id,
            escalation=EscalationRequest(**payload["escalation"]),
            source=payload["source"],
        )
    )

    # -- resolve it (standing in for a human approving via the review UI) --
    approval_queue.resolve(task_id, HumanDecision(action=DecisionAction.APPROVE))
    final_state = app.invoke(Command(resume={"action": "approve"}), config)

    # -- the pipeline actually completed, after a redo cycle --
    assert final_state["status"] == "completed"
    assert final_state["review_cycles"] == 2
    assert research_calls["count"] == 2  # first attempt, then redo with feedback
    assert final_state["completed"]["st-research"].success is True
    assert final_state["completed"]["st-write"].success is True
    assert final_state["completed"]["st-research"].tool_calls[0].tool_name == "web_search"
    assert "12%" in final_state["completed"]["st-research"].output

    # -- memory retrieval actually influenced the planning prompt --
    assert any("AcmeCo's fiscal year ends in June." in p for p in captured_prompts)
    assert any("Keep investor summaries to two paragraphs." in p for p in captured_prompts)

    # -- a new long-term memory was recorded from this run --
    records = long_term_memory.list_user_memories("alice")
    assert len(records) == 2  # the seeded one + this run's
    assert any("AcmeCo grew revenue 12% YoY in Q3." in r.facts for r in records)

    # -- working memory was cleared on completion --
    assert working_memory.get_all(task_id) == {}

    # -- the approval was recorded as resolved, not left pending --
    assert approval_queue.list_pending() == []
    resolved = approval_queue.get(task_id)
    assert resolved.decision.action == DecisionAction.APPROVE
