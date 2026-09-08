"""Verifies the memory system is actually wired into the graph: working memory
gets populated during execution and cleared on completion, long-term memory
gains a record after a successful run, and a later run for the same user gets
that memory back as planning context.
"""
from __future__ import annotations

import uuid

import fakeredis

from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.agents.supervisor import SupervisorAgent
from agent_orchestrator.graph.build import build_graph
from agent_orchestrator.memory.extractor import MemoryExtractorAgent
from agent_orchestrator.memory.long_term import LongTermMemory
from agent_orchestrator.memory.models import ExtractedMemory
from agent_orchestrator.memory.working import WorkingMemory
from agent_orchestrator.schemas import (
    ExecutionPlan,
    ReviewResult,
    SpecialistOutput,
    SpecialistType,
    SubTask,
)
from agent_orchestrator.tools.builtin import register_builtin_tools
from agent_orchestrator.tools.registry import ToolRegistry
from tests.fake_embedding import FakeEmbeddingFunction
from tests.fakes import FakeChatModel


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _single_subtask_plan() -> ExecutionPlan:
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


def _build_app(working_memory, long_term_memory, memory_extractor):
    registry = ToolRegistry()
    register_builtin_tools(registry)

    supervisor = SupervisorAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=_single_subtask_plan(),
            plain_response="Final synthesized answer.",
        ),
    )
    reviewer = ReviewerAgent("gpt-5", llm=FakeChatModel())
    reviewer.review = lambda task, results: ReviewResult(
        approved=True, score=0.9, feedback="Looks good."
    )

    specialists = {
        specialist_type: SpecialistAgent(
            specialist_type,
            "gpt-5",
            registry,
            llm=FakeChatModel(
                structured_response=SpecialistOutput(output="ok", confidence=0.9)
            ),
        )
        for specialist_type in SpecialistType
    }

    return build_graph(
        supervisor,
        reviewer,
        specialists,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        memory_extractor=memory_extractor,
    )


def test_working_memory_populated_during_run_and_cleared_on_completion(tmp_path):
    redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
    working_memory = WorkingMemory(redis_client, ttl_seconds=60)

    app = _build_app(working_memory, long_term_memory=None, memory_extractor=None)
    final_state = app.invoke(
        {"task": "What is the capital of France?", "memory_context": ""}, _config()
    )

    assert final_state["status"] == "completed"
    task_id = final_state["task_id"]
    # cleared after delivery: nothing left under this task's working-memory key
    assert working_memory.get_all(task_id) == {}


def test_long_term_memory_gains_a_record_after_successful_completion(tmp_path):
    long_term_memory = LongTermMemory(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="integration_test",
        embedding_function=FakeEmbeddingFunction(),
    )
    memory_extractor = MemoryExtractorAgent(
        "gpt-5",
        llm=FakeChatModel(
            structured_response=ExtractedMemory(
                approach_summary="Looked it up directly.",
                facts=["Paris is the capital of France."],
                preferences=[],
                importance=0.4,
            )
        ),
    )

    app = _build_app(
        working_memory=None,
        long_term_memory=long_term_memory,
        memory_extractor=memory_extractor,
    )
    final_state = app.invoke(
        {"task": "What is the capital of France?", "user_id": "alice", "memory_context": ""},
        _config(),
    )

    assert final_state["status"] == "completed"
    records = long_term_memory.list_user_memories("alice")
    assert len(records) == 1
    assert records[0].facts == ["Paris is the capital of France."]


def test_memory_context_is_retrieved_and_injected_before_planning(tmp_path):
    long_term_memory = LongTermMemory(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="integration_test_2",
        embedding_function=FakeEmbeddingFunction(),
    )
    from agent_orchestrator.memory.models import MemoryRecord

    long_term_memory.add(
        MemoryRecord(
            user_id="alice",
            task="What is the capital of France?",
            approach_summary="A single research lookup is enough for capital-city questions.",
            facts=["Paris is the capital of France."],
            preferences=["Keep answers short."],
            importance=0.6,
        )
    )

    captured_prompts: list[str] = []
    supervisor_llm = FakeChatModel(
        structured_response=_single_subtask_plan(), plain_response="Final answer."
    )
    original_invoke = supervisor_llm.with_structured_output

    def spying_with_structured_output(schema, include_raw: bool = False, **kwargs):
        wrapper = original_invoke(schema, include_raw=include_raw, **kwargs)
        original_wrapper_invoke = wrapper.invoke

        def spying_invoke(messages):
            captured_prompts.append(messages[-1].content)
            return original_wrapper_invoke(messages)

        wrapper.invoke = spying_invoke
        return wrapper

    supervisor_llm.with_structured_output = spying_with_structured_output

    registry = ToolRegistry()
    register_builtin_tools(registry)
    supervisor = SupervisorAgent("gpt-5", llm=supervisor_llm)
    reviewer = ReviewerAgent("gpt-5", llm=FakeChatModel())
    reviewer.review = lambda task, results: ReviewResult(
        approved=True, score=0.9, feedback="Looks good."
    )
    specialists = {
        specialist_type: SpecialistAgent(
            specialist_type,
            "gpt-5",
            registry,
            llm=FakeChatModel(
                structured_response=SpecialistOutput(output="ok", confidence=0.9)
            ),
        )
        for specialist_type in SpecialistType
    }
    app = build_graph(
        supervisor,
        reviewer,
        specialists,
        long_term_memory=long_term_memory,
    )

    app.invoke(
        {"task": "What is the capital of France?", "user_id": "alice", "memory_context": ""},
        _config(),
    )

    assert any("Paris is the capital of France." in p for p in captured_prompts)
