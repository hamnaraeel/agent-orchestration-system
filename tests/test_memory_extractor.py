from agent_orchestrator.memory.extractor import MemoryExtractorAgent
from agent_orchestrator.memory.models import ExtractedMemory
from agent_orchestrator.schemas import (
    ExecutionPlan,
    SpecialistType,
    SubTask,
    SubtaskResult,
    ToolCallLog,
)
from tests.fakes import FakeChatModel


def test_extract_builds_memory_record_with_aggregated_tools_used():
    extracted = ExtractedMemory(
        approach_summary="Researched first, then wrote a summary.",
        facts=["AcmeCo's fiscal year ends in June."],
        preferences=["Prefers bullet points."],
        importance=0.7,
    )
    agent = MemoryExtractorAgent(
        "gpt-5", llm=FakeChatModel(structured_response=extracted)
    )

    plan = ExecutionPlan(
        goal="Summarize AcmeCo earnings",
        confidence=0.9,
        reasoning="Research then write.",
        subtasks=[
            SubTask(
                id="st-1",
                description="Research",
                specialist=SpecialistType.RESEARCH,
                expected_output_format="bullets",
                estimated_complexity=2,
            ),
        ],
    )
    results = {
        "st-1": SubtaskResult(
            subtask_id="st-1",
            specialist=SpecialistType.RESEARCH,
            success=True,
            output="facts found",
            tool_calls=[
                ToolCallLog(
                    tool_name="web_search",
                    inputs={"query": "AcmeCo"},
                    output="results",
                    success=True,
                    latency_ms=12.0,
                )
            ],
        ),
        "st-2": SubtaskResult(
            subtask_id="st-2",
            specialist=SpecialistType.WRITING,
            success=True,
            output="summary",
            tool_calls=[
                ToolCallLog(
                    tool_name="file_write",
                    inputs={"path": "out.md", "content": "..."},
                    output="wrote",
                    success=True,
                    latency_ms=5.0,
                )
            ],
        ),
    }

    record = agent.extract("alice", "Summarize AcmeCo earnings", plan, results, "final summary text")

    assert record.user_id == "alice"
    assert record.approach_summary == extracted.approach_summary
    assert record.facts == extracted.facts
    assert record.preferences == extracted.preferences
    assert record.importance == 0.7
    assert record.tools_used == ["file_write", "web_search"]
