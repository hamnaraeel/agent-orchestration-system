import pytest
from pydantic import ValidationError

from agent_orchestrator.schemas import ExecutionPlan, SpecialistType, SubTask


def test_subtask_requires_valid_complexity_range():
    with pytest.raises(ValidationError):
        SubTask(
            id="st-1",
            description="do something",
            specialist=SpecialistType.RESEARCH,
            expected_output_format="text",
            estimated_complexity=6,
        )


def test_execution_plan_holds_ordered_subtasks_with_dependencies():
    plan = ExecutionPlan(
        goal="Answer the question",
        confidence=0.8,
        reasoning="Needs research before writing.",
        subtasks=[
            SubTask(
                id="st-1",
                description="Research the topic",
                specialist=SpecialistType.RESEARCH,
                expected_output_format="bullet list of facts",
                estimated_complexity=2,
            ),
            SubTask(
                id="st-2",
                description="Write it up",
                specialist=SpecialistType.WRITING,
                depends_on=["st-1"],
                expected_output_format="markdown report",
                estimated_complexity=2,
            ),
        ],
    )
    assert plan.subtasks[1].depends_on == ["st-1"]
    assert plan.subtasks[0].depends_on == []
