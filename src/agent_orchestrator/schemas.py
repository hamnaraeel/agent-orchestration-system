"""Structured-output schemas shared by every agent and the LangGraph state machine.

These are the contracts the supervisor, specialists, and reviewer are forced to
produce via `.with_structured_output(...)`, so a malformed plan or review is a
validation error rather than a string an agent could paraphrase.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SpecialistType(str, Enum):
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    WRITING = "writing"
    CODE_EXECUTION = "code_execution"


class SubTask(BaseModel):
    id: str = Field(description="Short unique identifier, e.g. 'st-1'.")
    description: str = Field(description="What this subtask must accomplish.")
    specialist: SpecialistType
    required_inputs: list[str] = Field(
        default_factory=list,
        description="What information/data this subtask needs to run, in plain language.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of other subtasks that must complete before this one can start.",
    )
    expected_output_format: str = Field(
        description="Short description of the expected shape of the output, e.g. 'markdown table' or 'JSON list of facts'."
    )
    estimated_complexity: int = Field(
        ge=1, le=5, description="1 = trivial, 5 = very complex."
    )
    sensitive: bool = Field(
        default=False,
        description=(
            "True if this subtask involves a financial transaction, deleting or "
            "overwriting data, or sending a communication externally on the user's "
            "behalf -- anything where a wrong answer would be costly."
        ),
    )


class ExecutionPlan(BaseModel):
    goal: str = Field(description="Restatement of what the overall task is trying to achieve.")
    subtasks: list[SubTask]
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Supervisor's confidence that this plan will satisfy the request.",
    )
    reasoning: str = Field(description="Why the task was decomposed this way.")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCallLog(BaseModel):
    tool_name: str
    inputs: dict
    output: str | None = None
    success: bool
    latency_ms: float
    error: str | None = None


class SubtaskResult(BaseModel):
    subtask_id: str
    specialist: SpecialistType
    success: bool
    output: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    error: str | None = None


class SpecialistOutput(BaseModel):
    output: str = Field(description="The final result of the subtask, ready to hand back to the supervisor.")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident the specialist is in this output.")


class ReviewResult(BaseModel):
    approved: bool
    score: float = Field(ge=0.0, le=1.0, description="Overall quality score of the aggregated output.")
    feedback: str = Field(description="Actionable feedback for the specialist(s) if not approved.")
    subtasks_to_redo: list[str] = Field(
        default_factory=list,
        description="Subtask IDs that need to be redone, if any.",
    )
    requires_human_review: bool = Field(
        default=False,
        description="True if this output involves sensitive operations or is too risky to auto-approve.",
    )


class EscalationLevel(str, Enum):
    NOTIFY = "notify"
    APPROVE_ACTION = "approve_action"
    APPROVE_PLAN = "approve_plan"
    TAKE_OVER = "take_over"


class EscalationRequest(BaseModel):
    level: EscalationLevel
    reason: str
    context: dict = Field(default_factory=dict)


class DecisionAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    TAKE_OVER = "take_over"


class HumanDecision(BaseModel):
    action: DecisionAction
    feedback: str | None = Field(
        default=None, description="Guidance for a 'modify' decision."
    )
    output: str | None = Field(
        default=None, description="The human-provided output for a 'take_over' decision."
    )
