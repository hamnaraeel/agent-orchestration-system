"""Extracts a long-term memory from a just-completed task: what approach
worked, what facts came up, and what preferences the user seemed to have.
"""
from __future__ import annotations

from ..agents.base import BaseAgent
from ..schemas import ExecutionPlan, SubtaskResult
from .models import ExtractedMemory, MemoryRecord

_SYSTEM_PROMPT = """\
You just watched a multi-agent system complete a task. Extract what's worth \
remembering for next time a similar task comes in:

- `approach_summary`: what decomposition/approach was used and why it worked.
- `facts`: concrete, reusable domain facts discovered (not task-specific trivia).
- `preferences`: anything about how the user likes results delivered (format, \
tone, level of detail, constraints) -- only if actually evidenced by the task.
- `importance`: how useful this memory will be for future similar tasks. Give \
routine, easily-repeatable tasks a lower score than tasks with a hard-won or \
non-obvious approach.
"""


class MemoryExtractorAgent(BaseAgent):
    def __init__(self, model_name: str, llm=None) -> None:
        super().__init__(model_name, _SYSTEM_PROMPT, llm=llm)

    def extract(
        self,
        user_id: str,
        task: str,
        plan: ExecutionPlan,
        results: dict[str, SubtaskResult],
        final_output: str,
    ) -> MemoryRecord:
        outputs = "\n\n".join(
            f"[{sid}] ({r.specialist.value}): {r.output}" for sid, r in results.items()
        )
        tools_used = sorted(
            {call.tool_name for r in results.values() for call in r.tool_calls}
        )
        prompt = (
            f"Task: {task}\n\nPlan reasoning: {plan.reasoning}\n\n"
            f"Subtask outputs:\n{outputs}\n\nFinal delivered output:\n{final_output}"
        )
        extracted = self.structured(prompt, ExtractedMemory)
        return MemoryRecord(
            user_id=user_id,
            task=task,
            approach_summary=extracted.approach_summary,
            tools_used=tools_used,
            facts=extracted.facts,
            preferences=extracted.preferences,
            success=True,
            importance=extracted.importance,
        )
