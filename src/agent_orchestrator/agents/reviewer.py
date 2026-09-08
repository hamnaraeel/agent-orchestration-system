"""Reviewer agent: validates specialist outputs before they reach synthesis,
flags subtasks that need to be redone, and marks sensitive outputs for human
review.
"""
from __future__ import annotations

from ..schemas import ReviewResult, SubtaskResult
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Reviewer agent. You receive the original task and every completed \
subtask's output. Judge whether the outputs, taken together, actually satisfy \
the task's requirements and expected output formats.

- Set `approved` to false if any output is incomplete, wrong, or doesn't match \
its expected format, and list the offending subtask ids in `subtasks_to_redo` \
with specific, actionable `feedback`.
- Set `requires_human_review` to true if the task involves financial \
transactions, data deletion, external communications sent on the user's \
behalf, or anything else where a wrong answer would be costly.
- `score` should reflect overall quality, not just pass/fail.
"""


class ReviewerAgent(BaseAgent):
    def __init__(self, model_name: str, llm=None) -> None:
        super().__init__(model_name, _SYSTEM_PROMPT, llm=llm)

    def review(self, task: str, results: dict[str, SubtaskResult]) -> ReviewResult:
        outputs = "\n\n".join(
            f"[{sid}] ({r.specialist.value}, success={r.success}): {r.output or r.error}"
            for sid, r in results.items()
        )
        prompt = f"Original task:\n{task}\n\nSubtask outputs:\n{outputs}"
        return self.structured(prompt, ReviewResult)
