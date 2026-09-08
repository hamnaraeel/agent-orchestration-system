"""Supervisor agent: decomposes a task into a validated ExecutionPlan and,
once specialists finish, synthesizes their outputs into the final deliverable.
"""
from __future__ import annotations

from ..schemas import ExecutionPlan, SubtaskResult
from .base import BaseAgent

_PLANNING_SYSTEM_PROMPT = """\
You are the Supervisor agent in a multi-agent orchestration system. Given a \
user task, break it into an ordered list of subtasks, each assigned to exactly \
one specialist: research, data_analysis, writing, or code_execution.

Rules:
- Every subtask needs a unique id (e.g. "st-1", "st-2").
- Use `depends_on` to list the ids of subtasks that must finish first. Only \
depend on ids you defined earlier in the plan.
- Keep subtasks as independent as possible so they can run in parallel when \
they don't depend on each other.
- Set `confidence` honestly: lower it if the task is ambiguous, open-ended, or \
you're unsure a clean decomposition is possible.
"""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are the Supervisor agent. All specialists have finished their subtasks and \
the reviewer has approved the results. Combine the subtask outputs into a \
single, coherent deliverable that directly answers the original task. Do not \
mention the internal subtask structure to the user.
"""


class SupervisorAgent(BaseAgent):
    def __init__(self, model_name: str, llm=None) -> None:
        super().__init__(model_name, _PLANNING_SYSTEM_PROMPT, llm=llm)

    def create_plan(self, task: str, memory_context: str = "") -> ExecutionPlan:
        prompt = f"Task:\n{task}"
        if memory_context:
            prompt += (
                "\n\nRelevant memory from past tasks (use it to inform this plan, "
                f"but don't assume it applies verbatim):\n{memory_context}"
            )
        return self.structured(prompt, ExecutionPlan)

    def synthesize(self, task: str, results: dict[str, SubtaskResult]) -> str:
        self.system_prompt = _SYNTHESIS_SYSTEM_PROMPT
        outputs = "\n\n".join(
            f"[{sid}] ({r.specialist.value}): {r.output}" for sid, r in results.items()
        )
        prompt = f"Original task:\n{task}\n\nSubtask outputs:\n{outputs}"
        return self.respond(prompt)
