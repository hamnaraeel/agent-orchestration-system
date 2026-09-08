"""Answers a human reviewer's clarifying questions about a paused task, using
only the packaged escalation context (task, plan, completed outputs, review)
as reference material -- it has no live access to the running graph.
"""
from __future__ import annotations

import json

from .base import BaseAgent

_SYSTEM_PROMPT = """\
A human reviewer is deciding whether to approve, reject, modify, or take over \
a paused task. You have the full context of what happened so far. Answer \
their question directly and concisely, using only this context. If the \
context doesn't contain the answer, say so plainly -- don't guess.
"""


class ClarificationAgent(BaseAgent):
    def __init__(self, model_name: str, llm=None) -> None:
        super().__init__(model_name, _SYSTEM_PROMPT, llm=llm)

    def answer(self, context: dict, question: str) -> str:
        prompt = f"Context:\n{json.dumps(context, indent=2, default=str)}\n\nQuestion: {question}"
        return self.respond(prompt)
