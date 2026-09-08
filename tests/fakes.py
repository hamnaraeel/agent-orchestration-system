"""A minimal fake chat model so agent/graph tests don't need real API keys or
network access. Implements just enough of the LangChain chat model interface
(`invoke`, `bind_tools`, `with_structured_output`) for BaseAgent to work.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage


class _StructuredWrapper:
    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class FakeChatModel:
    def __init__(
        self,
        structured_response=None,
        plain_response: str = "",
        tool_call_turns: list[AIMessage] | None = None,
    ) -> None:
        self.structured_response = structured_response
        self.plain_response = plain_response
        self.tool_call_turns = tool_call_turns or []
        self._turn = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return _StructuredWrapper(self.structured_response)

    def invoke(self, _messages):
        if self._turn < len(self.tool_call_turns):
            message = self.tool_call_turns[self._turn]
            self._turn += 1
            return message
        return AIMessage(content=self.plain_response)
