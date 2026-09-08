"""A minimal fake chat model so agent/graph tests don't need real API keys or
network access. Implements just enough of the LangChain chat model interface
(`invoke`, `bind_tools`, `with_structured_output(..., include_raw=True)`) for
BaseAgent to work, including optional token usage for cost-tracking tests.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage


def _with_total(usage: dict | None) -> dict | None:
    if usage is None:
        return None
    if "total_tokens" in usage:
        return usage
    return {**usage, "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)}


class _StructuredWrapper:
    def __init__(self, value, usage: dict | None):
        self._value = value
        self._usage = usage

    def invoke(self, _messages):
        return {
            "raw": AIMessage(content="", usage_metadata=self._usage),
            "parsed": self._value,
            "parsing_error": None,
        }


class FakeChatModel:
    def __init__(
        self,
        structured_response=None,
        plain_response: str = "",
        tool_call_turns: list[AIMessage] | None = None,
        usage: dict | None = None,
    ) -> None:
        self.structured_response = structured_response
        self.plain_response = plain_response
        self.tool_call_turns = tool_call_turns or []
        self.usage = _with_total(usage)
        self._turn = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema, include_raw: bool = False, **_kwargs):
        return _StructuredWrapper(self.structured_response, self.usage)

    def invoke(self, _messages):
        if self._turn < len(self.tool_call_turns):
            message = self.tool_call_turns[self._turn]
            self._turn += 1
        else:
            message = AIMessage(content=self.plain_response)
        if self.usage is not None and message.usage_metadata is None:
            message = message.model_copy(update={"usage_metadata": self.usage})
        return message
