"""Shared agent base: wraps a chat model and forces structured output.

Model routing is by prefix so the supervisor, reviewer, and each specialist can
be pointed at whichever provider/model suits them (see config.py) without any
agent code caring which one it got.
"""
from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from ..config import settings
from ..schemas import TokenUsage

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def get_llm(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    # pydantic-settings loads .env into `settings` only -- it does not export
    # those values into os.environ, which is what each provider's client
    # falls back to reading on its own. Pass the key explicitly whenever
    # settings actually has one; otherwise omit it so a key exported directly
    # in the shell (bypassing .env) still works via the client's own fallback.
    if model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        kwargs = {"api_key": settings.anthropic_api_key} if settings.anthropic_api_key else {}
        return ChatAnthropic(model=model_name, temperature=temperature, **kwargs)
    if model_name.startswith(("gpt", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI

        kwargs = {"api_key": settings.openai_api_key} if settings.openai_api_key else {}
        return ChatOpenAI(model=model_name, temperature=temperature, **kwargs)
    if model_name.startswith("groq:"):
        from langchain_groq import ChatGroq

        kwargs = {"api_key": settings.groq_api_key} if settings.groq_api_key else {}
        return ChatGroq(model=model_name.removeprefix("groq:"), temperature=temperature, **kwargs)
    raise ValueError(f"Don't know which provider serves model '{model_name}'.")


class BaseAgent:
    """Common structured-output plumbing for supervisor/specialist/reviewer agents.

    Pass `llm` directly (e.g. a fake in tests) to skip provider auto-detection.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm = llm if llm is not None else get_llm(model_name)
        # Token usage from the most recent call (structured/respond, or a raw
        # message a subclass records via `_record_usage`). None when the
        # provider didn't report usage. Not thread-safe across concurrent
        # tasks sharing one agent instance -- fine for a single-task-at-a-time
        # process; a future Celery worker gets its own agent instances anyway.
        self.last_usage: TokenUsage | None = None
        # The most recent user-turn prompt this agent sent, and the raw text
        # response -- so the trace explorer can show "the LLM prompt and
        # response" for a node, not just its parsed/summarized output.
        self.last_prompt: str | None = None
        self.last_response_text: str | None = None

    def _record_usage(self, message: BaseMessage) -> None:
        usage = getattr(message, "usage_metadata", None)
        self.last_usage = (
            TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
            if usage
            else None
        )

    def structured(self, user_prompt: str, schema: type[SchemaT]) -> SchemaT:
        self.last_prompt = user_prompt
        # "json_schema" (not the default "function_calling") is what's
        # portable across all three providers we route to: Anthropic doesn't
        # support "json_mode", and forcing tool_choice ("function_calling")
        # is unreliable on at least some Groq-hosted models -- confirmed by a
        # real run against groq:openai/gpt-oss-20b/120b, which rejected the
        # forced tool call with "Tool choice is required, but model did not
        # call a tool" even though the model answered correctly in prose.
        structured_llm = self.llm.with_structured_output(
            schema, include_raw=True, method="json_schema"
        )
        result = structured_llm.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        )
        self._record_usage(result["raw"])
        parsed = result["parsed"]
        self.last_response_text = parsed.model_dump_json() if parsed is not None else None
        return parsed

    def respond(self, user_prompt: str) -> str:
        self.last_prompt = user_prompt
        result = self.llm.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        )
        self._record_usage(result)
        text = result.content if hasattr(result, "content") else str(result)
        self.last_response_text = text
        return text
