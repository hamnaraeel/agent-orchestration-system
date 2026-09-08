"""Shared agent base: wraps a chat model and forces structured output.

Model routing is by prefix so the supervisor, reviewer, and each specialist can
be pointed at whichever provider/model suits them (see config.py) without any
agent code caring which one it got.
"""
from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def get_llm(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    if model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)
    if model_name.startswith(("gpt", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)
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

    def structured(self, user_prompt: str, schema: type[SchemaT]) -> SchemaT:
        structured_llm = self.llm.with_structured_output(schema)
        return structured_llm.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        )

    def respond(self, user_prompt: str) -> str:
        result = self.llm.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        )
        return result.content if hasattr(result, "content") else str(result)
