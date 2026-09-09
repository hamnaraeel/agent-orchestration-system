"""Specialist agents: each owns one domain, has access to a subset of the tool
registry, and runs a small tool-calling loop (bind_tools -> execute -> respond)
before handing a structured SubtaskResult back to the supervisor.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ..schemas import (
    SpecialistOutput,
    SpecialistType,
    SubTask,
    SubtaskResult,
    TokenUsage,
    ToolCallLog,
)
from ..tools.registry import ToolRegistry
from .base import BaseAgent, get_llm

_DEGRADE_GRACEFULLY = (
    " Only call a tool if you have a concrete reason to think it will return "
    "something useful -- don't call file/database tools speculatively hoping "
    "data happens to exist there. If a tool doesn't have what you need after "
    "one or two tries, stop searching and answer from your own knowledge "
    "instead, clearly noting where you're estimating or uncertain rather "
    "than leaving the subtask unanswered."
)

_SYSTEM_PROMPTS: dict[SpecialistType, str] = {
    SpecialistType.RESEARCH: (
        "You are the Research specialist. Use the tools available to you to "
        "gather facts needed for the subtask, then summarize what you found."
        + _DEGRADE_GRACEFULLY
    ),
    SpecialistType.DATA_ANALYSIS: (
        "You are the Data Analysis specialist. Use the tools available to you "
        "to query, compute, or transform data, then report your findings."
        + _DEGRADE_GRACEFULLY
    ),
    SpecialistType.WRITING: (
        "You are the Writing specialist. Produce clear, well-structured prose "
        "that satisfies the subtask's expected output format."
    ),
    SpecialistType.CODE_EXECUTION: (
        "You are the Code Execution specialist. Write and run code to satisfy "
        "the subtask, then report the result."
        + _DEGRADE_GRACEFULLY
    ),
}

_MAX_TOOL_ITERATIONS = 4


class SpecialistAgent(BaseAgent):
    def __init__(
        self,
        specialist_type: SpecialistType,
        model_name: str,
        tool_registry: ToolRegistry,
        llm=None,
    ) -> None:
        super().__init__(model_name, _SYSTEM_PROMPTS[specialist_type], llm=llm)
        self.specialist_type = specialist_type
        self.tool_registry = tool_registry
        self._lc_tools = self._build_langchain_tools()

    def _build_langchain_tools(self) -> list[StructuredTool]:
        tools = []
        for registered in self.tool_registry.list_tools_for(self.specialist_type):

            def _make_runner(tool_name: str):
                def _run(**kwargs) -> str:
                    log = self.tool_registry.invoke(
                        tool_name, self.specialist_type, **kwargs
                    )
                    if not log.success:
                        raise RuntimeError(log.error or f"Tool '{tool_name}' failed.")
                    return log.output or ""

                return _run

            tools.append(
                StructuredTool.from_function(
                    func=_make_runner(registered.name),
                    name=registered.name,
                    description=registered.description,
                    args_schema=registered.args_schema,
                )
            )
        return tools

    def run(
        self,
        subtask: SubTask,
        context: dict[str, SubtaskResult],
        feedback: str | None = None,
    ) -> SubtaskResult:
        dependency_context = "\n".join(
            f"[{sid}] {result.output}"
            for sid, result in context.items()
            if sid in subtask.depends_on
        )
        prompt = (
            f"Subtask: {subtask.description}\n"
            f"Required inputs: {', '.join(subtask.required_inputs) or 'none'}\n"
            f"Expected output format: {subtask.expected_output_format}\n"
        )
        if dependency_context:
            prompt += f"\nOutputs from dependency subtasks:\n{dependency_context}"
        if feedback:
            prompt += (
                f"\n\nThis is a retry. Feedback on the previous attempt:\n{feedback}\n"
                "Address this feedback in your new attempt."
            )
        self.last_prompt = prompt

        messages: list = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        tool_calls_made: list[ToolCallLog] = []
        total_usage = TokenUsage()

        def _accumulate() -> None:
            nonlocal total_usage
            if self.last_usage is not None:
                total_usage = TokenUsage(
                    input_tokens=total_usage.input_tokens + self.last_usage.input_tokens,
                    output_tokens=total_usage.output_tokens + self.last_usage.output_tokens,
                )

        try:
            llm_with_tools = (
                self.llm.bind_tools(self._lc_tools) if self._lc_tools else self.llm
            )
            for _ in range(_MAX_TOOL_ITERATIONS):
                ai_message: AIMessage = llm_with_tools.invoke(messages)
                self._record_usage(ai_message)
                _accumulate()
                messages.append(ai_message)
                if not getattr(ai_message, "tool_calls", None):
                    break
                for call in ai_message.tool_calls:
                    before = len(self.tool_registry.call_log)
                    tool = next((t for t in self._lc_tools if t.name == call["name"]), None)
                    if tool is None:
                        # A hallucinated tool name: feed the error back like any
                        # other tool failure so the model can self-correct
                        # within the loop, instead of aborting the whole attempt.
                        result = f"ERROR: no such tool '{call['name']}'."
                    else:
                        try:
                            result = tool.invoke(call["args"])
                        except Exception as exc:  # noqa: BLE001
                            result = f"ERROR: {exc}"
                        tool_calls_made.extend(self.tool_registry.call_log[before:])
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )
            else:
                # Ran out of tool-calling iterations without the model settling
                # on a final answer. Rather than failing the subtask outright,
                # give it one last forced turn with tools removed so it must
                # answer directly from whatever it has gathered so far.
                messages.append(
                    HumanMessage(
                        content="You must answer now, using only what you've already "
                        "gathered above -- no more tool calls are available. If you "
                        "couldn't find the specific information requested, give your "
                        "best answer from general knowledge and say so plainly."
                    )
                )
                ai_message = self.llm.invoke(messages)
                self._record_usage(ai_message)
                _accumulate()
                messages.append(ai_message)

            final = self.llm.with_structured_output(
                SpecialistOutput, include_raw=True, method="json_schema"
            ).invoke(messages)
            self._record_usage(final["raw"])
            _accumulate()
            self.last_usage = total_usage
            parsed = final["parsed"]
            self.last_response_text = parsed.output
            return SubtaskResult(
                subtask_id=subtask.id,
                specialist=self.specialist_type,
                success=True,
                output=parsed.output,
                confidence=parsed.confidence,
                tool_calls=tool_calls_made,
            )
        except Exception as exc:  # noqa: BLE001 - specialist failures are routed, not raised
            self.last_usage = total_usage
            return SubtaskResult(
                subtask_id=subtask.id,
                specialist=self.specialist_type,
                success=False,
                error=str(exc),
                tool_calls=tool_calls_made,
                confidence=0.0,
            )


def build_specialists(
    tool_registry: ToolRegistry, model_map: dict[SpecialistType, str]
) -> dict[SpecialistType, SpecialistAgent]:
    return {
        specialist_type: SpecialistAgent(specialist_type, model_name, tool_registry)
        for specialist_type, model_name in model_map.items()
    }
