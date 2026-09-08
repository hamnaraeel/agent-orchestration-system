"""Specialist agents: each owns one domain, has access to a subset of the tool
registry, and runs a small tool-calling loop (bind_tools -> execute -> respond)
before handing a structured SubtaskResult back to the supervisor.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ..schemas import SpecialistOutput, SpecialistType, SubTask, SubtaskResult, ToolCallLog
from ..tools.registry import ToolRegistry
from .base import BaseAgent, get_llm

_SYSTEM_PROMPTS: dict[SpecialistType, str] = {
    SpecialistType.RESEARCH: (
        "You are the Research specialist. Use the tools available to you to "
        "gather facts needed for the subtask, then summarize what you found."
    ),
    SpecialistType.DATA_ANALYSIS: (
        "You are the Data Analysis specialist. Use the tools available to you "
        "to query, compute, or transform data, then report your findings."
    ),
    SpecialistType.WRITING: (
        "You are the Writing specialist. Produce clear, well-structured prose "
        "that satisfies the subtask's expected output format."
    ),
    SpecialistType.CODE_EXECUTION: (
        "You are the Code Execution specialist. Write and run code to satisfy "
        "the subtask, then report the result."
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

        messages: list = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        tool_calls_made: list[ToolCallLog] = []

        try:
            llm_with_tools = (
                self.llm.bind_tools(self._lc_tools) if self._lc_tools else self.llm
            )
            for _ in range(_MAX_TOOL_ITERATIONS):
                ai_message: AIMessage = llm_with_tools.invoke(messages)
                messages.append(ai_message)
                if not getattr(ai_message, "tool_calls", None):
                    break
                for call in ai_message.tool_calls:
                    before = len(self.tool_registry.call_log)
                    tool = next(t for t in self._lc_tools if t.name == call["name"])
                    try:
                        result = tool.invoke(call["args"])
                    except Exception as exc:  # noqa: BLE001
                        result = f"ERROR: {exc}"
                    tool_calls_made.extend(self.tool_registry.call_log[before:])
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )
            else:
                raise RuntimeError(
                    f"Exceeded {_MAX_TOOL_ITERATIONS} tool-calling iterations without a final answer."
                )

            final = self.llm.with_structured_output(SpecialistOutput).invoke(messages)
            return SubtaskResult(
                subtask_id=subtask.id,
                specialist=self.specialist_type,
                success=True,
                output=final.output,
                confidence=final.confidence,
                tool_calls=tool_calls_made,
            )
        except Exception as exc:  # noqa: BLE001 - specialist failures are routed, not raised
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
