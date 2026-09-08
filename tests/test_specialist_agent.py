from langchain_core.messages import AIMessage

from agent_orchestrator import config
from agent_orchestrator.agents.specialists import SpecialistAgent
from agent_orchestrator.schemas import SpecialistOutput, SpecialistType, SubTask
from agent_orchestrator.tools.builtin import register_builtin_tools
from agent_orchestrator.tools.registry import ToolRegistry
from tests.fakes import FakeChatModel


def test_specialist_runs_a_tool_then_returns_structured_result(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "sandbox_workdir", str(tmp_path))

    registry = ToolRegistry()
    register_builtin_tools(registry)

    fake_llm = FakeChatModel(
        structured_response=SpecialistOutput(output="wrote the file", confidence=0.9),
        tool_call_turns=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_write",
                        "args": {"path": "notes.txt", "content": "hello"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="done", tool_calls=[]),
        ],
    )

    agent = SpecialistAgent(
        SpecialistType.CODE_EXECUTION, "gpt-5", registry, llm=fake_llm
    )
    subtask = SubTask(
        id="st-1",
        description="Write hello to notes.txt",
        specialist=SpecialistType.CODE_EXECUTION,
        expected_output_format="confirmation message",
        estimated_complexity=1,
    )

    result = agent.run(subtask, context={})

    assert result.success is True
    assert result.output == "wrote the file"
    assert result.confidence == 0.9
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "file_write"
    assert (tmp_path / "notes.txt").read_text() == "hello"


def test_specialist_captures_tool_failure_as_unsuccessful_result(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "sandbox_workdir", str(tmp_path))

    registry = ToolRegistry()
    register_builtin_tools(registry)

    fake_llm = FakeChatModel(
        tool_call_turns=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_read",
                        "args": {"path": "missing.txt"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="couldn't find it", tool_calls=[]),
        ],
        structured_response=SpecialistOutput(output="file missing", confidence=0.3),
    )

    agent = SpecialistAgent(
        SpecialistType.CODE_EXECUTION, "gpt-5", registry, llm=fake_llm
    )
    subtask = SubTask(
        id="st-1",
        description="Read missing.txt",
        specialist=SpecialistType.CODE_EXECUTION,
        expected_output_format="file contents",
        estimated_complexity=1,
    )

    result = agent.run(subtask, context={})

    assert result.success is True  # the agent recovers and still answers
    assert result.tool_calls[0].success is False
