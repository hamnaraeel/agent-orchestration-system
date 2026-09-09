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


def test_specialist_accumulates_token_usage_across_the_tool_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "sandbox_workdir", str(tmp_path))

    registry = ToolRegistry()
    register_builtin_tools(registry)

    fake_llm = FakeChatModel(
        structured_response=SpecialistOutput(output="done", confidence=0.9),
        tool_call_turns=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "file_write", "args": {"path": "n.txt", "content": "x"}, "id": "c1"}
                ],
            ),
            AIMessage(content="done", tool_calls=[]),
        ],
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    agent = SpecialistAgent(SpecialistType.CODE_EXECUTION, "gpt-5", registry, llm=fake_llm)
    subtask = SubTask(
        id="st-1",
        description="Write to n.txt",
        specialist=SpecialistType.CODE_EXECUTION,
        expected_output_format="confirmation",
        estimated_complexity=1,
    )

    agent.run(subtask, context={})

    # two tool-loop turns + the final structured call = 3 calls, each (10, 5)
    assert agent.last_usage.input_tokens == 30
    assert agent.last_usage.output_tokens == 15


def test_specialist_degrades_gracefully_when_tool_iterations_are_exhausted(tmp_path, monkeypatch):
    """If the model keeps calling tools without ever settling on an answer,
    the specialist should force one final tools-off turn and salvage that as
    its answer, rather than failing the whole subtask outright."""
    monkeypatch.setattr(config.settings, "sandbox_workdir", str(tmp_path))

    registry = ToolRegistry()
    register_builtin_tools(registry)

    always_calls_a_tool = AIMessage(
        content="",
        tool_calls=[{"name": "file_read", "args": {"path": "missing.txt"}, "id": "c1"}],
    )
    fake_llm = FakeChatModel(
        structured_response=SpecialistOutput(
            output="Best-effort answer from general knowledge.", confidence=0.4
        ),
        tool_call_turns=[always_calls_a_tool] * 4,  # exhausts _MAX_TOOL_ITERATIONS
        plain_response="Best-effort answer from general knowledge.",
    )

    agent = SpecialistAgent(SpecialistType.RESEARCH, "gpt-5", registry, llm=fake_llm)
    subtask = SubTask(
        id="st-1",
        description="Find a fact that isn't available locally",
        specialist=SpecialistType.RESEARCH,
        expected_output_format="text",
        estimated_complexity=1,
    )

    result = agent.run(subtask, context={})

    assert result.success is True
    assert result.output == "Best-effort answer from general knowledge."
    assert len(result.tool_calls) == 4  # all four attempts still logged
