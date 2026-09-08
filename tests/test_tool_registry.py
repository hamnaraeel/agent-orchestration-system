import pytest
from pydantic import BaseModel

from agent_orchestrator.schemas import SpecialistType
from agent_orchestrator.tools.registry import (
    RateLimitExceeded,
    ToolNotAllowed,
    ToolNotFound,
    ToolRegistry,
)


class _EchoInput(BaseModel):
    text: str


class _NoInput(BaseModel):
    pass


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echoes its input back.",
        func=lambda text: text,
        allowed_specialists={SpecialistType.RESEARCH},
        args_schema=_EchoInput,
        rate_limit_per_minute=2,
    )
    registry.register(
        name="boom",
        description="Always fails.",
        func=lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
        allowed_specialists={SpecialistType.RESEARCH},
        args_schema=_NoInput,
    )
    return registry


def test_invoke_logs_success():
    registry = make_registry()
    log = registry.invoke("echo", SpecialistType.RESEARCH, text="hi")
    assert log.success is True
    assert log.output == "hi"
    assert registry.call_log == [log]


def test_invoke_logs_failure_without_raising():
    registry = make_registry()
    log = registry.invoke("boom", SpecialistType.RESEARCH)
    assert log.success is False
    assert "kaboom" in log.error


def test_unknown_tool_raises():
    registry = make_registry()
    with pytest.raises(ToolNotFound):
        registry.invoke("nonexistent", SpecialistType.RESEARCH)


def test_disallowed_specialist_raises():
    registry = make_registry()
    with pytest.raises(ToolNotAllowed):
        registry.invoke("echo", SpecialistType.WRITING, text="hi")


def test_rate_limit_enforced():
    registry = make_registry()
    registry.invoke("echo", SpecialistType.RESEARCH, text="1")
    registry.invoke("echo", SpecialistType.RESEARCH, text="2")
    with pytest.raises(RateLimitExceeded):
        registry.invoke("echo", SpecialistType.RESEARCH, text="3")
