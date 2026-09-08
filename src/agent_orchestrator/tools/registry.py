"""Tool registry: every tool an agent can call is registered here with a schema,
the specialists allowed to use it, and a rate limit. Every invocation is logged.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from ..schemas import SpecialistType, ToolCallLog


@dataclass
class RegisteredTool:
    name: str
    description: str
    func: Callable[..., str]
    allowed_specialists: set[SpecialistType]
    args_schema: type[BaseModel]
    rate_limit_per_minute: int = 30

    def __post_init__(self) -> None:
        self._call_timestamps: deque[float] = deque()

    def _check_rate_limit(self) -> None:
        now = time.monotonic()
        window_start = now - 60
        while self._call_timestamps and self._call_timestamps[0] < window_start:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= self.rate_limit_per_minute:
            raise RateLimitExceeded(
                f"Tool '{self.name}' exceeded {self.rate_limit_per_minute} calls/minute."
            )
        self._call_timestamps.append(now)


class RateLimitExceeded(Exception):
    pass


class ToolNotFound(Exception):
    pass


class ToolNotAllowed(Exception):
    pass


class ToolRegistry:
    """Central registry of tools, gated by specialist type and rate limit.

    Every call, successful or not, is appended to `.call_log` as a `ToolCallLog`
    so the trace explorer (Phase 4) has a single place to pull tool activity from.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.call_log: list[ToolCallLog] = []

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        allowed_specialists: set[SpecialistType],
        args_schema: type[BaseModel],
        rate_limit_per_minute: int = 30,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            func=func,
            allowed_specialists=allowed_specialists,
            args_schema=args_schema,
            rate_limit_per_minute=rate_limit_per_minute,
        )

    def list_tools_for(self, specialist: SpecialistType) -> list[RegisteredTool]:
        return [t for t in self._tools.values() if specialist in t.allowed_specialists]

    def invoke(self, tool_name: str, specialist: SpecialistType, **kwargs) -> ToolCallLog:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolNotFound(f"No such tool: '{tool_name}'.")
        if specialist not in tool.allowed_specialists:
            raise ToolNotAllowed(
                f"Specialist '{specialist.value}' may not call tool '{tool_name}'."
            )

        tool._check_rate_limit()
        start = time.monotonic()
        try:
            output = tool.func(**kwargs)
            log = ToolCallLog(
                tool_name=tool_name,
                inputs=kwargs,
                output=output,
                success=True,
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures are data, not crashes
            log = ToolCallLog(
                tool_name=tool_name,
                inputs=kwargs,
                success=False,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )
        self.call_log.append(log)
        return log
