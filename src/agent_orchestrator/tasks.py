"""Celery wiring for distributed specialist execution.

Off by default (`settings.use_celery_for_specialists = False`): LangGraph
already runs specialists concurrently within one process via `Send`, which is
enough for a single-machine demo. Celery adds a genuinely different
capability on top -- specialist work runs in a *separate worker process* (or
pool of them, on other machines), so a crashed/slow specialist can't take
down the orchestrator, and the worker pool scales independently. `run_specialist`
in `graph/build.py` calls `run_specialist_task.delay(...).get(...)` instead of
calling a `SpecialistAgent` directly when this is turned on.

The worker process doesn't share the orchestrator's Python objects, so this
task reconstructs a `ToolRegistry` and `SpecialistAgent` from scratch on every
call (register_builtin_tools() is deterministic, so this is equivalent to,
just not the same object as, the orchestrator's registry -- rate limits reset
per call, which is an acceptable tradeoff for a demo-scale system).

Run a worker with: celery -A agent_orchestrator.tasks worker --loglevel=info
"""
from __future__ import annotations

from celery import Celery

from .agents.specialists import SpecialistAgent
from .config import settings
from .schemas import SpecialistType, SubTask, SubtaskResult
from .tools.builtin import register_builtin_tools
from .tools.registry import ToolRegistry

celery_app = Celery(
    "agent_orchestrator",
    broker=settings.celery_broker_url or settings.redis_url,
    backend=settings.celery_result_backend or settings.redis_url,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]


@celery_app.task(name="agent_orchestrator.run_specialist")
def run_specialist_task(
    specialist_type_value: str,
    model_name: str,
    subtask_data: dict,
    context_data: dict,
    feedback: str | None,
) -> dict:
    specialist_type = SpecialistType(specialist_type_value)
    subtask = SubTask.model_validate(subtask_data)
    context = {sid: SubtaskResult.model_validate(r) for sid, r in context_data.items()}

    registry = ToolRegistry()
    register_builtin_tools(registry)
    agent = SpecialistAgent(specialist_type, model_name, registry)

    result = agent.run(subtask, context, feedback=feedback)
    return {
        "subtask_result": result.model_dump(),
        "model": model_name,
        "prompt": agent.last_prompt,
        "response": agent.last_response_text,
        "usage": (
            {
                "input_tokens": agent.last_usage.input_tokens,
                "output_tokens": agent.last_usage.output_tokens,
            }
            if agent.last_usage is not None
            else None
        ),
    }
