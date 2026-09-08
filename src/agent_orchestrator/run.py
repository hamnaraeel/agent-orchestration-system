"""CLI entry point: runs a single task through the full orchestrator graph."""
from __future__ import annotations

import argparse
import json

from .agents.reviewer import ReviewerAgent
from .agents.specialists import build_specialists
from .agents.supervisor import SupervisorAgent
from .config import settings
from .graph.build import build_graph
from .schemas import SpecialistType
from .tools.builtin import register_builtin_tools
from .tools.registry import ToolRegistry


def build_app():
    registry = ToolRegistry()
    register_builtin_tools(registry)

    specialist_models = {
        specialist_type: settings.specialist_model for specialist_type in SpecialistType
    }
    specialists = build_specialists(registry, specialist_models)
    supervisor = SupervisorAgent(settings.supervisor_model)
    reviewer = ReviewerAgent(settings.reviewer_model)

    return build_graph(supervisor, reviewer, specialists), registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a task through the agent orchestrator.")
    parser.add_argument("task", help="The task to run.")
    args = parser.parse_args()

    app, registry = build_app()
    final_state = app.invoke({"task": args.task, "memory_context": ""})

    print(
        json.dumps(
            {
                "status": final_state.get("status"),
                "final_output": final_state.get("final_output"),
                "escalation": (
                    final_state["escalation"].model_dump()
                    if final_state.get("escalation")
                    else None
                ),
                "tool_calls": len(registry.call_log),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
