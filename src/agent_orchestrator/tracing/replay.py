"""Time-travel replay/debugging, built directly on LangGraph's checkpoint
history rather than a custom re-run-from-scratch mechanism: `get_state_history`
gives every checkpoint a past run passed through, and `update_state` on a
specific one forks a new branch from that exact point with a modified value,
which `invoke(None, ...)` then resumes -- only the downstream nodes re-run,
so a step's upstream history (and the original run's checkpoints) stay intact
for comparison.

This only works against a *persistent* checkpointer (SqliteSaver) -- the
in-memory default a fresh `build_graph()` call gets for tests has no history
once that process exits, which is exactly why the CLI/API build their graph
with `graph.checkpointing.sqlite_checkpointer` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.graph.state import CompiledStateGraph


@dataclass
class CheckpointSummary:
    checkpoint_id: str
    next_nodes: tuple[str, ...]
    config: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)


def list_checkpoints(app: CompiledStateGraph, task_id: str) -> list[CheckpointSummary]:
    """Every checkpoint for this task, oldest first."""
    config = {"configurable": {"thread_id": task_id}}
    history = list(app.get_state_history(config))
    history.reverse()
    return [
        CheckpointSummary(
            checkpoint_id=snapshot.config["configurable"]["checkpoint_id"],
            next_nodes=snapshot.next,
            config=snapshot.config,
            values=snapshot.values,
        )
        for snapshot in history
    ]


def replay_from_checkpoint(
    app: CompiledStateGraph,
    checkpoint: CheckpointSummary,
    state_update: dict,
) -> dict:
    """Fork the run at `checkpoint` with `state_update` merged in, then
    resume from there. The fork is a new checkpoint on the *same* thread, so
    the original checkpoints this branched from are untouched and still
    retrievable via `list_checkpoints` for comparison."""
    forked_config = app.update_state(checkpoint.config, state_update)
    return app.invoke(None, forked_config)
