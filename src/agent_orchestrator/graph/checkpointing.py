"""Checkpointer construction, shared between the graph's fast in-memory
default (used by tests and anything that doesn't need cross-process replay),
a persistent SQLite-backed one (single-machine local/dev use), and a
Postgres-backed one (docker-compose's "persistent state" service, for
multiple API/worker processes sharing one checkpoint store).
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from .. import schemas
from ..memory import models as memory_models


def _msgpack_allowlist_for(*modules) -> list[tuple[str, str]]:
    """Every class defined in `modules`, as (module, name) pairs -- lets the
    checkpointer's msgpack serializer round-trip our schema/memory types
    without the "unregistered type" deprecation warning (and without falling
    back to pickle, which `LANGGRAPH_STRICT_MSGPACK=true` would otherwise
    block for anything not on this list)."""
    return [
        (module.__name__, name)
        for module in modules
        for name, obj in vars(module).items()
        if isinstance(obj, type) and obj.__module__ == module.__name__
    ]


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=_msgpack_allowlist_for(schemas, memory_models)
    )


def default_checkpointer() -> MemorySaver:
    """Fast, in-process, and forgotten when the process exits."""
    return MemorySaver(serde=_serde())


def sqlite_checkpointer(db_path: str) -> SqliteSaver:
    """Persists checkpoints to a file, so a paused task survives a restart
    and `get_state_history` has real history to replay/fork from across
    separate CLI or API invocations."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn, serde=_serde())


def postgres_checkpointer(conn_string: str) -> PostgresSaver:
    """Persists checkpoints to Postgres -- unlike `sqlite_checkpointer`, this
    can be shared by multiple API/worker processes/containers at once, which
    is why docker-compose uses it instead."""
    import psycopg

    conn = psycopg.connect(conn_string, autocommit=True)
    saver = PostgresSaver(conn, serde=_serde())
    saver.setup()
    return saver
