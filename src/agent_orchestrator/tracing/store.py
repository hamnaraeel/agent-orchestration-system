"""Persistent trace/cost store, backed by plain sqlite3 (no ORM -- the schema
is small and the queries are simple aggregates).

Two tables:
  - `tasks`: one row per task run, with the totals the dashboard aggregates
    over (tokens, cost, tool calls, wall-clock time, human review time,
    whether it was escalated).
  - `spans`: one row per graph-node execution (plus one per individual tool
    call), each carrying its own agent/status/latency/tokens/cost and a JSON
    blob of node-specific attributes -- this is what the trace explorer walks
    to build its tree view.

`task_type` is a coarse heuristic (the sorted set of specialists a plan uses,
e.g. "research+writing"), not a real classifier -- documented as such rather
than presented as more precise than it is.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    user_id TEXT,
    task_text TEXT,
    task_type TEXT,
    status TEXT,
    started_at REAL,
    ended_at REAL,
    wall_clock_ms REAL,
    human_review_ms REAL DEFAULT 0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    escalation_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    node_name TEXT,
    agent TEXT,
    status TEXT,
    tool_name TEXT,
    started_at REAL,
    ended_at REAL,
    latency_ms REAL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    attributes_json TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);

CREATE INDEX IF NOT EXISTS idx_spans_task_id ON spans (task_id);
"""


@dataclass
class SpanRecord:
    id: int
    task_id: str
    node_name: str
    agent: str
    status: str
    started_at: float
    ended_at: float
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_name: str | None = None
    attributes: dict = field(default_factory=dict)


@dataclass
class TaskRecord:
    task_id: str
    user_id: str
    task_text: str
    task_type: str | None
    status: str
    started_at: float
    ended_at: float | None
    wall_clock_ms: float | None
    human_review_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_tool_calls: int
    escalation_count: int


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(**{k: row[k] for k in row.keys()})


def _row_to_span(row: sqlite3.Row) -> SpanRecord:
    data = {k: row[k] for k in row.keys() if k != "attributes_json"}
    data["attributes"] = json.loads(row["attributes_json"] or "{}")
    return SpanRecord(**data)


class TraceStore:
    def __init__(self, db_path: str = "./traces.db") -> None:
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def start_task(self, task_id: str, user_id: str, task_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks "
                "(task_id, user_id, task_text, status, started_at) "
                "VALUES (?, ?, ?, 'running', ?)",
                (task_id, user_id, task_text, time.time()),
            )

    def add_span(
        self,
        task_id: str,
        node_name: str,
        agent: str,
        status: str,
        started_at: float,
        ended_at: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        tool_name: str | None = None,
        attributes: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO spans "
                "(task_id, node_name, agent, status, tool_name, started_at, ended_at, "
                "latency_ms, input_tokens, output_tokens, cost_usd, attributes_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    node_name,
                    agent,
                    status,
                    tool_name,
                    started_at,
                    ended_at,
                    (ended_at - started_at) * 1000,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    json.dumps(attributes or {}, default=str),
                ),
            )

    def finish_task(
        self,
        task_id: str,
        status: str,
        task_type: str | None = None,
        human_review_ms: float = 0.0,
        escalation_count: int = 0,
    ) -> None:
        with self._connect() as conn:
            totals = conn.execute(
                "SELECT COALESCE(SUM(input_tokens),0) AS in_tok, "
                "COALESCE(SUM(output_tokens),0) AS out_tok, "
                "COALESCE(SUM(cost_usd),0) AS cost, "
                "COALESCE(SUM(CASE WHEN tool_name IS NOT NULL THEN 1 ELSE 0 END),0) AS tool_calls "
                "FROM spans WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            ended_at = time.time()
            started_at = conn.execute(
                "SELECT started_at FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()["started_at"]
            conn.execute(
                "UPDATE tasks SET status=?, task_type=?, ended_at=?, wall_clock_ms=?, "
                "human_review_ms=?, total_input_tokens=?, total_output_tokens=?, "
                "total_cost_usd=?, total_tool_calls=?, escalation_count=? WHERE task_id=?",
                (
                    status,
                    task_type,
                    ended_at,
                    (ended_at - started_at) * 1000,
                    human_review_ms,
                    totals["in_tok"],
                    totals["out_tok"],
                    totals["cost"],
                    totals["tool_calls"],
                    escalation_count,
                    task_id,
                ),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return _row_to_task(row) if row else None

    def list_tasks(self, limit: int = 50, user_id: str | None = None) -> list[TaskRecord]:
        query = "SELECT * FROM tasks"
        params: tuple = ()
        if user_id:
            query += " WHERE user_id = ?"
            params = (user_id,)
        query += " ORDER BY started_at DESC LIMIT ?"
        params = params + (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [_row_to_task(r) for r in rows]

    def get_spans(self, task_id: str) -> list[SpanRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE task_id = ? ORDER BY started_at ASC, id ASC",
                (task_id,),
            ).fetchall()
            return [_row_to_span(r) for r in rows]

    def cost_by_task_type(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT COALESCE(task_type, 'unknown') AS task_type, "
                "COUNT(*) AS task_count, SUM(total_cost_usd) AS total_cost_usd, "
                "AVG(total_cost_usd) AS avg_cost_usd "
                "FROM tasks WHERE status != 'running' "
                "GROUP BY task_type ORDER BY total_cost_usd DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def most_expensive_agents(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent, COUNT(*) AS call_count, "
                "SUM(cost_usd) AS total_cost_usd, "
                "SUM(input_tokens) AS total_input_tokens, "
                "SUM(output_tokens) AS total_output_tokens, "
                "AVG(latency_ms) AS avg_latency_ms "
                "FROM spans GROUP BY agent ORDER BY total_cost_usd DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def tool_usage_patterns(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tool_name, COUNT(*) AS call_count, "
                "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count, "
                "AVG(latency_ms) AS avg_latency_ms "
                "FROM spans WHERE tool_name IS NOT NULL "
                "GROUP BY tool_name ORDER BY call_count DESC"
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["success_rate"] = d["success_count"] / d["call_count"] if d["call_count"] else 0.0
                result.append(d)
            return result

    def escalation_rate_trends(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date(started_at, 'unixepoch') AS day, "
                "COUNT(*) AS task_count, "
                "SUM(CASE WHEN escalation_count > 0 THEN 1 ELSE 0 END) AS escalated_count "
                "FROM tasks WHERE status != 'running' "
                "GROUP BY day ORDER BY day ASC"
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["escalation_rate"] = (
                    d["escalated_count"] / d["task_count"] if d["task_count"] else 0.0
                )
                result.append(d)
            return result
