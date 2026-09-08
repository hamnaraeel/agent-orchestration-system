"""Short-term working memory: a Redis hash per task run, holding the current
plan, completed subtask results, and any error logs so every agent (and, later,
the trace/review UIs) can see the same in-flight state. Scoped to one task and
cleared when that task finishes; a TTL is set as a safety net in case a crash
skips the explicit `clear()`.
"""
from __future__ import annotations

import json
from typing import Any, Protocol


class RedisLike(Protocol):
    """The subset of the redis-py interface we depend on (also satisfied by fakeredis)."""

    def hset(self, name: str, key: str, value: str) -> Any: ...
    def hget(self, name: str, key: str) -> Any: ...
    def hgetall(self, name: str) -> Any: ...
    def expire(self, name: str, ttl: int) -> Any: ...
    def delete(self, *names: str) -> Any: ...


class WorkingMemory:
    def __init__(self, redis_client: RedisLike, ttl_seconds: int = 3600) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int = 3600) -> "WorkingMemory":
        import redis

        return cls(redis.from_url(url, decode_responses=True), ttl_seconds)

    @staticmethod
    def _key(task_id: str) -> str:
        return f"task:{task_id}:working_memory"

    def save(self, task_id: str, field: str, value: Any) -> None:
        key = self._key(task_id)
        self.redis.hset(key, field, json.dumps(value, default=str))
        self.redis.expire(key, self.ttl_seconds)

    def get(self, task_id: str, field: str) -> Any | None:
        raw = self.redis.hget(self._key(task_id), field)
        return json.loads(raw) if raw is not None else None

    def get_all(self, task_id: str) -> dict[str, Any]:
        raw = self.redis.hgetall(self._key(task_id))
        return {field: json.loads(value) for field, value in raw.items()}

    def clear(self, task_id: str) -> None:
        self.redis.delete(self._key(task_id))
