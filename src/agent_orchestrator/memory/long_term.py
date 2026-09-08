"""Long-term semantic memory: a ChromaDB collection of past-task memories.

Each memory is embedded from its task + approach + facts + preferences text so
similar future tasks can retrieve it. Retrieval bumps `access_count` and
`last_accessed_at`, which feeds `effective_importance` (recency + frequency
decayed relevance) -- the score used to rank the dashboard and drive
expiration. `consolidate` merges near-duplicate memories for a user so the
collection doesn't just grow linearly with every run of a repeated task.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from ..config import settings
from .models import MemoryRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LongTermMemory:
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_function=None,
        half_life_days: float | None = None,
        min_importance: float | None = None,
        min_age_days_before_expiry: float | None = None,
        consolidation_similarity_threshold: float | None = None,
    ) -> None:
        import chromadb

        # An explicit persist_directory (tests, or a caller that wants local
        # mode regardless of global config) always wins; otherwise follow
        # settings.chroma_mode ("http" talks to a Chroma server container --
        # see docker-compose.yml -- "persistent" writes to a local directory).
        if persist_directory is None and settings.chroma_mode == "http":
            self._client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        else:
            self._client = chromadb.PersistentClient(
                path=persist_directory or settings.chroma_persist_dir
            )
        collection_kwargs: dict = {"metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            collection_kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.memory_collection_name, **collection_kwargs
        )

        self.half_life_days = half_life_days or settings.memory_importance_half_life_days
        self.min_importance = (
            settings.memory_min_importance if min_importance is None else min_importance
        )
        self.min_age_days_before_expiry = (
            settings.memory_min_age_days_before_expiry
            if min_age_days_before_expiry is None
            else min_age_days_before_expiry
        )
        self.consolidation_similarity_threshold = (
            consolidation_similarity_threshold
            or settings.memory_consolidation_similarity_threshold
        )

    # -- (de)serialization -------------------------------------------------

    @staticmethod
    def _to_metadata(record: MemoryRecord) -> dict:
        return {
            "user_id": record.user_id,
            "task": record.task,
            "approach_summary": record.approach_summary,
            "tools_used": json.dumps(record.tools_used),
            "facts": json.dumps(record.facts),
            "preferences": json.dumps(record.preferences),
            "success": record.success,
            "importance": record.importance,
            "access_count": record.access_count,
            "created_at": record.created_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat(),
        }

    @staticmethod
    def _from_result(memory_id: str, metadata: dict) -> MemoryRecord:
        return MemoryRecord(
            id=memory_id,
            user_id=metadata["user_id"],
            task=metadata["task"],
            approach_summary=metadata["approach_summary"],
            tools_used=json.loads(metadata.get("tools_used", "[]")),
            facts=json.loads(metadata.get("facts", "[]")),
            preferences=json.loads(metadata.get("preferences", "[]")),
            success=bool(metadata.get("success", True)),
            importance=float(metadata.get("importance", 0.5)),
            access_count=int(metadata.get("access_count", 0)),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            last_accessed_at=datetime.fromisoformat(metadata["last_accessed_at"]),
        )

    # -- writes --------------------------------------------------------

    def add(self, record: MemoryRecord) -> None:
        self._collection.add(
            ids=[record.id],
            documents=[record.embedding_text()],
            metadatas=[self._to_metadata(record)],
        )

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def delete_user_memories(self, user_id: str) -> int:
        records = self.list_user_memories(user_id)
        if records:
            self._collection.delete(where={"user_id": user_id})
        return len(records)

    # -- reads -----------------------------------------------------------

    def list_user_memories(self, user_id: str) -> list[MemoryRecord]:
        results = self._collection.get(where={"user_id": user_id})
        return [
            self._from_result(memory_id, metadata)
            for memory_id, metadata in zip(results["ids"], results["metadatas"])
        ]

    def query(
        self, query_text: str, user_id: str | None = None, top_k: int | None = None
    ) -> list[MemoryRecord]:
        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k or settings.memory_top_k,
            where={"user_id": user_id} if user_id else None,
        )
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        records = [self._from_result(i, m) for i, m in zip(ids, metadatas)]
        for record in records:
            self._bump_access(record)
        return records

    def _bump_access(self, record: MemoryRecord) -> None:
        record.access_count += 1
        record.last_accessed_at = _utcnow()
        self._collection.update(ids=[record.id], metadatas=[self._to_metadata(record)])

    # -- importance, consolidation, expiration ----------------------------

    def effective_importance(self, record: MemoryRecord, now: datetime | None = None) -> float:
        now = now or _utcnow()
        age_days = max((now - record.last_accessed_at).total_seconds() / 86400, 0.0)
        decay = 0.5 ** (age_days / self.half_life_days)
        frequency_boost = 1 + math.log1p(record.access_count)
        return record.importance * frequency_boost * decay

    def consolidate(self, user_id: str) -> int:
        """Merge memories for `user_id` that are near-duplicates of each other.
        Returns how many records were merged away."""
        records = {r.id: r for r in self.list_user_memories(user_id)}
        if len(records) < 2:
            return 0

        consumed: set[str] = set()
        for record in list(records.values()):
            if record.id in consumed:
                continue
            neighbors = self._collection.query(
                query_texts=[record.embedding_text()],
                n_results=min(5, len(records)),
                where={"user_id": user_id},
            )
            for neighbor_id, distance in zip(neighbors["ids"][0], neighbors["distances"][0]):
                if neighbor_id == record.id or neighbor_id in consumed:
                    continue
                similarity = 1 - distance
                if similarity < self.consolidation_similarity_threshold:
                    continue
                other = records[neighbor_id]
                record.facts = list(dict.fromkeys(record.facts + other.facts))
                record.preferences = list(dict.fromkeys(record.preferences + other.preferences))
                record.tools_used = list(dict.fromkeys(record.tools_used + other.tools_used))
                record.access_count += other.access_count
                record.importance = max(record.importance, other.importance)
                consumed.add(other.id)

            if record.id not in consumed:
                self._collection.update(ids=[record.id], metadatas=[self._to_metadata(record)])

        if consumed:
            self._collection.delete(ids=list(consumed))
        return len(consumed)

    def expire(self, user_id: str | None = None) -> int:
        """Delete memories whose decayed importance has fallen below the
        minimum threshold, once they're past the initial grace period."""
        if user_id:
            records = self.list_user_memories(user_id)
        else:
            results = self._collection.get()
            records = [
                self._from_result(memory_id, metadata)
                for memory_id, metadata in zip(results["ids"], results["metadatas"])
            ]

        now = _utcnow()
        to_delete = [
            record.id
            for record in records
            if (now - record.created_at).total_seconds() / 86400
            >= self.min_age_days_before_expiry
            and self.effective_importance(record, now) < self.min_importance
        ]
        if to_delete:
            self._collection.delete(ids=to_delete)
        return len(to_delete)
