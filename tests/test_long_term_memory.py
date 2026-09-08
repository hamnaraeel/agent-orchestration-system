from datetime import timedelta

import pytest

from agent_orchestrator.memory.long_term import LongTermMemory, _utcnow
from agent_orchestrator.memory.models import MemoryRecord
from tests.fake_embedding import FakeEmbeddingFunction


def make_memory(tmp_path, **overrides) -> LongTermMemory:
    return LongTermMemory(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_memories",
        embedding_function=FakeEmbeddingFunction(),
        half_life_days=overrides.get("half_life_days", 30.0),
        min_importance=overrides.get("min_importance", 0.05),
        min_age_days_before_expiry=overrides.get("min_age_days_before_expiry", 7.0),
        consolidation_similarity_threshold=overrides.get(
            "consolidation_similarity_threshold", 0.93
        ),
    )


def make_record(**overrides) -> MemoryRecord:
    defaults = dict(
        user_id="alice",
        task="Summarize quarterly earnings for AcmeCo",
        approach_summary="Pulled the 10-Q, extracted revenue and margin figures, wrote a two-paragraph summary.",
        tools_used=["web_search", "file_write"],
        facts=["AcmeCo's fiscal year ends in June."],
        preferences=["Prefers concise, bullet-point summaries."],
        importance=0.6,
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def test_add_then_query_finds_similar_memory(tmp_path):
    memory = make_memory(tmp_path)
    record = make_record()
    memory.add(record)

    results = memory.query("earnings summary for AcmeCo", user_id="alice", top_k=1)

    assert len(results) == 1
    assert results[0].id == record.id
    assert results[0].task == record.task


def test_query_bumps_access_count(tmp_path):
    memory = make_memory(tmp_path)
    record = make_record()
    memory.add(record)

    results = memory.query("earnings summary", user_id="alice", top_k=1)
    assert results[0].access_count == 1

    results_again = memory.query("earnings summary", user_id="alice", top_k=1)
    assert results_again[0].access_count == 2


def test_query_scoped_to_user_id(tmp_path):
    memory = make_memory(tmp_path)
    memory.add(make_record(user_id="alice"))
    memory.add(make_record(user_id="bob", task="Bob's unrelated task about kayaks"))

    results = memory.query("earnings", user_id="bob", top_k=5)
    assert all(r.user_id == "bob" for r in results)


def test_list_and_delete_user_memories(tmp_path):
    memory = make_memory(tmp_path)
    memory.add(make_record(user_id="alice"))
    memory.add(make_record(user_id="alice", task="Second task"))
    memory.add(make_record(user_id="bob"))

    assert len(memory.list_user_memories("alice")) == 2

    deleted = memory.delete_user_memories("alice")
    assert deleted == 2
    assert memory.list_user_memories("alice") == []
    assert len(memory.list_user_memories("bob")) == 1


def test_effective_importance_decays_with_age(tmp_path):
    memory = make_memory(tmp_path, half_life_days=10.0)
    fresh = make_record(importance=0.5)
    stale = make_record(importance=0.5, last_accessed_at=_utcnow() - timedelta(days=10))

    now = _utcnow()
    assert memory.effective_importance(stale, now) == pytest.approx(
        memory.effective_importance(fresh, now) / 2
    )


def test_consolidate_merges_near_duplicate_memories(tmp_path):
    memory = make_memory(tmp_path, consolidation_similarity_threshold=0.5)
    memory.add(make_record(task="Summarize AcmeCo Q3 earnings", facts=["fact A"]))
    memory.add(make_record(task="Summarize AcmeCo Q3 earnings report", facts=["fact B"]))

    merged = memory.consolidate("alice")

    assert merged == 1
    remaining = memory.list_user_memories("alice")
    assert len(remaining) == 1
    assert set(remaining[0].facts) == {"fact A", "fact B"}


def test_expire_removes_low_importance_memories_past_grace_period(tmp_path):
    memory = make_memory(tmp_path, min_importance=0.2, min_age_days_before_expiry=1.0)
    old_and_unimportant = make_record(
        importance=0.05,
        created_at=_utcnow() - timedelta(days=30),
        last_accessed_at=_utcnow() - timedelta(days=30),
    )
    fresh_but_unimportant = make_record(importance=0.05)
    old_and_important = make_record(
        importance=0.9,
        created_at=_utcnow() - timedelta(days=30),
        last_accessed_at=_utcnow() - timedelta(days=30),
    )
    memory.add(old_and_unimportant)
    memory.add(fresh_but_unimportant)
    memory.add(old_and_important)

    expired = memory.expire("alice")

    assert expired == 1
    remaining_ids = {r.id for r in memory.list_user_memories("alice")}
    assert old_and_unimportant.id not in remaining_ids
    assert fresh_but_unimportant.id in remaining_ids  # too young to expire yet
    assert old_and_important.id in remaining_ids  # still important enough
