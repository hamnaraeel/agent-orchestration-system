import fakeredis

from agent_orchestrator.memory.working import WorkingMemory


def make_memory() -> WorkingMemory:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return WorkingMemory(client, ttl_seconds=60)


def test_save_and_get_round_trips_json():
    memory = make_memory()
    memory.save("task-1", "plan", {"goal": "do the thing", "steps": [1, 2, 3]})
    assert memory.get("task-1", "plan") == {"goal": "do the thing", "steps": [1, 2, 3]}


def test_get_missing_field_returns_none():
    memory = make_memory()
    assert memory.get("task-1", "nonexistent") is None


def test_get_all_returns_every_saved_field():
    memory = make_memory()
    memory.save("task-1", "task", "summarize X")
    memory.save("task-1", "subtask:st-1", {"success": True})
    all_fields = memory.get_all("task-1")
    assert all_fields == {"task": "summarize X", "subtask:st-1": {"success": True}}


def test_clear_removes_all_fields_for_the_task():
    memory = make_memory()
    memory.save("task-1", "task", "summarize X")
    memory.clear("task-1")
    assert memory.get_all("task-1") == {}


def test_scoped_to_task_id_does_not_leak_between_tasks():
    memory = make_memory()
    memory.save("task-1", "task", "task one")
    memory.save("task-2", "task", "task two")
    assert memory.get("task-1", "task") == "task one"
    assert memory.get("task-2", "task") == "task two"


def test_ttl_is_set_on_save():
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    memory = WorkingMemory(client, ttl_seconds=60)
    memory.save("task-1", "task", "summarize X")
    ttl = client.ttl("task:task-1:working_memory")
    assert 0 < ttl <= 60
