import threading
import time

import fakeredis
import pytest

from agent_orchestrator.human_loop.approval_queue import ApprovalQueue
from agent_orchestrator.human_loop.models import ApprovalStatus, PendingApproval
from agent_orchestrator.schemas import DecisionAction, EscalationLevel, EscalationRequest, HumanDecision


def make_queue() -> ApprovalQueue:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return ApprovalQueue(client, poll_interval=0.05)


def make_escalation(**overrides) -> EscalationRequest:
    defaults = dict(level=EscalationLevel.APPROVE_PLAN, reason="Low confidence.", context={})
    defaults.update(overrides)
    return EscalationRequest(**defaults)


def test_submit_then_get_round_trips():
    queue = make_queue()
    approval = PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence")
    queue.submit(approval)

    fetched = queue.get("t1")
    assert fetched is not None
    assert fetched.task_id == "t1"
    assert fetched.status == ApprovalStatus.PENDING


def test_get_missing_returns_none():
    queue = make_queue()
    assert queue.get("nonexistent") is None


def test_list_pending_only_includes_unresolved():
    queue = make_queue()
    queue.submit(PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence"))
    queue.submit(PendingApproval(task_id="t2", escalation=make_escalation(), source="specialist_retry"))
    queue.resolve("t1", HumanDecision(action=DecisionAction.APPROVE))

    pending = {a.task_id for a in queue.list_pending()}
    assert pending == {"t2"}


def test_list_all_includes_resolved_and_pending():
    queue = make_queue()
    queue.submit(PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence"))
    queue.resolve("t1", HumanDecision(action=DecisionAction.APPROVE))
    queue.submit(PendingApproval(task_id="t2", escalation=make_escalation(), source="specialist_retry"))

    all_ids = {a.task_id for a in queue.list_all()}
    assert all_ids == {"t1", "t2"}


def test_resolve_records_decision_and_removes_from_pending():
    queue = make_queue()
    queue.submit(PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence"))

    decision = HumanDecision(action=DecisionAction.MODIFY, feedback="Try a narrower scope.")
    resolved = queue.resolve("t1", decision)

    assert resolved.status == ApprovalStatus.RESOLVED
    assert resolved.decision == decision
    assert resolved.resolved_at is not None
    assert queue.get("t1").decision == decision
    assert queue.list_pending() == []


def test_resolve_missing_raises():
    queue = make_queue()
    with pytest.raises(KeyError):
        queue.resolve("nonexistent", HumanDecision(action=DecisionAction.APPROVE))


def test_notify_records_a_non_blocking_informational_entry():
    queue = make_queue()
    escalation = make_escalation(level=EscalationLevel.NOTIFY, reason="Low score, proceeding.")
    queue.notify("t1", escalation)

    approval = queue.get("t1")
    assert approval.status == ApprovalStatus.NOTIFIED
    assert approval.resolved_at is not None
    assert queue.list_pending() == []  # never blocks anything


def test_append_message_builds_a_chat_transcript():
    queue = make_queue()
    queue.submit(PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence"))

    queue.append_message("t1", "human", "Why is confidence so low?")
    queue.append_message("t1", "agent", "The request was ambiguous about scope.")

    messages = queue.get("t1").messages
    assert [(m.role, m.content) for m in messages] == [
        ("human", "Why is confidence so low?"),
        ("agent", "The request was ambiguous about scope."),
    ]


def test_wait_for_decision_blocks_until_resolved():
    queue = make_queue()
    queue.submit(PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence"))
    decision = HumanDecision(action=DecisionAction.APPROVE)

    def resolve_after_delay():
        time.sleep(0.1)
        queue.resolve("t1", decision)

    thread = threading.Thread(target=resolve_after_delay)
    thread.start()
    result = queue.wait_for_decision("t1")
    thread.join()

    assert result == decision


def test_wait_for_decision_times_out():
    queue = make_queue()
    queue.submit(PendingApproval(task_id="t1", escalation=make_escalation(), source="plan_low_confidence"))

    with pytest.raises(TimeoutError):
        queue.wait_for_decision("t1", timeout=0.1)
