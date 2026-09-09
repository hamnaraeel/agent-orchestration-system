"""Trace explorer: pick a past task, see its execution as an ordered list of
color-coded spans (agent, decision, tools called, latency, cost, errors),
expand any span for its full context (LLM prompt/response included), and
browse cost/performance analytics across every task that's been run.
"""
from datetime import datetime, timezone

import streamlit as st

from agent_orchestrator.config import settings
from agent_orchestrator.tracing.store import TraceStore

st.title("Trace Explorer")

# Material Symbols (Streamlit's built-in icon set) -- bare names here; each
# use site adds the ":material/" wrapper the widget it's passed to expects.
STATUS_ICON = {
    "success": "check_circle",
    "warning": "warning",
    "failure": "cancel",
    "escalated": "priority_high",
}
_DEFAULT_ICON = "help"


def _material(status: str) -> str:
    return f":material/{STATUS_ICON.get(status, _DEFAULT_ICON)}:"


@st.cache_resource
def get_store() -> TraceStore:
    return TraceStore(settings.trace_db_path)


def _fmt_time(epoch: float | None) -> str:
    if not epoch:
        return "-"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


store = get_store()

tab_traces, tab_analytics = st.tabs(["Traces", "Cost & Performance"])

with tab_traces:
    user_filter = st.text_input("Filter by user_id (optional)")
    tasks = store.list_tasks(limit=100, user_id=user_filter or None)

    if not tasks:
        st.info("No tasks recorded yet. Run one with `python -m agent_orchestrator.run \"...\"`.")
        st.stop()

    labels = [
        f"[{t.status}] {_fmt_time(t.started_at)} · {t.task_text[:60]} "
        f"(${t.total_cost_usd:.4f})"
        for t in tasks
    ]
    choice = st.selectbox("Task", options=range(len(tasks)), format_func=lambda i: labels[i])
    task = tasks[choice]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", task.status, icon=_material(task.status))
    col2.metric("Wall clock", f"{(task.wall_clock_ms or 0) / 1000:.2f}s")
    col3.metric("Tokens", f"{task.total_input_tokens + task.total_output_tokens:,}")
    col4.metric("Cost", f"${task.total_cost_usd:.4f}")
    col5.metric("Tool calls", task.total_tool_calls)

    if task.human_review_ms:
        st.caption(f"Human review time: {task.human_review_ms / 1000:.1f}s")
    if task.escalation_count:
        st.caption(f"Escalated {task.escalation_count} time(s)")

    st.subheader("Task")
    st.write(task.task_text)

    st.subheader("Spans")
    spans = store.get_spans(task.task_id)
    for span in spans:
        header = f"`{span.node_name}` — {span.agent} — {span.latency_ms:.0f}ms"
        if span.input_tokens or span.output_tokens:
            header += f" — {span.input_tokens + span.output_tokens} tok — ${span.cost_usd:.5f}"
        with st.expander(header, icon=_material(span.status)):
            attrs = span.attributes
            if attrs.get("prompt"):
                st.markdown("**Prompt:**")
                st.code(attrs["prompt"])
            if attrs.get("response"):
                st.markdown("**Response:**")
                st.code(
                    attrs["response"]
                    if isinstance(attrs["response"], str)
                    else str(attrs["response"])
                )
            if attrs.get("error"):
                st.error(attrs["error"])
            if attrs.get("tool_calls"):
                st.markdown("**Tool calls:**")
                st.json(attrs["tool_calls"])
            other = {
                k: v
                for k, v in attrs.items()
                if k not in ("prompt", "response", "error", "tool_calls")
            }
            if other:
                st.markdown("**Other attributes:**")
                st.json(other)

with tab_analytics:
    st.subheader("Cost by task type")
    st.dataframe(store.cost_by_task_type(), use_container_width=True)

    st.subheader("Most expensive agents")
    st.dataframe(store.most_expensive_agents(), use_container_width=True)

    st.subheader("Tool usage patterns")
    tool_patterns = store.tool_usage_patterns()
    if tool_patterns:
        st.dataframe(tool_patterns, use_container_width=True)
    else:
        st.caption("No tool calls recorded yet.")

    st.subheader("Escalation rate trends")
    st.dataframe(store.escalation_rate_trends(), use_container_width=True)
