"""Time-travel replay: load a past task, step through every checkpoint it
passed through, fork one with a modified value, resume, and compare the
replayed outcome against the original.

Needs a task run through the CLI first (which checkpoints to
`settings.checkpoint_db_path`, not just in memory) -- copy its `task_id` from
the CLI's output or the trace explorer.

Run with: streamlit run ui/replay.py
"""
import json

import streamlit as st

from agent_orchestrator.run import build_app
from agent_orchestrator.tracing.replay import list_checkpoints, replay_from_checkpoint

st.set_page_config(page_title="Replay Debugger", layout="wide")
st.title("Replay Debugger")


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@st.cache_resource
def get_app():
    app, _, _ = build_app(enable_memory=True)
    return app


app = get_app()

task_id = st.text_input("Task ID (from a previous CLI run)")
if not task_id:
    st.info("Paste a task_id to load its checkpoint history.")
    st.stop()

try:
    checkpoints = list_checkpoints(app, task_id)
except Exception as exc:  # noqa: BLE001
    st.error(f"Couldn't load checkpoints for '{task_id}': {exc}")
    st.stop()

if not checkpoints:
    st.warning("No checkpoints found for that task_id.")
    st.stop()

st.caption(f"{len(checkpoints)} checkpoints for this task.")

labels = [
    f"{i}: about to run {list(c.next_nodes) or '(none -- terminal)'}"
    for i, c in enumerate(checkpoints)
]
index = st.selectbox("Step", options=range(len(checkpoints)), format_func=lambda i: labels[i])
checkpoint = checkpoints[index]

st.subheader("State at this step")
st.json(_jsonable(checkpoint.values))

st.subheader("Modify and replay from here")
st.caption(
    "Enter a JSON object of the fields to overwrite at this step (merged into "
    "the state above), then resume execution from there."
)
default_update = "{\n  \n}"
state_update_text = st.text_area("State update (JSON)", value=default_update, height=120)

if st.button("Replay from this step", type="primary"):
    try:
        state_update = json.loads(state_update_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
    else:
        result = replay_from_checkpoint(app, checkpoint, state_update)
        st.session_state["replay_result"] = _jsonable(
            {k: v for k, v in result.items() if k != "__interrupt__"}
        )
        st.session_state["replay_interrupted"] = "__interrupt__" in result

if "replay_result" in st.session_state:
    col_original, col_replayed = st.columns(2)
    with col_original:
        st.markdown("**Original run's final state**")
        st.json(_jsonable(checkpoints[-1].values))
    with col_replayed:
        st.markdown("**Replayed final state**")
        if st.session_state.get("replay_interrupted"):
            st.warning("The replayed run hit another escalation and is paused.")
        st.json(st.session_state["replay_result"])
