"""Memory dashboard: what the system remembers about a given user, with
maintenance actions (consolidate, expire) and a per-user/per-memory delete for
data-deletion requests.
"""
import streamlit as st

from agent_orchestrator.memory.long_term import LongTermMemory

st.title("Memory Dashboard")

memory = LongTermMemory()

user_id = st.text_input("User ID", value="anonymous")

col1, col2, col3 = st.columns(3)
if col1.button("Consolidate duplicates"):
    merged = memory.consolidate(user_id)
    st.success(f"Merged {merged} near-duplicate memories.")
if col2.button("Expire stale memories (all users)"):
    expired = memory.expire()
    st.success(f"Expired {expired} memories.")
if col3.button("Delete ALL memories for this user", type="primary"):
    deleted = memory.delete_user_memories(user_id)
    st.warning(f"Deleted {deleted} memories for '{user_id}'.")

records = memory.list_user_memories(user_id)
records.sort(key=lambda r: memory.effective_importance(r), reverse=True)

st.caption(f"{len(records)} memories for '{user_id}'")

for record in records:
    score = memory.effective_importance(record)
    with st.expander(f"{record.task}  (importance {score:.2f})"):
        st.write("**Approach:**", record.approach_summary)
        if record.facts:
            st.write("**Facts:**", record.facts)
        if record.preferences:
            st.write("**Preferences:**", record.preferences)
        st.write("**Tools used:**", record.tools_used or "none")
        st.write(
            f"Access count: {record.access_count} · "
            f"Created: {record.created_at} · Last accessed: {record.last_accessed_at}"
        )
        if st.button("Delete this memory", key=f"delete-{record.id}"):
            memory.delete(record.id)
            st.rerun()
