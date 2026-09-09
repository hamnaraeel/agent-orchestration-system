"""Single entry point for every operator-facing UI: trace explorer, human
review queue, memory dashboard, and replay debugger, as pages of one
Streamlit app (one process, one port) rather than four separate ones.

Run with: streamlit run ui/app.py
"""
import streamlit as st

st.set_page_config(page_title="Agent Orchestrator", layout="wide")

pages = [
    st.Page(
        "pages/trace_explorer.py",
        title="Trace Explorer",
        icon=":material/query_stats:",
        default=True,
    ),
    st.Page("pages/review_queue.py", title="Review Queue", icon=":material/gavel:"),
    st.Page("pages/memory_dashboard.py", title="Memory Dashboard", icon=":material/memory:"),
    st.Page("pages/replay.py", title="Replay Debugger", icon=":material/replay:"),
]

st.navigation(pages).run()
