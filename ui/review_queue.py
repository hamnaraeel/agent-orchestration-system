"""Human review interface: browse pending escalations, inspect the full
context (task, plan, completed steps so far, the specific decision point,
relevant memories), ask the paused task's context a clarifying question, and
resolve with approve / reject / modify / take over.

Resolving here only writes to the approval queue -- it's whichever process is
blocked in `ApprovalQueue.wait_for_decision` (the CLI's `run_task`, or a
future worker) that actually resumes the paused graph.

Run with: streamlit run ui/review_queue.py
"""
import streamlit as st

from agent_orchestrator.agents.clarifier import ClarificationAgent
from agent_orchestrator.config import settings
from agent_orchestrator.human_loop.approval_queue import ApprovalQueue
from agent_orchestrator.schemas import DecisionAction, HumanDecision

st.set_page_config(page_title="Human Review Queue", layout="wide")
st.title("Human Review Queue")


@st.cache_resource
def get_queue() -> ApprovalQueue:
    return ApprovalQueue.from_url(settings.redis_url)


@st.cache_resource
def get_clarifier() -> ClarificationAgent:
    return ClarificationAgent(settings.reviewer_model)


queue = get_queue()
clarifier = get_clarifier()

pending = queue.list_pending()

if not pending:
    st.info("No pending approvals.")
    st.stop()

labels = [f"{a.task_id[:8]} · {a.source} ({a.escalation.level.value})" for a in pending]
choice = st.selectbox(
    "Pending escalations", options=range(len(pending)), format_func=lambda i: labels[i]
)
approval = pending[choice]

st.subheader(f"Task: {approval.escalation.context.get('task', '(unknown)')}")
st.write(
    f"**Level:** {approval.escalation.level.value}  \n"
    f"**Source:** {approval.source}  \n"
    f"**Reason:** {approval.escalation.reason}"
)

with st.expander("Full context (plan, completed steps, review, memory)", expanded=False):
    st.json(approval.escalation.context)

st.divider()
st.subheader("Ask a clarifying question")
for message in approval.messages:
    role = "You" if message.role == "human" else "Agent"
    st.markdown(f"**{role}:** {message.content}")

question = st.text_input("Question", key=f"question-{approval.task_id}")
if st.button("Ask") and question:
    answer = clarifier.answer(approval.escalation.context, question)
    queue.append_message(approval.task_id, "human", question)
    queue.append_message(approval.task_id, "agent", answer)
    st.rerun()

st.divider()
st.subheader("Decision")
action = st.radio("Action", ["approve", "reject", "modify", "take_over"], horizontal=True)
feedback = None
output = None
if action == "modify":
    feedback = st.text_area("Guidance for the agents")
elif action == "take_over":
    output = st.text_area("Your output")

if st.button("Submit decision", type="primary"):
    decision = HumanDecision(action=DecisionAction(action), feedback=feedback, output=output)
    queue.resolve(approval.task_id, decision)
    st.success("Decision recorded -- the waiting task will resume shortly.")
    st.rerun()
