# Agent Orchestration System

A multi-agent orchestration platform where a supervisor agent decomposes complex tasks, delegates subtasks to specialized tool-using agents, maintains persistent memory across interactions, and escalates to a human operator when confidence is low or the task requires approval — with full observability into every agent decision.

> It's not an AI demo — it's production infrastructure for autonomous AI workflows.

## Tech Stack

| Component | Tool / Library | Why This Choice |
|---|---|---|
| Language | Python 3.11+ | Ecosystem standard |
| Orchestration | LangGraph | State machine for agent workflows |
| LLM Providers | OpenAI + Anthropic | Multi-model agent routing |
| Tool Framework | Custom + MCP | Extensible tool integration |
| Memory | PostgreSQL + ChromaDB | Short-term + semantic long-term |
| Queue | Redis + Celery | Async task execution |
| Review UI | React or Streamlit | Human-in-the-loop interface |
| Containerization | Docker + docker-compose | Full system orchestration |

## Project Layout

```
src/agent_orchestrator/
  schemas.py       # ExecutionPlan, SubTask, SubtaskResult, ReviewResult, EscalationRequest
  config.py        # env-driven settings: model routing, thresholds, sandbox paths
  tools/
    registry.py    # ToolRegistry: rate limits, specialist gating, call logging
    builtin.py     # web_search, file_read/write, code_execution, db_query, api_call
  agents/
    base.py        # BaseAgent: provider routing (OpenAI/Anthropic) + structured output
    supervisor.py  # task decomposition (create_plan) + final synthesis
    specialists.py # tool-calling loop per specialist (research/data/writing/code)
    reviewer.py    # validates aggregated outputs, flags redo/human review
  graph/
    state.py       # OrchestratorState (LangGraph state + merge reducers)
    build.py       # the state machine: plan -> dispatch -> review -> synthesize -> deliver
  run.py           # CLI entry point
tests/             # graph smoke tests + unit tests, all using a fake chat model (no API key needed)
```

Run it locally:

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
pytest -q
python -m agent_orchestrator.run "Research topic X and write a two-paragraph summary"
```

Note: `human_escalation` in the graph is currently a placeholder — it records an
`EscalationRequest` on the final state and the graph ends. The real approval
queue and review UI are built in Phase 3.

## Build Progress

Tracking checklist for the build guide below. Check items off as they're completed.

### Phase 1: Agent Architecture (Day 1–4)
- [x] Design the agent hierarchy (Supervisor / Specialist / Reviewer as LangGraph nodes with input/output schemas)
- [x] Build the task decomposition engine (structured execution plans with dependencies, assigned specialist, expected output format, complexity estimate)
- [x] Implement the tool registry (web search, file read/write, sandboxed code execution, DB query, API calls; logged inputs/outputs/latency/success)
- [x] Build the LangGraph state machine (intake → planning → execution → review → synthesis → delivery, with retry/reject/escalate conditional edges)

### Phase 2: Memory System (Day 4–7)
- [ ] Implement short-term working memory (Redis, scoped to a single task)
- [ ] Build long-term semantic memory (ChromaDB embeddings of tasks, approaches, tools used, facts, preferences)
- [ ] Implement memory retrieval for planning (inject similar past tasks/approaches/preferences into planning prompt)
- [ ] Add memory management (importance scoring, consolidation, expiration, dashboard, delete endpoint)

### Phase 3: Human-in-the-Loop System (Day 7–10)
- [ ] Define escalation triggers (low confidence, repeated specialist failure, sensitive ops, low review score, explicit user request)
- [ ] Build the approval queue (pause execution, package context, push to review queue, notify reviewer, wait for decision)
- [ ] Implement granular approval levels (Notify / Approve action / Approve plan / Take over)
- [ ] Build the review interface (task context, decision point, proposed action + reasoning, relevant memories, action buttons, clarifying-question chat)

### Phase 4: Observability and Debugging (Day 10–12)
- [ ] Implement full execution tracing (OpenTelemetry spans for planning, tool calls, review, memory retrieval, escalation)
- [ ] Build the trace explorer UI (tree/graph view, per-node agent/decision/tools/latency/cost/errors, color-coded status, expandable prompt/response)
- [ ] Add cost and performance tracking (tokens by agent/model, tool calls, wall-clock time, human review time, cost aggregation)
- [ ] Build the replay system (reload past executions, step through decisions, modify inputs, diff against original)

### Phase 5: Integration and End-to-End Testing (Day 12–13)
- [ ] Build a compelling demo scenario (research task: web search → data extraction → analysis → written summary)
- [ ] Containerize the full system (docker-compose: API, Redis, PostgreSQL, ChromaDB, Celery workers, trace UI, review UI + demo script)
- [ ] Write end-to-end tests (decomposition validity, specialist tool use, reviewer catching bad output, memory improving repeat planning, escalation triggers, failure recovery)

### Phase 6: Polish for Portfolio (Day 13–14)
- [ ] Record the demo (< 5 minutes, full task lifecycle)
- [ ] Write the narrative and architecture diagram

## Step-by-Step Build Guide

### Phase 1: Build the Agent Architecture (Day 1–4)
1. **Design the agent hierarchy**: Create three layers. The Supervisor Agent receives complex tasks, creates execution plans, and delegates to specialists. Specialist Agents each own a domain (research, data analysis, writing, code execution) and have access to domain-specific tools. The Reviewer Agent validates outputs from specialists before returning to the supervisor. Model each agent as a LangGraph node with defined input/output schemas.
2. **Build the task decomposition engine**: The supervisor's core capability: take a complex request and break it into an ordered list of subtasks, each assigned to a specialist. Include dependencies (subtask B needs the output of subtask A). Use structured output to enforce a valid execution plan with: subtask description, assigned specialist, required inputs, expected output format, and estimated complexity.
3. **Implement the tool registry**: Build a registry where tools are registered with: a name, description, input/output schemas, the specialist agents that can use them, and rate limits. Start with: web search, file read/write, code execution (sandboxed), database query, and API calls. Each tool invocation is logged with inputs, outputs, latency, and success/failure.
4. **Build the LangGraph state machine**: Wire the agents into a LangGraph graph with: task intake → planning → parallel/sequential specialist execution → review → synthesis → delivery. Include conditional edges: if a specialist fails, retry with a different approach; if the reviewer rejects output, route back to the specialist with feedback; if confidence is low, route to human escalation.

### Phase 2: Build the Memory System (Day 4–7)
1. **Implement short-term working memory**: During a task execution, all agents share a working memory store: the current execution plan, outputs from completed subtasks, intermediate results, and error logs. Store in Redis for fast access. This memory is scoped to a single task and cleared when the task completes.
2. **Build long-term semantic memory**: After task completion, extract and embed key information: what the user asked for, what approach worked, what tools were used, any domain-specific facts discovered, and user preferences observed. Store in ChromaDB. Future tasks can query this memory to inform planning. This is how the system gets smarter over time.
3. **Implement memory retrieval for planning**: When the supervisor creates an execution plan, it first queries long-term memory for: similar past tasks and their execution plans, approaches that worked well (and those that didn't), user-specific preferences and context, and relevant domain facts. Retrieved memories are injected into the planning prompt.
4. **Add memory management**: Implement: memory importance scoring (frequently accessed memories are more important), memory consolidation (merge similar memories into higher-level summaries), memory expiration (stale memories decay over time), and a memory dashboard showing what the system "remembers" about each user. Include a delete endpoint for user data requests.

### Phase 3: Build the Human-in-the-Loop System (Day 7–10)
1. **Define escalation triggers**: The system escalates to a human when: the supervisor's confidence in its plan is below a threshold, a specialist fails twice on the same subtask, the task involves sensitive operations (financial transactions, data deletion, external communications), the reviewer's quality score for a deliverable is below threshold, or the user explicitly requests human review.
2. **Build the approval queue**: When escalation triggers, the system: pauses execution at the current step, packages the full context (original task, plan, completed steps, current step that needs approval, the agent's proposed action), pushes it to a review queue, and notifies the human reviewer. The system waits for approval, rejection, or modification before continuing.
3. **Implement granular approval levels**: Not all escalations need the same review depth. Define levels: Notify (proceed but inform the human), Approve action (human confirms the next step), Approve plan (human reviews the full execution plan before any work begins), and Take over (human directly provides the output, agents stand down). Map escalation triggers to appropriate levels.
4. **Build the review interface**: A UI showing: the task context and execution progress, the specific decision point requiring human input, the agent's proposed action with its reasoning, relevant memories and past similar decisions, and action buttons (approve, modify, reject, take over). Include a chat panel for the human to ask the agent clarifying questions before deciding.

### Phase 4: Build Observability and Debugging (Day 10–12)
1. **Implement full execution tracing**: Every task execution produces a trace tree: the supervisor's planning decisions, each specialist's tool calls and reasoning steps, the reviewer's evaluations, memory retrievals and their influence on decisions, and human escalation events and resolutions. Use OpenTelemetry spans with custom attributes.
2. **Build the trace explorer UI**: A visual representation of the agent workflow as a tree/graph. Each node shows: the agent that acted, what it decided, what tools it called, latency and cost, and any errors. Color-code by status (success/warning/failure/escalated). Clicking a node expands the full context including the LLM prompt and response.
3. **Add cost and performance tracking**: Per task, track: total LLM tokens used (by agent and model), total tool calls, total wall-clock time, human review time (if escalated), and total cost. Aggregate across tasks to show: cost per task type, most expensive agents, tool usage patterns, and escalation rate trends.
4. **Build the replay system**: For debugging, allow replaying any past task execution: load the original task and context, step through each agent decision, modify any input at any step and see how the execution diverges, and compare the replayed execution against the original. This is invaluable for diagnosing failures and testing improvements.

### Phase 5: Integration and End-to-End Testing (Day 12–13)
1. **Build a compelling demo scenario**: Design a complex task that showcases the full system: a research task that requires web search, data extraction, analysis, and a written summary. Show: the supervisor decomposing the task, specialists working in parallel, the reviewer catching an issue and sending it back, memory informing a decision, and a human approving the final deliverable.
2. **Containerize the full system**: Docker-compose with: the orchestration API, Redis (working memory), PostgreSQL (persistent state), ChromaDB (long-term memory), Celery workers (async specialists), the trace explorer UI, and the human review UI. Include a demo script that runs the showcase scenario automatically.
3. **Write end-to-end tests**: Test: task decomposition produces valid plans, specialists correctly use their tools, the reviewer catches deliberately bad outputs, memory retrieval improves planning for repeated similar tasks, human escalation triggers at the right moments, and the system recovers gracefully from agent failures.

### Phase 6: Polish for Portfolio (Day 13–14)
1. **Record the demo**: Show the full task lifecycle: complex request in, supervisor planning, specialists executing with tool calls, reviewer validating, human approving a sensitive step, memory saving lessons learned, and the trace explorer showing every decision. Under 5 minutes.
2. **Write the narrative**: Frame it as: "I built a multi-agent orchestration system where AI agents decompose complex tasks, use tools to execute them, learn from past interactions via persistent memory, and escalate to humans when confidence is low. It's not an AI demo — it's production infrastructure for autonomous AI workflows." Lead with the architecture diagram showing the agent hierarchy and decision flow.
