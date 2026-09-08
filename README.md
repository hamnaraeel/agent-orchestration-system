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
    build.py       # the state machine: intake -> plan -> dispatch -> review -> synthesize -> deliver
  memory/
    models.py      # MemoryRecord, ExtractedMemory
    working.py     # WorkingMemory: Redis hash per task_id, cleared on completion
    long_term.py   # LongTermMemory: ChromaDB collection, importance/consolidation/expiration
    extractor.py   # MemoryExtractorAgent: LLM extraction of approach/facts/preferences
  human_loop/
    models.py          # PendingApproval, ApprovalStatus, ChatMessage
    approval_queue.py  # ApprovalQueue: Redis-backed queue a runner submits to and blocks on
  tracing/
    store.py       # TraceStore: SQLite tasks+spans, cost/agent/tool/escalation aggregates
    otel.py        # OpenTelemetry TracerProvider (console + optional OTLP export)
    pricing.py     # token counts -> $ cost, from settings.model_pricing
    recorder.py    # run_traced_task: drives the graph, persists spans, emits OTel spans
    replay.py      # time-travel: list_checkpoints / replay_from_checkpoint (fork + resume)
  api.py           # FastAPI: memory + approval-queue + trace/cost + replay endpoints
  run.py           # CLI entry point + `run_task`/`run_traced_task`: bridges interrupts to the queue
  tasks.py         # Celery task: reconstructs + runs one specialist in a worker process
ui/
  memory_dashboard.py  # Streamlit page over long-term memory (per-user view + delete)
  review_queue.py      # Streamlit review interface: context, clarifying-question chat, decide
  trace_explorer.py    # Streamlit: span-by-span trace view + cost/performance analytics
  replay.py            # Streamlit: step through a past run's checkpoints, fork, compare
tests/             # unit + integration + end-to-end tests, all using a fake chat model (no API key needed)
demo.py            # showcase scenario: memory -> escalation -> parallel specialists -> redo -> delivery
Dockerfile, docker-compose.yml  # containerized deployment (see Phase 5 below)
```

Run it locally:

```
python -m venv .venv && source .venv/bin/activate   # requires Python 3.11 or 3.12 -- see note below
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
pytest -q
python -m agent_orchestrator.run "Research topic X and write a two-paragraph summary" --user-id alice

# memory + approval-queue + trace/cost + replay API
uvicorn agent_orchestrator.api:app --reload
streamlit run ui/memory_dashboard.py
streamlit run ui/review_queue.py
streamlit run ui/trace_explorer.py
streamlit run ui/replay.py
```

Notes:
- **Python version**: ChromaDB depends on `onnxruntime`, which does not yet ship
  wheels for Python 3.13 on all platforms. Use Python 3.11 or 3.12 for this
  project (`pyproject.toml` pins `requires-python` accordingly).
- **Redis**: short-term working memory *and* the human approval queue need a
  reachable Redis (`REDIS_URL`, default `redis://localhost:6379/0`). Quickest
  local option: `docker run -p 6379:6379 redis`. Long-term memory (ChromaDB)
  persists to a local directory and needs no server.
- **First ChromaDB run**: unless you inject a custom embedding function,
  ChromaDB downloads a small (~80MB) ONNX embedding model on first use and
  caches it — that first call needs network access.
- Run with `--no-memory` to skip Redis and ChromaDB entirely; escalations are
  then prompted for directly on the terminal instead of going through the
  approval queue.
- Run with `--require-approval` to force a human to review the plan (`APPROVE_PLAN`)
  before any specialist work begins, regardless of confidence.

### How escalation works

`human_escalation` calls LangGraph's `interrupt()`, which durably suspends the
graph (via a checkpointer) until resumed with `Command(resume=decision)`. The
graph has no idea a queue exists -- `run_task` in `run.py` is the runner that
bridges an interrupt to a Redis-backed `ApprovalQueue`: it submits the paused
context and blocks on `wait_for_decision`, which is exactly what a human
resolving it through the review API/UI (in a *separate* process) unblocks.

| Trigger | Level | Meaning |
|---|---|---|
| Plan confidence below threshold | `APPROVE_PLAN` | Review the plan before any work begins |
| Plan contains a sensitive subtask (financial, data-destructive, external comms) | `APPROVE_PLAN` | Same, flagged by the supervisor at planning time |
| `--require-approval` passed | `APPROVE_PLAN` | Explicit user request |
| Reviewer flags a deliverable as sensitive/risky | `APPROVE_ACTION` | Confirm before delivering |
| Reviewer never approves after `max_review_cycles` | `TAKE_OVER` | System is stuck, human should just finish it |
| A specialist fails the same subtask past `max_specialist_retries` | `TAKE_OVER` | Same, for one subtask |
| Reviewer approves but the score is still below `review_score_threshold` | `NOTIFY` | Non-blocking -- logged, execution proceeds automatically |

The level is a severity hint for the UI, not a restriction: whichever level
fires, a human resolving a blocking escalation always chooses one of
**approve** / **reject** / **modify** (with feedback) / **take over** (with
their own output). `reject` always stops the task; `take_over` either
supplies that one stuck subtask's output (execution continues) or the whole
task's final output (execution ends), depending on where the escalation fired.

### How observability works

Every graph node appends one entry to `state["trace_events"]` (node, agent,
status, start/end time, token usage, node-specific attributes like the plan's
confidence or a specialist's tool calls) -- a plain accumulating list, not a
side channel, so it survives interrupt/resume for free. `run_traced_task`
(the instrumented counterpart of `run_task`) reads the new entries after each
`invoke` and, for each one: computes cost from `settings.model_pricing`,
writes a row to `TraceStore` (SQLite: `tasks` + `spans` tables, plus one span
per individual tool call), and emits a real OpenTelemetry span with the same
attributes -- so this plugs into Jaeger/Honeycomb/etc. via
`OTEL_EXPORTER_OTLP_ENDPOINT` without code changes, while the trace explorer
and cost dashboards (`ui/trace_explorer.py`) are powered by `TraceStore`
directly, since "cost per task type" and "most expensive agent" aren't things
a span exporter aggregates for you without a metrics backend of its own.

`task_type` (used for cost grouping) is a coarse heuristic -- the sorted set
of specialists a plan uses (e.g. `"research+writing"`) -- not a real
classifier; treat it as a grouping key, not a precise label. Similarly,
`settings.model_pricing`'s bundled rates are round, illustrative placeholders,
not verified current vendor pricing -- override them (see `.env.example`)
before trusting a cost figure.

**Replay** (`ui/replay.py`, `tracing/replay.py`) uses LangGraph's own
checkpoint history rather than a custom re-run mechanism: `get_state_history`
lists every checkpoint a task passed through, `update_state` on a specific one
forks a new branch from that exact point with a modified value, and `invoke`
resumes it -- only the downstream nodes re-run, so the original run's
checkpoints (and its final state) stay intact for side-by-side comparison.
This only works against a *persistent* checkpointer, which is why the CLI
(`run.build_app`) points the graph at a real one -- SQLite (`settings.checkpoint_db_path`)
locally, or Postgres (`settings.postgres_url`) under docker-compose -- instead
of the fast in-memory default `build_graph()` falls back to for tests.

## Containerized deployment

```
cp .env.example .env   # fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
docker compose up -d --build
docker compose --profile demo run --rm demo   # runs the showcase scenario
```

This brings up seven services: `redis` (working memory + approval queue),
`postgres` (checkpoint persistence, so a paused/resumed task and replay
history survive any single container restarting), `chroma` (long-term
memory, as a real server this time instead of a local directory), `api`
(the FastAPI app from `api.py`), `worker` (a Celery worker executing
specialist tool-calling loops), and four Streamlit UIs -- `trace-ui`
(:8501), `review-ui` (:8502), `memory-ui` (:8503), `replay-ui` (:8504).
`api` is on :8080.

All seven app-facing services (`api`, `worker`, the four UIs, `demo`) are
built from the same image (one `Dockerfile`, `command:` overridden per
service) and share an `app_data` volume so `TRACE_DB_PATH`, tool-sandbox
files, and anything else written to `/data` are visible across all of them
-- e.g. a task the CLI or `demo` runs shows up in `trace-ui` immediately.

`USE_CELERY_FOR_SPECIALISTS=true` in the compose environment is what routes
each specialist's tool-calling loop through the `worker` service instead of
running in-process -- the graph itself doesn't change; `graph/build.py`'s
`run_specialist` node just calls `run_specialist_task.delay(...).get(...)`
instead of the specialist directly when this is set (see `tasks.py`). Turn
it off (or just run `python -m agent_orchestrator.run` locally, where it
defaults to `false`) to execute specialists in-process instead.

Running only the CLI/tests doesn't need any of this -- `docker compose` is
for the "production-shaped" full stack; `pip install -e ".[dev]"` and a local
`.env` are all `pytest` or `python -m agent_orchestrator.run` need.

## Build Progress

Tracking checklist for the build guide below. Check items off as they're completed.

### Phase 1: Agent Architecture (Day 1–4)
- [x] Design the agent hierarchy (Supervisor / Specialist / Reviewer as LangGraph nodes with input/output schemas)
- [x] Build the task decomposition engine (structured execution plans with dependencies, assigned specialist, expected output format, complexity estimate)
- [x] Implement the tool registry (web search, file read/write, sandboxed code execution, DB query, API calls; logged inputs/outputs/latency/success)
- [x] Build the LangGraph state machine (intake → planning → execution → review → synthesis → delivery, with retry/reject/escalate conditional edges)

### Phase 2: Memory System (Day 4–7)
- [x] Implement short-term working memory (Redis, scoped to a single task)
- [x] Build long-term semantic memory (ChromaDB embeddings of tasks, approaches, tools used, facts, preferences)
- [x] Implement memory retrieval for planning (inject similar past tasks/approaches/preferences into planning prompt)
- [x] Add memory management (importance scoring, consolidation, expiration, dashboard, delete endpoint)

### Phase 3: Human-in-the-Loop System (Day 7–10)
- [x] Define escalation triggers (low confidence, repeated specialist failure, sensitive ops, low review score, explicit user request)
- [x] Build the approval queue (pause execution, package context, push to review queue, notify reviewer, wait for decision)
- [x] Implement granular approval levels (Notify / Approve action / Approve plan / Take over)
- [x] Build the review interface (task context, decision point, proposed action + reasoning, relevant memories, action buttons, clarifying-question chat)

### Phase 4: Observability and Debugging (Day 10–12)
- [x] Implement full execution tracing (OpenTelemetry spans for planning, tool calls, review, memory retrieval, escalation)
- [x] Build the trace explorer UI (tree/graph view, per-node agent/decision/tools/latency/cost/errors, color-coded status, expandable prompt/response)
- [x] Add cost and performance tracking (tokens by agent/model, tool calls, wall-clock time, human review time, cost aggregation)
- [x] Build the replay system (reload past executions, step through decisions, modify inputs, diff against original)

### Phase 5: Integration and End-to-End Testing (Day 12–13)
- [x] Build a compelling demo scenario (research task: web search → data extraction → analysis → written summary)
- [x] Containerize the full system (docker-compose: API, Redis, PostgreSQL, ChromaDB, Celery workers, trace UI, review UI + demo script)
- [x] Write end-to-end tests (decomposition validity, specialist tool use, reviewer catching bad output, memory improving repeat planning, escalation triggers, failure recovery)

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
