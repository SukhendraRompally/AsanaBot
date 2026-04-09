# AsanaBot

An agentic AI assistant for Asana. Talk to your workspace in plain English — AsanaBot reasons step-by-step, calls the right Asana APIs, and asks for confirmation before touching anything it can't undo.

---

## What It Does

AsanaBot is a conversational backend that lets you manage Asana through a chat interface. You describe what you want in natural language; the bot figures out the right sequence of API calls to make it happen.

**Read operations** (instant, no confirmation needed):
- List your workspaces, projects, and users
- Search tasks by keyword
- Get full task or project details

**Write operations** (paused for your approval):
- Create tasks (with assignee, project, due date)
- Update task fields
- Mark tasks complete
- Delete tasks

Multi-turn conversations are supported — you can refer back to things from earlier in the same session.

---

## Architecture

```
Client (HTTP)
    │
    ▼
app.py  ──  FastAPI server, NDJSON streaming, session state
    │
    ▼
engine.py  ──  ReAct reasoning loop (Thought → Action → Observation → ...)
    │
    ▼
asana_tools.py  ──  Asana REST API v1.0 connector, 11 tools
    │
    ▼
Asana API
```

### Three files, three responsibilities

| File | Role |
|---|---|
| `app.py` | HTTP layer — routes, session store, NDJSON streaming |
| `engine.py` | AI reasoning — ReAct loop, LLM calls, prompt, output parsing |
| `asana_tools.py` | Asana connector — auth, retry, pagination, 11 tool functions |

---

## Key Design Decisions

### ReAct reasoning loop
The agent uses the ReAct pattern (Reason + Act): the LLM outputs a thought, then a tool call, receives the observation, thinks again, and repeats until it has a complete answer. This keeps the reasoning transparent — every step is streamed to the client as it happens.

No function-calling API is used. The LLM is prompted to output raw JSON objects, which the engine parses. This makes the reasoning trace fully visible and keeps the loop compatible with any model that can follow JSON instructions.

### Human-in-the-Loop for destructive actions
Write operations (create, update, complete, delete) never execute automatically. When the engine decides to call one, it pauses and emits a `confirmation_required` event. The action is frozen in a server-side session (10-minute TTL). The client shows the user a plain-English summary of what will happen and waits for explicit approval before anything changes in Asana.

This is enforced in the engine, not just the UI — the code path that executes a confirmed action skips the LLM entirely and runs the frozen args directly. This prevents the LLM from producing different args on a second call.

### Streaming NDJSON
Responses are streamed as newline-delimited JSON. Each reasoning step (thought, action, observation) is sent to the client as it's produced rather than waiting for the full answer. This makes the agent feel responsive even on multi-step queries that involve several Asana API calls.

Event types: `thought` | `action` | `observation` | `confirmation_required` | `result`

### Conversation memory
Each session maintains a conversation history of clean user/assistant summary pairs — not the raw ReAct traces. Raw traces (tool call JSON, observations) are not persisted between turns to keep the context window lean and avoid confusing the model with prior API payloads.

Sessions expire after 2 hours of inactivity. Pending confirmations expire after 10 minutes.

### GID disambiguation
Asana identifies everything by opaque numeric GIDs. The system prompt enforces strict resolution rules: the LLM must call a lookup tool to resolve any name to a GID before using it. It is never allowed to guess or reuse a GID from memory without fresh confirmation. If a name matches multiple entities, the agent lists them all and asks the user to pick.

### Retry and error handling
All Asana API calls use exponential backoff (tenacity) on 429 and 5xx errors, up to 5 attempts. Non-retryable 4xx errors (bad GID, auth failure) raise immediately. Parse errors from the LLM trigger an inline correction — the engine injects an error message back into the conversation and retries, up to 3 times, before surfacing an error to the user.

---

## Tradeoffs

**JSON prompting vs. function calling**
Prompting the LLM to output raw JSON makes the reasoning steps fully transparent and works with any model. The downside is that output parsing can fail if the model goes off-format. The self-healing retry (up to 3 parse errors) handles most cases in practice, but native function calling would be more reliable structurally.

**In-memory session state**
Sessions and conversation history are stored in module-level dicts. This is fine for a single-process deployment but means state is lost on restart and won't work across multiple instances. The fix is Redis — the code is structured so only `_session_store` and `_conversation_store` need to be swapped out.

**Conversation history is summarised, not full**
Only the final user message and assistant answer from each turn are stored in history, not the full ReAct trace. This keeps the prompt small but means the LLM can't reference intermediate observations from a prior turn. In practice this is rarely needed — if a task GID came up two turns ago, the agent will look it up again.

**No streaming backpressure**
NDJSON events are yielded as fast as the engine produces them. There's no flow control between the engine and the client. For very long ReAct chains this is fine, but a slow client could accumulate a large buffer.

**Task cap on project queries**
`get_tasks_for_project` paginates up to 500 tasks (5 pages × 100). Projects larger than that are silently truncated. The agent logs a warning but the user doesn't see it.

---

## Setup

**Requirements:** Python 3.11+, an Asana Personal Access Token, and Azure OpenAI credentials.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Fill in ASANA_PAT, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT,
# AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_VERSION

# 3. Start the server
./start.sh
# API:    http://localhost:8001
# Docs:   http://localhost:8001/docs
# Health: http://localhost:8001/health
```

---

## API

### `POST /chat`
Main conversational endpoint. Returns an NDJSON stream.

**Fresh message:**
```json
{"message": "Show me all tasks in the Q2 launch project"}
```

**Follow-up (continuing a session):**
```json
{"message": "Mark the first one complete", "session_id": "<id from prior result>"}
```

**Approve a destructive action:**
```json
{"confirmation": true, "session_id": "<id from confirmation_required event>"}
```

**Cancel:**
```json
{"confirmation": false, "session_id": "<id>", "message": ""}
```

### `GET /health`
Returns service status and whether Asana and Azure credentials are configured.

### `GET /tools`
Returns the list of registered tools — name, description, parameters, and whether each is destructive.

---

## Tools

| Tool | Type | Description |
|---|---|---|
| `get_workspaces` | read | List all accessible workspaces |
| `list_projects` | read | List active projects in a workspace |
| `get_project` | read | Full project details by GID |
| `get_tasks_for_project` | read | All tasks in a project (up to 500) |
| `search_tasks` | read | Full-text task search within a workspace |
| `get_task` | read | Full task details by GID |
| `get_users` | read | List users in a workspace |
| `create_task` | **destructive** | Create a new task |
| `update_task` | **destructive** | Update task fields |
| `complete_task` | **destructive** | Mark a task complete |
| `delete_task` | **destructive** | Permanently delete a task |
