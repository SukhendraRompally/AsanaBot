"""
app.py — FastAPI Server
Moveworks Agent Studio: Asana Reference Implementation

Responsibilities:
  - POST /chat    : Accept a user message, stream NDJSON events from the ReAct engine
  - GET  /health  : Service status and configuration check
  - GET  /tools   : Serializable TOOL_REGISTRY for the Replit frontend UI
  - Session store : Human-in-the-Loop gate — freezes pending destructive actions
                    until the user confirms or cancels

Principal-level notes:
  - _session_store is the ONLY global mutable state in this module.
    It is safe in a single-process event loop. For horizontal scaling,
    replace with Redis + aioredis.
  - Destructive actions are NEVER auto-executed. The engine yields a
    `confirmation_required` event; this handler freezes the pending action
    into a session and injects the session_id into the event before streaming
    it to the client.
  - The streaming generator wraps the entire engine in try/except to guarantee
    the NDJSON stream always closes cleanly — never mid-stream silence.
  - CORS is wide-open (allow_origins=["*"]) so Replit can reach this VM.
    Restrict to the Replit deployment URL in production.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import asana_tools
from asana_tools import TOOL_REGISTRY
from engine import EngineConfig, config_from_env, run_react_loop

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required environment variables (validated at startup)
# ---------------------------------------------------------------------------
REQUIRED_ENV_VARS: list[str] = [
    "ASANA_PAT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_VERSION",
]

# ---------------------------------------------------------------------------
# Session State — Human-in-the-Loop Gate + Conversation Memory
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS: int = 600        # Pending confirmation expires after 10 min
CONVERSATION_TTL_SECONDS: int = 7200  # Conversation memory expires after 2 hours


@dataclasses.dataclass
class SessionState:
    """
    Frozen state of a pending destructive action awaiting user confirmation.
    The engine pauses, app.py serializes this, and the client sends back
    `confirmation=true` + `session_id` to resume.
    """
    session_id: str
    pending_tool: str              # e.g. "create_task"
    pending_args: dict[str, Any]   # Exact args the LLM chose — never re-computed
    original_message: str          # The user's original request (for logging)
    created_at: datetime           # UTC timestamp for TTL expiry


@dataclasses.dataclass
class ConversationState:
    """
    Persistent conversation history for a session.
    Stores only clean user/assistant summary pairs — NOT the raw ReAct traces
    (tool calls, observations) from within a single turn. This keeps the
    context window lean and prevents prior JSON tool calls from confusing the LLM.

    Format of each entry in `history`:
        {"role": "user",      "content": "<what the user said>"}
        {"role": "assistant", "content": "<the final answer given>"}
    """
    session_id: str
    history: list[dict]            # Mutable list — engine appends to it in-place
    last_active: datetime          # Updated on every turn for TTL tracking


# Module-level stores — the only global mutable state in this module.
# For horizontal scaling, replace both with Redis + aioredis.
_session_store: dict[str, SessionState] = {}          # pending confirmations
_conversation_store: dict[str, ConversationState] = {}  # per-session memory


def _prune_expired_sessions() -> None:
    """
    Remove stale entries from both stores.
    Called at the start of every POST /chat request to prevent memory leaks.
    """
    now = datetime.now(timezone.utc)

    expired_sessions = [
        sid for sid, state in _session_store.items()
        if (now - state.created_at).total_seconds() > SESSION_TTL_SECONDS
    ]
    for sid in expired_sessions:
        logger.info("Pruning expired confirmation session %s (tool: %s)", sid, _session_store[sid].pending_tool)
        del _session_store[sid]

    expired_convos = [
        sid for sid, state in _conversation_store.items()
        if (now - state.last_active).total_seconds() > CONVERSATION_TTL_SECONDS
    ]
    for sid in expired_convos:
        logger.info("Pruning expired conversation %s (%d turns)", sid, len(_conversation_store[sid].history) // 2)
        del _conversation_store[sid]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    Fresh request:   {"message": "Show me all my projects"}
    Confirmation:    {"message": "", "confirmation": true, "session_id": "uuid4"}
    Cancellation:    {"message": "", "confirmation": false, "session_id": "uuid4"}
    """
    message: str = ""
    confirmation: bool = False
    session_id: str | None = None


class ToolInfo(BaseModel):
    """
    Serializable representation of a tool — the `function` callable is excluded.
    Returned by GET /tools for the Replit frontend to render the tool palette.
    """
    name: str
    description: str
    parameters: dict
    is_destructive: bool


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    asana_configured: bool
    azure_configured: bool
    active_sessions: int


# ---------------------------------------------------------------------------
# App Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: validate environment, log tool count.
    Shutdown: close the httpx AsyncClient to release connection pool.
    """
    # Validate required env vars before accepting traffic
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"AsanaBot cannot start — missing required environment variables: {missing}. "
            "Check your .env file."
        )

    logger.info(
        "AsanaBot starting on port 8001. %d tools registered. "
        "Asana PAT: %s... Azure deployment: %s",
        len(TOOL_REGISTRY),
        (os.getenv("ASANA_PAT") or "")[:8],
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    )

    yield  # Application runs here

    # Teardown: release the httpx connection pool
    client = asana_tools.get_asana_client()
    await client.aclose()
    logger.info("AsanaBot shutdown — httpx client closed.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Asana Agent — Moveworks Reference Implementation",
    description=(
        "A production-grade agentic backend for Asana. "
        "Combines a ReAct reasoning loop, an Asana v1.0 connector, "
        "and a Human-in-the-Loop trust layer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Replit needs open CORS; restrict in production
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    """
    Returns service status and whether required credentials are configured.
    Safe to poll — performs no Asana API calls.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        asana_configured=bool(os.getenv("ASANA_PAT")),
        azure_configured=bool(
            os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
        ),
        active_sessions=len(_session_store),
    )


@app.get("/tools", response_model=list[ToolInfo], tags=["Meta"])
async def list_tools() -> list[ToolInfo]:
    """
    Returns the full list of registered tools.
    The `function` callable is excluded — only metadata is serialized.
    Used by the Replit frontend to render the tool palette / sidebar.
    """
    return [
        ToolInfo(
            name=spec["name"],
            description=spec["description"],
            parameters=spec["parameters"],
            is_destructive=spec["is_destructive"],
        )
        for spec in TOOL_REGISTRY.values()
    ]


@app.post("/chat", tags=["Agent"])
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Main conversational endpoint. Returns an NDJSON stream where each line
    is a JSON event from the ReAct engine.

    Event types:
      thought              — The agent's internal reasoning step
      action               — A tool the agent intends to call
      observation          — The result returned by a tool call
      confirmation_required — A destructive action is pending user approval
                              (includes session_id for the client to send back)
      result               — Final outcome with status SUCCESS | ERROR | CANCELLED

    Two flows:
    ──────────
    1. Fresh:    POST {"message": "show my tasks"}
    2. Confirm:  POST {"confirmation": true,  "session_id": "<id>"}
    3. Cancel:   POST {"confirmation": false, "session_id": "<id>"}
    """
    return StreamingResponse(
        _build_stream(request),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )


# ---------------------------------------------------------------------------
# Internal Streaming Generator
# ---------------------------------------------------------------------------

async def _build_stream(request: ChatRequest) -> AsyncGenerator[bytes, None]:
    """
    Core generator that drives the engine and serializes events to NDJSON bytes.

    Conversation memory:
      Every result event now includes `session_id`. The client must send this
      back in all subsequent messages so the backend can load the conversation
      history and the engine can maintain context across turns.

    Request modes:
      Fresh turn:   {message: "...", session_id: "<id from prior result>"}
      First turn:   {message: "..."} — server assigns a new session_id
      Confirm:      {confirmation: true,  session_id: "<id>"}
      Cancel:       {confirmation: false, session_id: "<id>", message: ""}

    Guarantees:
      - Always yields at least one event (even on error)
      - Never raises — all exceptions produce an error result event
      - session_id is injected into every result and confirmation_required event
    """
    def _emit(event: dict) -> bytes:
        """Serialize a single event to a UTF-8 NDJSON line."""
        return (json.dumps(event, default=str) + "\n").encode("utf-8")

    def _inject_session(event: dict, sid: str) -> dict:
        """Add session_id to a result or confirmation_required event's content."""
        if event["type"] in ("result", "confirmation_required"):
            event["content"]["session_id"] = sid
        return event

    try:
        # ── Prune stale stores on every request ──────────────────────────
        _prune_expired_sessions()

        engine_config: EngineConfig = config_from_env()

        # ── Mode: Explicit cancel (empty message + session_id + confirmation=False)
        if request.session_id and not request.confirmation and not request.message.strip():
            session = _session_store.pop(request.session_id, None)
            if session:
                logger.info("Session %s cancelled by user.", request.session_id)
                yield _emit(_inject_session({
                    "type": "result",
                    "content": {
                        "status": "CANCELLED",
                        "message": f"Action '{session.pending_tool}' was cancelled.",
                        "data": None,
                    },
                }, request.session_id))
            else:
                yield _emit(_inject_session({
                    "type": "result",
                    "content": {
                        "status": "ERROR",
                        "message": "No pending action found for this session.",
                        "data": None,
                    },
                }, request.session_id))
            return

        # ── Mode: Confirmation ────────────────────────────────────────────
        if request.confirmation:
            if not request.session_id:
                yield _emit({
                    "type": "result",
                    "content": {
                        "status": "ERROR",
                        "message": "session_id is required when confirmation=true.",
                        "data": None,
                        "session_id": None,
                    },
                })
                return

            session = _session_store.pop(request.session_id, None)
            if not session:
                yield _emit(_inject_session({
                    "type": "result",
                    "content": {
                        "status": "ERROR",
                        "message": (
                            "Confirmation session not found or expired. "
                            "Sessions expire after 10 minutes — please start over."
                        ),
                        "data": None,
                    },
                }, request.session_id))
                return

            logger.info(
                "Executing confirmed action '%s' for session %s",
                session.pending_tool, request.session_id
            )

            # Load conversation history so the confirmed action is recorded in memory
            convo = _conversation_store.get(request.session_id)
            history = convo.history if convo else []

            pending_action = {"tool": session.pending_tool, "args": session.pending_args}
            async for event in run_react_loop(
                user_message=session.original_message,
                config=engine_config,
                conversation_history=history,
                pending_action=pending_action,
                confirmed=True,
            ):
                yield _emit(_inject_session(event, request.session_id))

            # Update last_active so the conversation doesn't expire prematurely
            if request.session_id in _conversation_store:
                _conversation_store[request.session_id].last_active = datetime.now(timezone.utc)
            return

        # ── Mode: Regular message (fresh turn or follow-up) ───────────────
        if not request.message.strip():
            yield _emit({
                "type": "result",
                "content": {
                    "status": "ERROR",
                    "message": "message cannot be empty.",
                    "data": None,
                    "session_id": request.session_id,
                },
            })
            return

        # Reuse the client's session_id if provided (continuing a conversation),
        # or mint a new one for a brand-new conversation.
        current_session_id = request.session_id or str(uuid.uuid4())

        # Load or create the conversation state for this session
        if current_session_id not in _conversation_store:
            _conversation_store[current_session_id] = ConversationState(
                session_id=current_session_id,
                history=[],
                last_active=datetime.now(timezone.utc),
            )
        convo_state = _conversation_store[current_session_id]
        # history is a mutable list — engine appends to it in-place after each turn
        history = convo_state.history

        turn_count = len(history) // 2  # each turn = 1 user + 1 assistant entry
        logger.info(
            "Chat turn %d (session %s): %s",
            turn_count + 1, current_session_id, request.message[:100]
        )

        async for event in run_react_loop(
            user_message=request.message,
            config=engine_config,
            conversation_history=history,
        ):
            # ── Intercept confirmation_required: freeze into session store ─
            if event["type"] == "confirmation_required":
                tool_name = event["content"]["tool"]
                tool_args = event["content"]["args"]

                _session_store[current_session_id] = SessionState(
                    session_id=current_session_id,
                    pending_tool=tool_name,
                    pending_args=tool_args,
                    original_message=request.message,
                    created_at=datetime.now(timezone.utc),
                )
                logger.info(
                    "Stored pending action '%s' in session %s",
                    tool_name, current_session_id
                )

            # Inject session_id into result and confirmation events
            yield _emit(_inject_session(event, current_session_id))

        # Update last_active after the turn completes
        convo_state.last_active = datetime.now(timezone.utc)

    except Exception as exc:
        # Safety net: guarantee the stream always closes with a readable error
        logger.exception("Unhandled exception in _build_stream")
        yield (
            json.dumps({
                "type": "result",
                "content": {
                    "status": "ERROR",
                    "message": f"Internal server error: {type(exc).__name__}: {exc}",
                    "data": None,
                },
            }) + "\n"
        ).encode("utf-8")
