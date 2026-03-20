"""
asana_tools.py — Asana v1.0 Connector Layer
Moveworks Agent Studio: Asana Reference Implementation

Responsibilities:
  - Authenticated httpx.AsyncClient (singleton) for all Asana API calls
  - Exponential-backoff retry on 429 / 5xx / network errors via tenacity
  - 11 tool functions (7 read-only, 4 destructive) covering the core Asana surface
  - GID resolution helpers (name → GID) for workspaces, projects, and users
  - TOOL_REGISTRY: the single source of truth consumed by engine.py and app.py

Notes:
  - All functions are `async def` — never block the event loop
  - AsanaAPIError is the only exception that escapes this module
    (tenacity re-raises after max retries, but only for retryable errors;
     non-retryable 4xx errors raise immediately)
  - The `function` key in TOOL_REGISTRY is the live callable; it is never
    serialized to JSON — only name/description/parameters/is_destructive are
    exposed over the wire
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypedDict

import httpx
from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASANA_BASE_URL: str = "https://app.asana.com/api/1.0"

# Default opt_fields for task list responses — avoids fetching the full graph.
# Requesting only what the agent needs keeps payloads small and responses fast.
TASK_LIST_OPT_FIELDS: str = "gid,name,notes,completed,due_on,assignee.name,projects.name"
TASK_DETAIL_OPT_FIELDS: str = (
    "gid,name,notes,completed,due_on,assignee.name,assignee.email,"
    "projects.name,workspace.name,tags.name,followers.name,created_at,modified_at"
)

# Maximum tasks to fetch when paginating (5 pages × 100 tasks)
MAX_TASK_PAGES: int = 5
PAGE_SIZE: int = 100


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class AsanaAPIError(Exception):
    """
    Raised when an Asana API call fails with a non-retryable error.
    Carries the HTTP status code so the engine can surface it intelligently.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    """
    Determines whether an exception warrants a retry attempt.
    - HTTP 429 (rate limit) and 5xx server errors → retry
    - httpx network/transport errors → retry
    - 4xx client errors (bad GID, bad auth) → do NOT retry
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


asana_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ---------------------------------------------------------------------------
# HTTP Client (singleton)
# ---------------------------------------------------------------------------
_client: httpx.AsyncClient | None = None


def get_asana_client() -> httpx.AsyncClient:
    """
    Returns the module-level singleton AsyncClient configured for Asana.
    Built lazily so load_dotenv() runs before the PAT is read.
    Reuses the connection pool across all tool calls — avoids per-request
    TLS handshake overhead.
    """
    global _client
    if _client is None or _client.is_closed:
        pat = os.getenv("ASANA_PAT")
        if not pat:
            raise RuntimeError(
                "ASANA_PAT is not set. Add it to your .env file."
            )
        _client = httpx.AsyncClient(
            base_url=ASANA_BASE_URL,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=20, max_keepalive_connections=10
            ),
        )
    return _client


# ---------------------------------------------------------------------------
# Internal HTTP helpers
# ---------------------------------------------------------------------------
async def _get(path: str, params: dict | None = None) -> Any:
    """
    Performs a GET request against the Asana API.
    Returns the `data` field from the response body.
    Raises AsanaAPIError on non-retryable 4xx errors.
    Raises httpx.HTTPStatusError on 5xx (to be caught by @asana_retry).
    """
    client = get_asana_client()
    response = await client.get(path, params=params)
    if response.status_code in (401, 403, 404):
        body = response.json()
        errors = body.get("errors", [{}])
        msg = errors[0].get("message", response.text) if errors else response.text
        raise AsanaAPIError(
            f"Asana API {response.status_code}: {msg}", response.status_code
        )
    response.raise_for_status()  # 5xx → httpx.HTTPStatusError → retried
    return response.json().get("data")


async def _post(path: str, payload: dict) -> Any:
    """Performs a POST request. Returns the `data` field."""
    client = get_asana_client()
    response = await client.post(path, json={"data": payload})
    if response.status_code in (400, 401, 403, 404, 422):
        body = response.json()
        errors = body.get("errors", [{}])
        msg = errors[0].get("message", response.text) if errors else response.text
        raise AsanaAPIError(
            f"Asana API {response.status_code}: {msg}", response.status_code
        )
    response.raise_for_status()
    return response.json().get("data")


async def _put(path: str, payload: dict) -> Any:
    """Performs a PUT request. Returns the `data` field."""
    client = get_asana_client()
    response = await client.put(path, json={"data": payload})
    if response.status_code in (400, 401, 403, 404, 422):
        body = response.json()
        errors = body.get("errors", [{}])
        msg = errors[0].get("message", response.text) if errors else response.text
        raise AsanaAPIError(
            f"Asana API {response.status_code}: {msg}", response.status_code
        )
    response.raise_for_status()
    return response.json().get("data")


async def _delete(path: str) -> None:
    """Performs a DELETE request. Returns nothing on success."""
    client = get_asana_client()
    response = await client.delete(path)
    if response.status_code in (401, 403, 404):
        body = response.json()
        errors = body.get("errors", [{}])
        msg = errors[0].get("message", response.text) if errors else response.text
        raise AsanaAPIError(
            f"Asana API {response.status_code}: {msg}", response.status_code
        )
    response.raise_for_status()


# ---------------------------------------------------------------------------
# READ-ONLY Tool Functions
# ---------------------------------------------------------------------------

@asana_retry
async def get_workspaces() -> list[dict]:
    """
    List all Asana workspaces accessible with the configured PAT.
    Returns a list of {gid, name, resource_type} objects.
    """
    data = await _get("/workspaces")
    return data or []


@asana_retry
async def search_tasks(workspace_gid: str, query: str) -> list[dict]:
    """
    Full-text search for tasks within a workspace.
    Returns up to 100 matching tasks with key fields.
    """
    data = await _get(
        f"/workspaces/{workspace_gid}/tasks/search",
        params={"text": query, "opt_fields": TASK_LIST_OPT_FIELDS, "limit": 50},
    )
    return data or []


@asana_retry
async def get_task(task_gid: str) -> dict:
    """
    Retrieve full task details by GID.
    Returns a rich task object including assignee, projects, tags, and dates.
    """
    data = await _get(f"/tasks/{task_gid}", params={"opt_fields": TASK_DETAIL_OPT_FIELDS})
    return data or {}


@asana_retry
async def list_projects(workspace_gid: str) -> list[dict]:
    """
    List all active projects in a workspace.
    Returns a list of {gid, name, resource_type} objects.
    """
    data = await _get(
        "/projects",
        params={
            "workspace": workspace_gid,
            "opt_fields": "gid,name,archived,owner.name",
            "archived": "false",
        },
    )
    return data or []


@asana_retry
async def get_project(project_gid: str) -> dict:
    """
    Retrieve full project details by GID.
    """
    data = await _get(
        f"/projects/{project_gid}",
        params={"opt_fields": "gid,name,notes,archived,owner.name,members.name,created_at,modified_at"},
    )
    return data or {}


@asana_retry
async def get_tasks_for_project(project_gid: str) -> list[dict]:
    """
    List all tasks within a project.
    Handles Asana pagination via the `offset` cursor.
    Caps at MAX_TASK_PAGES (500 tasks) to prevent memory explosion on large projects.
    """
    all_tasks: list[dict] = []
    params: dict[str, Any] = {
        "opt_fields": TASK_LIST_OPT_FIELDS,
        "limit": PAGE_SIZE,
    }
    pages_fetched = 0

    while pages_fetched < MAX_TASK_PAGES:
        client = get_asana_client()
        response = await client.get(f"/projects/{project_gid}/tasks", params=params)
        if response.status_code in (401, 403, 404):
            body = response.json()
            errors = body.get("errors", [{}])
            msg = errors[0].get("message", response.text) if errors else response.text
            raise AsanaAPIError(
                f"Asana API {response.status_code}: {msg}", response.status_code
            )
        response.raise_for_status()
        body = response.json()
        all_tasks.extend(body.get("data") or [])
        pages_fetched += 1

        next_page = body.get("next_page")
        if not next_page or not next_page.get("offset"):
            break  # No more pages
        params["offset"] = next_page["offset"]

    if pages_fetched == MAX_TASK_PAGES:
        logger.warning(
            "get_tasks_for_project: reached MAX_TASK_PAGES (%d) for project %s. "
            "Results may be truncated.",
            MAX_TASK_PAGES,
            project_gid,
        )

    return all_tasks


@asana_retry
async def get_users(workspace_gid: str) -> list[dict]:
    """
    List all users in a workspace.
    Used primarily for assignee GID resolution.
    Returns a list of {gid, name, email} objects.
    """
    data = await _get(
        f"/workspaces/{workspace_gid}/users",
        params={"opt_fields": "gid,name,email"},
    )
    return data or []


# ---------------------------------------------------------------------------
# DESTRUCTIVE Tool Functions
# ---------------------------------------------------------------------------

@asana_retry
async def create_task(
    workspace_gid: str,
    name: str,
    notes: str = "",
    assignee_gid: str | None = None,
    project_gid: str | None = None,
    due_on: str | None = None,
) -> dict:
    """
    Create a new task in Asana.
    DESTRUCTIVE: modifies Asana data.

    Args:
        workspace_gid: GID of the target workspace.
        name: Task title.
        notes: Task body / description.
        assignee_gid: GID of the user to assign to (optional).
        project_gid: GID of the project to add the task to (optional).
        due_on: Due date in YYYY-MM-DD format (optional).

    Returns the created task object.
    """
    payload: dict[str, Any] = {
        "workspace": workspace_gid,
        "name": name,
        "notes": notes,
    }
    if assignee_gid:
        payload["assignee"] = assignee_gid
    if project_gid:
        payload["projects"] = [project_gid]
    if due_on:
        payload["due_on"] = due_on

    data = await _post("/tasks", payload)
    return data or {}


@asana_retry
async def update_task(task_gid: str, **fields: Any) -> dict:
    """
    Update one or more fields on an existing task.
    DESTRUCTIVE: modifies Asana data.

    Common updatable fields: name, notes, due_on, assignee (GID), completed.
    Unknown fields are passed through to the API (Asana silently ignores invalid ones).

    Returns the updated task object.
    """
    data = await _put(f"/tasks/{task_gid}", fields)
    return data or {}


@asana_retry
async def complete_task(task_gid: str) -> dict:
    """
    Mark a task as complete.
    DESTRUCTIVE: modifies Asana data.
    Returns the updated task object.
    """
    data = await _put(f"/tasks/{task_gid}", {"completed": True})
    return data or {}


@asana_retry
async def delete_task(task_gid: str) -> dict:
    """
    Permanently delete a task.
    DESTRUCTIVE: this action cannot be undone.
    Returns an empty dict on success (Asana returns an empty data object).
    """
    await _delete(f"/tasks/{task_gid}")
    return {"deleted": True, "task_gid": task_gid}


# ---------------------------------------------------------------------------
# GID Resolution Helpers
# ---------------------------------------------------------------------------

async def resolve_workspace_gid(name: str) -> str | None:
    """
    Resolve a workspace name to its GID via case-insensitive substring match.
    Returns the first match's GID, or None if no match found.
    Logs a warning if multiple workspaces match (ambiguous).
    """
    workspaces = await get_workspaces()
    name_lower = name.lower()
    matches = [w for w in workspaces if name_lower in w.get("name", "").lower()]
    if not matches:
        logger.warning("resolve_workspace_gid: no workspace matched '%s'", name)
        return None
    if len(matches) > 1:
        logger.warning(
            "resolve_workspace_gid: '%s' matched %d workspaces (%s). Using first.",
            name, len(matches), [m["name"] for m in matches]
        )
    return matches[0]["gid"]


async def resolve_project_gid(workspace_gid: str, name: str) -> str | None:
    """
    Resolve a project name to its GID within a workspace.
    Case-insensitive substring match. Returns first match GID or None.
    """
    projects = await list_projects(workspace_gid)
    name_lower = name.lower()
    matches = [p for p in projects if name_lower in p.get("name", "").lower()]
    if not matches:
        logger.warning(
            "resolve_project_gid: no project matched '%s' in workspace %s",
            name, workspace_gid
        )
        return None
    if len(matches) > 1:
        logger.warning(
            "resolve_project_gid: '%s' matched %d projects (%s). Using first.",
            name, len(matches), [m["name"] for m in matches]
        )
    return matches[0]["gid"]


async def resolve_user_gid(workspace_gid: str, name_or_email: str) -> str | None:
    """
    Resolve a user's name or email to their GID within a workspace.
    Checks both 'name' and 'email' fields (case-insensitive substring).
    Returns first match GID or None.
    """
    users = await get_users(workspace_gid)
    query_lower = name_or_email.lower()
    matches = [
        u for u in users
        if query_lower in u.get("name", "").lower()
        or query_lower in u.get("email", "").lower()
    ]
    if not matches:
        logger.warning(
            "resolve_user_gid: no user matched '%s' in workspace %s",
            name_or_email, workspace_gid
        )
        return None
    if len(matches) > 1:
        logger.warning(
            "resolve_user_gid: '%s' matched %d users (%s). Using first.",
            name_or_email, len(matches), [m["name"] for m in matches]
        )
    return matches[0]["gid"]


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolSpec(TypedDict):
    name: str
    description: str
    function: Callable  # Live async callable — never serialized to JSON
    parameters: dict    # JSON Schema object consumed by the LLM system prompt
    is_destructive: bool


TOOL_REGISTRY: dict[str, ToolSpec] = {
    # ------------------------------------------------------------------
    # READ-ONLY tools
    # ------------------------------------------------------------------
    "get_workspaces": {
        "name": "get_workspaces",
        "description": (
            "List all Asana workspaces accessible with the configured PAT. "
            "Always call this first if you need a workspace_gid."
        ),
        "function": get_workspaces,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "is_destructive": False,
    },
    "search_tasks": {
        "name": "search_tasks",
        "description": (
            "Full-text search for tasks within a workspace. "
            "Use this to find a task by name or keyword before operating on it."
        ),
        "function": search_tasks,
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_gid": {
                    "type": "string",
                    "description": "GID of the workspace to search in.",
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for in task names and descriptions.",
                },
            },
            "required": ["workspace_gid", "query"],
        },
        "is_destructive": False,
    },
    "get_task": {
        "name": "get_task",
        "description": (
            "Retrieve full task details by GID. "
            "Returns assignee, projects, due date, notes, completion status, and more."
        ),
        "function": get_task,
        "parameters": {
            "type": "object",
            "properties": {
                "task_gid": {
                    "type": "string",
                    "description": "The GID of the task to retrieve.",
                },
            },
            "required": ["task_gid"],
        },
        "is_destructive": False,
    },
    "list_projects": {
        "name": "list_projects",
        "description": (
            "List all active projects in a workspace. "
            "Use this to find a project_gid when you only have the project name."
        ),
        "function": list_projects,
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_gid": {
                    "type": "string",
                    "description": "GID of the workspace.",
                },
            },
            "required": ["workspace_gid"],
        },
        "is_destructive": False,
    },
    "get_project": {
        "name": "get_project",
        "description": "Retrieve full project details by GID, including owner and members.",
        "function": get_project,
        "parameters": {
            "type": "object",
            "properties": {
                "project_gid": {
                    "type": "string",
                    "description": "The GID of the project to retrieve.",
                },
            },
            "required": ["project_gid"],
        },
        "is_destructive": False,
    },
    "get_tasks_for_project": {
        "name": "get_tasks_for_project",
        "description": (
            "List all tasks within a specific project. "
            "Returns up to 500 tasks with names, completion status, assignees, and due dates."
        ),
        "function": get_tasks_for_project,
        "parameters": {
            "type": "object",
            "properties": {
                "project_gid": {
                    "type": "string",
                    "description": "The GID of the project.",
                },
            },
            "required": ["project_gid"],
        },
        "is_destructive": False,
    },
    "get_users": {
        "name": "get_users",
        "description": (
            "List all users in a workspace. "
            "Use this to resolve a person's name to an assignee_gid before creating or updating a task."
        ),
        "function": get_users,
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_gid": {
                    "type": "string",
                    "description": "GID of the workspace.",
                },
            },
            "required": ["workspace_gid"],
        },
        "is_destructive": False,
    },
    # ------------------------------------------------------------------
    # DESTRUCTIVE tools — require Human-in-the-Loop confirmation
    # ------------------------------------------------------------------
    "create_task": {
        "name": "create_task",
        "description": (
            "Create a new task in Asana. "
            "DESTRUCTIVE — requires user confirmation before execution. "
            "Resolve workspace_gid, project_gid, and assignee_gid before calling this."
        ),
        "function": create_task,
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_gid": {
                    "type": "string",
                    "description": "GID of the target workspace.",
                },
                "name": {
                    "type": "string",
                    "description": "Title of the new task.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional description / body of the task.",
                },
                "assignee_gid": {
                    "type": "string",
                    "description": "GID of the user to assign the task to (optional).",
                },
                "project_gid": {
                    "type": "string",
                    "description": "GID of the project to add the task to (optional).",
                },
                "due_on": {
                    "type": "string",
                    "description": "Due date in YYYY-MM-DD format (optional).",
                },
            },
            "required": ["workspace_gid", "name"],
        },
        "is_destructive": True,
    },
    "update_task": {
        "name": "update_task",
        "description": (
            "Update one or more fields on an existing task. "
            "DESTRUCTIVE — requires user confirmation before execution. "
            "Updatable fields: name, notes, due_on, assignee (GID string), completed (bool)."
        ),
        "function": update_task,
        "parameters": {
            "type": "object",
            "properties": {
                "task_gid": {
                    "type": "string",
                    "description": "GID of the task to update.",
                },
                "name": {"type": "string", "description": "New task title."},
                "notes": {"type": "string", "description": "New task description."},
                "due_on": {"type": "string", "description": "New due date (YYYY-MM-DD)."},
                "assignee": {"type": "string", "description": "GID of the new assignee."},
            },
            "required": ["task_gid"],
        },
        "is_destructive": True,
    },
    "complete_task": {
        "name": "complete_task",
        "description": (
            "Mark a task as complete. "
            "DESTRUCTIVE — requires user confirmation before execution."
        ),
        "function": complete_task,
        "parameters": {
            "type": "object",
            "properties": {
                "task_gid": {
                    "type": "string",
                    "description": "GID of the task to mark as complete.",
                },
            },
            "required": ["task_gid"],
        },
        "is_destructive": True,
    },
    "delete_task": {
        "name": "delete_task",
        "description": (
            "Permanently delete a task from Asana. "
            "DESTRUCTIVE — this action CANNOT be undone. Requires user confirmation."
        ),
        "function": delete_task,
        "parameters": {
            "type": "object",
            "properties": {
                "task_gid": {
                    "type": "string",
                    "description": "GID of the task to permanently delete.",
                },
            },
            "required": ["task_gid"],
        },
        "is_destructive": True,
    },
}
