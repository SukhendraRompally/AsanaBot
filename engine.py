"""
engine.py — ReAct Reasoning Engine
Moveworks Agent Studio: Asana Reference Implementation

Implements the ReAct (Reason + Act) loop:
  Thought → Action → Observation → Thought → ... → Final Answer

This module has ZERO FastAPI imports. It is a pure Python async generator
that can be tested independently with `asyncio.run()`.

Key design decisions:
  - AsyncAzureOpenAI client (not sync) — never blocks the event loop
  - temperature=0.0 — deterministic tool selection, minimal hallucination
  - parse_error is "soft" — injects a correction into the conversation and
    retries rather than crashing; the loop is self-healing
  - Destructive actions PAUSE the loop and yield a confirmation_required event.
    app.py freezes the pending action into a session and the loop does not
    resume — on confirmation, the action is executed directly without re-calling
    the LLM (prevents non-determinism on retry)
  - max_iterations=10 guards against runaway loops
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from datetime import date
from typing import Any, AsyncGenerator, Literal

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from asana_tools import TOOL_REGISTRY, AsanaAPIError

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EngineConfig:
    """
    Holds all Azure OpenAI credentials and tuning parameters.
    Constructed once in app.py at startup and passed into run_react_loop.
    """
    azure_endpoint: str
    azure_api_key: str
    azure_deployment: str
    azure_api_version: str
    max_iterations: int = 10
    temperature: float = 0.0    # Deterministic for tool selection
    llm_timeout: float = 60.0   # Seconds before an LLM call times out


def config_from_env() -> EngineConfig:
    """
    Construct an EngineConfig from environment variables.
    Raises RuntimeError if any required variable is missing.
    """
    required = {
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "AZURE_OPENAI_VERSION": os.getenv("AZURE_OPENAI_VERSION"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

    return EngineConfig(
        azure_endpoint=required["AZURE_OPENAI_ENDPOINT"],
        azure_api_key=required["AZURE_OPENAI_KEY"],
        azure_deployment=required["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_api_version=required["AZURE_OPENAI_VERSION"],
    )


# ---------------------------------------------------------------------------
# Typed Event Dicts
# Yielded by run_react_loop and streamed as NDJSON by app.py
# ---------------------------------------------------------------------------

# Using plain dicts rather than TypedDicts/dataclasses so they serialize
# to JSON with json.dumps() without any custom encoder.

def _thought_event(content: str) -> dict:
    return {"type": "thought", "content": content}


def _action_event(tool: str, args: dict, is_destructive: bool) -> dict:
    return {
        "type": "action",
        "content": {"tool": tool, "args": args, "is_destructive": is_destructive},
    }


def _observation_event(content: Any) -> dict:
    return {"type": "observation", "content": content}


def _confirmation_required_event(tool: str, args: dict, message: str) -> dict:
    # Note: app.py injects `session_id` into content before sending to the client
    return {
        "type": "confirmation_required",
        "content": {"tool": tool, "args": args, "message": message},
    }


def _result_event(
    status: Literal["SUCCESS", "ERROR", "CANCELLED"],
    message: str,
    data: Any = None,
) -> dict:
    return {
        "type": "result",
        "content": {"status": status, "message": message, "data": data},
    }


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """
    Build the ReAct system prompt from TOOL_REGISTRY.
    Called once at module import time and cached in SYSTEM_PROMPT.
    """
    tool_lines: list[str] = []
    for spec in TOOL_REGISTRY.values():
        props = spec["parameters"].get("properties", {})
        required = spec["parameters"].get("required", [])
        param_parts = []
        for param_name, param_schema in props.items():
            suffix = "" if param_name in required else "?"
            param_parts.append(f"{param_name}{suffix}: {param_schema.get('type', 'any')}")
        params_str = ", ".join(param_parts) if param_parts else ""
        tag = "[DESTRUCTIVE]" if spec["is_destructive"] else "[READ-ONLY]"
        tool_lines.append(f'  {spec["name"]}({params_str}): {spec["description"]} {tag}')

    tool_block = "\n".join(tool_lines)
    today = date.today().isoformat()

    return f"""You are an intelligent Asana assistant built by Moveworks. You help users manage \
tasks, projects, and workspaces by reasoning step-by-step and calling tools.

TODAY'S DATE: {today}

AVAILABLE TOOLS:
{tool_block}

OUTPUT RULES — follow these exactly:
1. Always reason before acting. Write your reasoning in the "thought" field.
2. To call a tool, output EXACTLY one JSON object on a single line:
   {{"thought": "...", "action": {{"tool": "<tool_name>", "args": {{<args_as_key_value>}}}}}}
3. After receiving a tool result (Observation), continue reasoning with another JSON object.
4. When you have a complete answer for the user, output EXACTLY:
   {{"thought": "...", "final_answer": "<your complete, helpful answer>"}}
5. Never output markdown fences (``` or ```json). Output raw JSON only.
6. Never fabricate tool results. Only use data from Observations.
7. If a tool returns an error, acknowledge it in your Thought and try an alternative approach.
8. Never call a tool that is not in the AVAILABLE TOOLS list above.

GID RESOLUTION RULES:
- Asana GIDs are long numeric strings (e.g. "1234567890123456").
- If you have a workspace name but need its GID → call get_workspaces() first.
- If you have a project name but need its GID → call list_projects(workspace_gid) first.
- If you have a person's name but need their GID for assignee → call get_users(workspace_gid) first.
- Never guess a GID. Always resolve it from a tool call.
- If you are unsure of a GID (e.g. the user refers to "it" or "that task"), call the appropriate
  lookup tool to re-fetch the GID fresh before acting. Do NOT reuse a GID from memory if you are
  not 100% certain it came from a successful Observation in this conversation.

FINAL ANSWER RULES:
- When your final answer mentions specific tasks, projects, or users, ALWAYS include their GID
  in parentheses immediately after the name, e.g.: "Schedule kickoff meeting (GID: 1213741989707855)".
- This is critical: GIDs in the final answer carry into the next turn's context, allowing you to
  act on them without re-fetching. If you omit the GID, you will have to look it up again.

DESTRUCTIVE ACTION RULES:
- Tools marked [DESTRUCTIVE] modify or delete data in Asana.
- Before calling complete_task or delete_task, you MUST call get_task(task_gid) first to confirm
  the task GID is valid and get its name. Use the GID from that fresh Observation — never from memory.
- Before calling a [DESTRUCTIVE] tool, state clearly in your "thought" what you are about to do.
- The system will automatically pause and ask the user to confirm before executing [DESTRUCTIVE] actions.
- You do not need to ask the user yourself — just call the tool when you are ready.
"""


# Build and cache the system prompt at import time.
# If TOOL_REGISTRY changes, the process must be restarted.
SYSTEM_PROMPT: str = _build_system_prompt()


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def parse_llm_output(raw_text: str) -> dict:
    """
    Parse a raw LLM response into a structured dict.

    Returns one of:
      {"kind": "action",       "thought": str, "tool": str, "args": dict}
      {"kind": "final_answer", "thought": str, "answer": str}
      {"kind": "parse_error",  "raw": str, "reason": str}

    Never raises — all exceptions are caught and converted to parse_error.
    This keeps the engine loop self-healing.
    """
    try:
        # Strip markdown code fences if the model adds them despite instructions
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Extract the first JSON object from the text
        # (some models add commentary before or after the JSON)
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start == -1 or brace_end == 0:
            return {
                "kind": "parse_error",
                "raw": raw_text,
                "reason": "No JSON object found in response.",
            }
        json_str = text[brace_start:brace_end]

        parsed = json.loads(json_str)

        if not isinstance(parsed, dict):
            return {
                "kind": "parse_error",
                "raw": raw_text,
                "reason": "Parsed JSON is not an object.",
            }

        thought = parsed.get("thought", "")

        if "final_answer" in parsed:
            return {
                "kind": "final_answer",
                "thought": thought,
                "answer": str(parsed["final_answer"]),
            }

        if "action" in parsed:
            action = parsed["action"]
            if not isinstance(action, dict):
                return {
                    "kind": "parse_error",
                    "raw": raw_text,
                    "reason": "'action' field is not an object.",
                }
            tool = action.get("tool", "")
            args = action.get("args", {})
            if not tool:
                return {
                    "kind": "parse_error",
                    "raw": raw_text,
                    "reason": "'action.tool' field is missing or empty.",
                }
            return {
                "kind": "action",
                "thought": thought,
                "tool": tool,
                "args": args if isinstance(args, dict) else {},
            }

        return {
            "kind": "parse_error",
            "raw": raw_text,
            "reason": "JSON object has neither 'action' nor 'final_answer' key.",
        }

    except json.JSONDecodeError as e:
        return {
            "kind": "parse_error",
            "raw": raw_text,
            "reason": f"JSON decode error: {e}",
        }
    except Exception as e:
        return {
            "kind": "parse_error",
            "raw": raw_text,
            "reason": f"Unexpected parse error: {e}",
        }


# ---------------------------------------------------------------------------
# Confirmation message builder
# ---------------------------------------------------------------------------

def build_confirmation_message(tool_name: str, args: dict) -> str:
    """
    Produce a human-readable English summary of what a destructive action will do.
    The Replit frontend displays this in the confirmation dialog.
    """
    templates: dict[str, str] = {
        "create_task": (
            "I'm about to create a new task named **\"{name}\"** "
            "in workspace `{workspace_gid}`"
            "{assignee_clause}{project_clause}{due_clause}. "
            "Do you want to proceed?"
        ),
        "update_task": (
            "I'm about to update task `{task_gid}` with the following changes: "
            "{changes}. Do you want to proceed?"
        ),
        "complete_task": (
            "I'm about to mark task `{task_gid}` as **complete**. "
            "Do you want to proceed?"
        ),
        "delete_task": (
            "I'm about to **permanently delete** task `{task_gid}`. "
            "⚠️ This action CANNOT be undone. Do you want to proceed?"
        ),
    }

    if tool_name == "create_task":
        assignee_clause = (
            f", assigned to `{args['assignee_gid']}`"
            if args.get("assignee_gid") else ""
        )
        project_clause = (
            f", in project `{args['project_gid']}`"
            if args.get("project_gid") else ""
        )
        due_clause = (
            f", due {args['due_on']}"
            if args.get("due_on") else ""
        )
        return templates["create_task"].format(
            name=args.get("name", "Untitled"),
            workspace_gid=args.get("workspace_gid", "?"),
            assignee_clause=assignee_clause,
            project_clause=project_clause,
            due_clause=due_clause,
        )

    if tool_name == "update_task":
        skip = {"task_gid"}
        changes = ", ".join(
            f"{k}={v!r}" for k, v in args.items() if k not in skip
        ) or "no changes specified"
        return templates["update_task"].format(
            task_gid=args.get("task_gid", "?"),
            changes=changes,
        )

    if tool_name == "complete_task":
        return templates["complete_task"].format(task_gid=args.get("task_gid", "?"))

    if tool_name == "delete_task":
        return templates["delete_task"].format(task_gid=args.get("task_gid", "?"))

    # Fallback for any future destructive tools
    return (
        f"I'm about to call **{tool_name}** with args: `{json.dumps(args)}`. "
        "Do you want to proceed?"
    )


# ---------------------------------------------------------------------------
# ReAct Engine
# ---------------------------------------------------------------------------

async def run_react_loop(
    user_message: str,
    config: EngineConfig,
    conversation_history: list[dict] | None = None,
    pending_action: dict | None = None,
    confirmed: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    The core ReAct (Reason + Act) loop.

    Three modes:
    ────────────
    1. Fresh request (pending_action=None, confirmed=False):
       Runs the full ReAct loop — calls the LLM, parses tool calls,
       executes read-only tools, pauses on destructive tools.

    2. Confirmation resume (pending_action=<dict>, confirmed=True):
       Skips the LLM entirely. Executes the frozen pending action directly
       and yields a result event. This prevents LLM non-determinism on retry.

    Multi-turn memory:
    ──────────────────
    conversation_history is a mutable list of {"role", "content"} dicts
    representing prior user/assistant exchanges (NOT the internal ReAct tool
    calls — just the summarised user message and final answer from each turn).
    The engine appends the current turn's exchange to this list in-place so
    app.py can persist it for the next request without any extra return value.

    Yields:
    ───────
    dict events of type: thought | action | observation | confirmation_required | result

    The caller (app.py) serializes each event to a JSON line (NDJSON).
    """
    # ── Mode 2: Execute confirmed destructive action ──────────────────────
    if pending_action and confirmed:
        tool_name = pending_action["tool"]
        tool_args = pending_action["args"]
        spec = TOOL_REGISTRY.get(tool_name)

        if not spec:
            yield _result_event(
                "ERROR",
                f"Confirmed action references unknown tool '{tool_name}'.",
            )
            return

        yield _action_event(tool_name, tool_args, is_destructive=True)

        try:
            # Safety net: for task-level destructive actions, verify the task GID
            # exists before executing. This catches hallucinated GIDs that slipped
            # through the LLM's reasoning (the most common failure mode).
            if tool_name in ("complete_task", "delete_task", "update_task"):
                task_gid = tool_args.get("task_gid")
                if task_gid:
                    from asana_tools import get_task
                    try:
                        await get_task(task_gid)
                    except AsanaAPIError as verify_err:
                        if verify_err.status_code == 404:
                            yield _result_event(
                                "ERROR",
                                f"Cannot execute '{tool_name}': task GID `{task_gid}` was not found "
                                f"in Asana. The agent may have used an incorrect GID. "
                                f"Please try again — ask the agent to look up the task by name first.",
                            )
                            return
                        raise  # re-raise non-404 errors (e.g. 403, 429) for normal handling

            result = await spec["function"](**tool_args)
            yield _observation_event(result)
            summary = f"Done! {tool_name} executed successfully."
            # Record the confirmed action in history so the next turn knows it happened
            if conversation_history is not None:
                conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "assistant", "content": summary})
            yield _result_event("SUCCESS", summary, result)
        except AsanaAPIError as e:
            yield _observation_event({"error": str(e), "status_code": e.status_code})
            yield _result_event("ERROR", f"Asana API error: {e}")
        except TypeError as e:
            yield _result_event("ERROR", f"Invalid arguments for {tool_name}: {e}")
        except Exception as e:
            logger.exception("Unexpected error executing confirmed action %s", tool_name)
            yield _result_event("ERROR", f"Unexpected error: {e}")
        return

    # ── Mode 1: Full ReAct loop ───────────────────────────────────────────
    llm = AsyncAzureOpenAI(
        api_key=config.azure_api_key,
        azure_endpoint=config.azure_endpoint,
        api_version=config.azure_api_version,
    )

    # Build conversation: system prompt + prior turns (summarised) + current message.
    # We do NOT inject the raw ReAct traces from prior turns — only the clean
    # user/assistant summaries. This keeps the context window lean and avoids
    # confusing the LLM with prior tool call JSON.
    conversation: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(conversation_history or []),
        {"role": "user", "content": user_message},
    ]

    valid_tool_names = list(TOOL_REGISTRY.keys())
    parse_error_count = 0
    MAX_PARSE_ERRORS = 3  # Stop retrying after 3 consecutive parse failures

    for iteration in range(config.max_iterations):
        logger.debug("ReAct iteration %d / %d", iteration + 1, config.max_iterations)

        # ── Call the LLM ──────────────────────────────────────────────────
        try:
            response = await llm.chat.completions.create(
                model=config.azure_deployment,
                temperature=config.temperature,
                timeout=config.llm_timeout,
                messages=conversation,
            )
        except Exception as e:
            # Surface LLM-level errors (timeout, rate limit, auth) as result events
            yield _result_event("ERROR", f"LLM error: {type(e).__name__}: {e}")
            return

        raw_text = response.choices[0].message.content or ""
        logger.debug("LLM raw response: %s", raw_text[:300])

        # ── Parse LLM output ──────────────────────────────────────────────
        parsed = parse_llm_output(raw_text)

        if parsed["kind"] == "parse_error":
            parse_error_count += 1
            logger.warning(
                "parse_error (#%d): %s — raw: %s",
                parse_error_count, parsed["reason"], raw_text[:200]
            )
            if parse_error_count >= MAX_PARSE_ERRORS:
                yield _result_event(
                    "ERROR",
                    f"The LLM produced {MAX_PARSE_ERRORS} consecutive invalid responses. "
                    "Please try rephrasing your request.",
                )
                return
            # Inject correction into conversation and retry
            conversation.append({"role": "assistant", "content": raw_text})
            conversation.append({
                "role": "user",
                "content": (
                    f"Your last response was not valid. Reason: {parsed['reason']}. "
                    "Output ONLY a single JSON object with either "
                    "{\"thought\": \"...\", \"action\": {\"tool\": \"...\", \"args\": {...}}} "
                    "or {\"thought\": \"...\", \"final_answer\": \"...\"}. "
                    "No markdown, no extra text."
                ),
            })
            continue

        # Successful parse — reset error counter
        parse_error_count = 0

        # ── Emit thought ──────────────────────────────────────────────────
        if parsed.get("thought"):
            yield _thought_event(parsed["thought"])

        # ── Final answer ──────────────────────────────────────────────────
        if parsed["kind"] == "final_answer":
            # Persist this turn as a clean user/assistant exchange for future turns.
            # Stored in conversation_history (mutated in-place) — app.py saves it.
            if conversation_history is not None:
                conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "assistant", "content": parsed["answer"]})
            yield _result_event("SUCCESS", parsed["answer"])
            return

        # ── Tool call ─────────────────────────────────────────────────────
        tool_name: str = parsed["tool"]
        tool_args: dict = parsed["args"]

        # Guard: hallucinated tool name
        if tool_name not in TOOL_REGISTRY:
            logger.warning("LLM hallucinated tool name: '%s'", tool_name)
            conversation.append({"role": "assistant", "content": raw_text})
            conversation.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' does not exist. "
                    f"Available tools: {valid_tool_names}. "
                    "Choose a tool from that list and try again."
                ),
            })
            continue

        spec = TOOL_REGISTRY[tool_name]
        yield _action_event(tool_name, tool_args, spec["is_destructive"])

        # ── Destructive action: pause and request confirmation ─────────────
        if spec["is_destructive"]:
            message = build_confirmation_message(tool_name, tool_args)
            yield _confirmation_required_event(tool_name, tool_args, message)
            # Return here — app.py will store the pending action in the session.
            # The loop does NOT continue. Confirmation is handled via a new request.
            return

        # ── Read-only action: execute immediately ─────────────────────────
        try:
            result = await spec["function"](**tool_args)
        except AsanaAPIError as e:
            result = {"error": str(e), "status_code": e.status_code}
        except TypeError as e:
            result = {"error": f"Invalid arguments for {tool_name}: {e}"}
        except Exception as e:
            logger.exception("Unexpected error calling tool %s", tool_name)
            result = {"error": f"Unexpected error: {e}"}

        yield _observation_event(result)

        # Append both the assistant's raw JSON and the observation to the conversation
        conversation.append({"role": "assistant", "content": raw_text})
        conversation.append({
            "role": "user",
            "content": f"Observation: {json.dumps(result, default=str)}",
        })

    # Fell out of the loop — max iterations reached
    yield _result_event(
        "ERROR",
        f"Reached the maximum of {config.max_iterations} reasoning steps without "
        "a final answer. Try breaking your request into smaller pieces.",
    )
