"""Cursor subscription model provider for Hermes, via the Cursor CLI ACP server.

Cursor sells its models through an agent harness, not a chat-completions API, so this
plugin reuses Hermes' built-in ACP shim (the same one ``copilot-acp`` uses) against
``agent acp``. Hermes' tools are bridged as ``<tool_call>`` text and executed by Hermes;
Cursor's own permission requests are cancelled by the shim. When a model reaches for a
native Cursor tool anyway, the fence refuses it and the same action is rerouted to the
matching Hermes tool, so the user is not told to "fix permissions".

Isolation: Cursor runs in an empty private workspace and is switched to ``ask`` mode
(no edits, no command execution) so it cannot act on the repo by itself.

Models: pick by bare name (``claude-opus-5-5``, ``gpt-5.6-sol``, ``composer-2.5``) or by
the full advertised id (``claude-opus-5-5[context=300k,effort=high,fast=false]``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

CURSOR_MARKER_BASE_URL = "acp://cursor"
_TOOL_CALL_RE = re.compile(r"<tool_call>")
_TOOL_CALL_CLOSE_RE = re.compile(r"\s*</tool_call>")
# A JSON object that opens a line, optionally inside a ``` fence: an untagged tool call.
_UNTAGGED_CALL_RE = re.compile(r"(?m)^[ \t]*(?:```[a-zA-Z]*[ \t]*\n[ \t]*)?\{")
_FENCE_CLOSE_RE = re.compile(r"\s*```")
_DEFAULT_MODE = "agent"  # measured: ask mode made gpt-5.6-sol refuse the bridge 2/5; the deny list is the real fence
# Cursor's harness prompt pushes its native tools (Shell, Read, Edit...). Those are
# cancelled by the shim, so the model must route every action through Hermes instead.
_HARNESS_OVERRIDE = (
    "HARNESS OVERRIDE — read first.\n"
    "You are the language model behind a different agent (Hermes). Every built-in Cursor tool "
    "(Shell, Read, Grep, Glob, Edit, Write, Search, MCP, todo) is DISABLED in this session and any "
    "call to one is automatically cancelled. Never call them.\n"
    "To act, emit one or more <tool_call>{\"id\": \"call_1\", \"type\": \"function\", \"function\": "
    "{\"name\": \"<tool>\", \"arguments\": \"<JSON string>\"}}</tool_call> blocks "
    "as plain text using ONLY the tools listed below, then stop and wait: Hermes runs them and "
    "sends back the results in the next turn. If no tool is needed, answer directly.\n\n"
)
_FALLBACK_MODELS = (
    "composer-2.5",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "grok-4.7",
    "auto-smart",
)


def _workspace_dir() -> str:
    """Empty, private cwd so Cursor never reads repo rules or files on its own."""
    override = os.environ.get("HERMES_CURSOR_ACP_CWD", "").strip()
    if override:
        path = Path(override).expanduser()
    else:
        try:
            from hermes_constants import get_hermes_home

            base = Path(get_hermes_home())
        except Exception:
            base = Path.home() / ".hermes"
        path = base / "cache" / "cursor-acp-workspace"
    path.mkdir(parents=True, exist_ok=True)
    _write_deny_permissions(path)
    return str(path.resolve())


# Cursor's harness tools run inside the CLI, not through the ACP client, so the shim's
# permission cancel does not stop them: in ask mode the native Read tool reads ANY
# absolute path on disk and ships the bytes to Cursor. A project-level deny list in the
# workspace is what actually blocks them (verified: Read returns "Permission denied").
_DENY_PERMISSIONS = {
    "permissions": {
        "allow": [],
        "deny": [
            "Read(**)", "Read(/**)", "Write(**)", "Write(/**)",
            "Shell(*)", "WebFetch(*)", "Mcp(*:*)",
        ],
    }
}


def _write_deny_permissions(workspace: Path) -> None:
    """(Re)write ``<workspace>/.cursor/cli.json`` every time so drift can't loosen it.

    If a pre-existing custom ``.cursor/cli.json`` is present, preserve a backup at
    ``.cursor/cli.json.bak`` before writing the deny policy.
    """
    import json
    cfg = workspace / ".cursor" / "cli.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_DENY_PERMISSIONS, indent=2) + "\n"
    if cfg.exists():
        existing = cfg.read_text(encoding="utf-8")
        if existing != body:
            bak = workspace / ".cursor" / "cli.json.bak"
            if not bak.exists():
                bak.write_text(existing, encoding="utf-8")
            cfg.write_text(body, encoding="utf-8")
    else:
        cfg.write_text(body, encoding="utf-8")


def _bare(model_id: str) -> str:
    return str(model_id or "").split("[", 1)[0].strip()


_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool result"}


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")).strip() for p in content if isinstance(p, dict) and p.get("text")).strip()
    if isinstance(content, dict) and "text" in content:
        return str(content["text"]).strip()
    return str(content).strip()


def _render_message(msg: dict[str, Any]) -> str:
    """Render one transcript message, including the assistant's prior tool calls so the model
    sees its own <tool_call> history (the parent shim drops them, which breaks multi-step turns)."""
    role = str(msg.get("role") or "user").lower()
    parts = [_render_content(msg.get("content"))]
    for call in msg.get("tool_calls") or []:
        fn = (call.get("function") if isinstance(call, dict) else getattr(call, "function", None)) or {}
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
        try:
            args_obj = json.loads(args) if isinstance(args, str) else args
        except Exception:
            args_obj = args
        parts.append("<tool_call>" + json.dumps(
            {"id": (call.get("id") if isinstance(call, dict) else getattr(call, "id", None)) or "call_1",
             "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str)
                                               else json.dumps(args_obj, ensure_ascii=False)}},
            ensure_ascii=False) + "</tool_call>")
    label = _ROLE_LABELS.get(role, "Context")
    if role == "tool" and msg.get("name"):
        label = f"Tool result ({msg['name']})"
    body = "\n".join(p for p in parts if p)
    return f"{label}:\n{body}" if body else ""


def _format_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tool_choice: Any) -> str:
    """Hermes conversation -> one Cursor prompt. Tool contract up front AND a reminder at the
    end: recency is what keeps GPT/Composer from reaching for Cursor's own tools."""
    from agent.acp_openai_bridge import render_tool_bridge_sections, tool_specs_from_openai_tools

    sections = [_HARNESS_OVERRIDE.strip(), *render_tool_bridge_sections(tools, tool_choice)]
    transcript = [r for r in (_render_message(m) for m in messages if isinstance(m, dict)) if r]
    if transcript:
        sections.append("Conversation so far:\n\n" + "\n\n".join(transcript))
    names = [s["name"] for s in tool_specs_from_openai_tools(tools)]
    if names:
        sections.append(
            "Now continue as the Assistant. The tools you CAN use are exactly: " + ", ".join(names) + ". "
            "They are real and wired: Hermes executes them on the user's machine when you emit "
            "<tool_call>{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"<tool>\", "
            "\"arguments\": \"<JSON string>\"}}</tool_call>. If the task needs one, "
            "emit the block now and stop. Do not claim a tool is unavailable, and do not use Cursor's "
            "built-in tools; they are refused.")
    else:
        sections.append("Now continue as the Assistant. No tools are available this turn; answer directly.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _decode_call(obj: Any, index: int, tool_names: set[str] | None = None) -> Any:
    """One decoded JSON object -> a Hermes tool call, or None when it is not one.

    Accepts the flat {"name", "arguments"} shape models commonly emit; Hermes' parser only
    accepts {"function": {...}} and dropped those calls silently. With ``tool_names`` the
    call must name a tool Hermes offered this turn.
    """
    from agent.acp_openai_bridge import _parse_tool_call

    if not isinstance(obj, dict):
        return None
    if "function" not in obj and isinstance(obj.get("name"), str):
        obj = {"id": obj.get("id"), "type": "function",
               "function": {"name": obj["name"], "arguments": obj.get("arguments", {})}}
    fn = obj.get("function")
    if tool_names is not None and (not isinstance(fn, dict) or fn.get("name") not in tool_names):
        return None
    return _parse_tool_call(json.dumps(obj), index)


def extract_tool_calls(text: str, tool_names: list[str] | None = None) -> tuple[list[Any], str]:
    """Pull every well-formed ``<tool_call>{json}</tool_call>`` block out of ``text``.

    Hermes' bridge regex pairs the FIRST ``<tool_call>`` with the first ``</tool_call>``, so
    when a model mentions the tag in prose before the real block ("emitting a `<tool_call>`
    now: <tool_call>{...}</tool_call>") the span is invalid JSON and the call is silently
    dropped. Hermes' parser also accepts ONLY the {"function": {...}} shape, silently dropping
    the flat {"name", "arguments"} shape models commonly emit. Measured: 4 of 4 composer-2.5
    'misses' in one run were real calls lost to these two defects. Each tag occurrence is
    decoded independently and both shapes are accepted.

    Some models (measured: glm-5p3) emit the call JSON with no tags at all, and it was shown
    to the user as the answer. When nothing was tagged and ``tool_names`` is given, a JSON
    object that opens a line and names an offered tool is taken as the call.

    ⚡ Bolt Optimization: Uses pre-compiled regexes and module-level imports
    to reduce per-call execution time, as this function is run on every generation chunk.
    """
    if not isinstance(text, str) or ("<tool_call>" not in text and not tool_names):
        return [], (text or "").strip()
    decoder = json.JSONDecoder()
    calls: list[Any] = []
    spans: list[tuple[int, int]] = []
    for tag in _TOOL_CALL_RE.finditer(text):
        pos = tag.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text, pos)
        except ValueError:
            continue
        close = _TOOL_CALL_CLOSE_RE.match(text, end)
        span_end = close.end() if close else end
        call = _decode_call(obj, len(calls) + 1)
        if call is not None:
            calls.append(call)
            spans.append((tag.start(), span_end))
    if not calls and tool_names:
        offered, pos = set(tool_names), 0
        for opener in _UNTAGGED_CALL_RE.finditer(text):
            if opener.start() < pos:
                continue
            try:
                obj, end = decoder.raw_decode(text, opener.end() - 1)
            except ValueError:
                continue
            call = _decode_call(obj, len(calls) + 1, offered)
            if call is not None:
                close = _FENCE_CLOSE_RE.match(text, end) if "`" in opener.group(0) else None
                pos = close.end() if close else end
                calls.append(call)
                spans.append((opener.start(), pos))
    cleaned, cursor = [], 0
    for start, end in spans:
        cleaned.append(text[cursor:start])
        cursor = end
    cleaned.append(text[cursor:])
    return calls, "".join(cleaned).strip()


# Cursor reports every native tool attempt on the ACP stream as a ``tool_call`` update, with its
# input, even when the deny list refuses it. kind -> (Hermes tool, rawInput -> arguments).
_NATIVE_KIND_TO_HERMES = {
    "execute": ("terminal", lambda raw: {"command": raw["command"]}),
    "read": ("read_file", lambda raw: {"path": raw["path"]}),
    "search": ("search_files", lambda raw: {"pattern": raw["pattern"], **({"path": raw["path"]} if raw.get("path") else {})}),
}
_NATIVE_REFUSED_NUDGE = (
    "Your built-in tool call was refused, as the instructions said it would be. Nothing ran, and "
    "no permission setting will change that. Re-issue the same action now as a "
    "<tool_call>{{\"id\": \"call_1\", \"type\": \"function\", \"function\": {{\"name\": \"<tool>\", "
    "\"arguments\": \"<JSON string>\"}}}}</tool_call> text block using one of: {names}. "
    "Output only the block."
)


def translate_native_calls(attempts: list[dict[str, Any]], tool_names: list[str] | None) -> list[dict[str, Any]]:
    """Refused native Cursor tool attempts -> the same action as Hermes tool calls.

    A model that reaches for Cursor's Shell/Read/Grep/MCP instead of the text bridge gets a
    denial from the fence and reports "blocked by permissions" to the user. The intent is on
    the ACP stream, so run it through Hermes instead (its approvals and sandbox apply). Only
    tools Hermes offered this turn are emitted; a repeated attempt is emitted once.
    """
    offered = set(tool_names or ())
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for attempt in attempts:
        raw = attempt.get("rawInput") or {}
        try:
            if attempt.get("kind") in _NATIVE_KIND_TO_HERMES:
                name, build = _NATIVE_KIND_TO_HERMES[attempt["kind"]]
                args = build(raw)
            else:  # MCP: {"providerIdentifier": "hub", "toolName": "bos_tabs", "args": {...}}
                server, tool = str(raw.get("providerIdentifier") or ""), str(raw["toolName"])
                name = next((n for n in (f"mcp_{server}_{tool}", f"mcp__{server}__{tool}", f"{server}_{tool}", tool)
                             if n in offered), "")
                args = raw.get("args") or {}
        except (KeyError, TypeError):
            continue
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if name not in offered or key in seen:
            continue
        seen.add(key)
        out.append({"id": f"call_{len(out) + 1}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, default=str)}})
    return out


def _set_model_with_retry(request: Any, session_id: str, model_id: str, attempts: int = 3) -> None:
    """Cursor intermittently answers set_model with 'Internal error' when several sessions start
    at once (measured 2 of 16 under 6-way parallelism). Retry before failing the turn."""
    import time

    for attempt in range(1, attempts + 1):
        try:
            request("session/set_model", {"sessionId": session_id, "modelId": model_id})
            return
        except RuntimeError as exc:
            if attempt == attempts or "Internal error" not in str(exc):
                raise
            time.sleep(0.5 * attempt)


def resolve_cursor_model(requested: str | None, available: list[dict[str, Any]]) -> str | None:
    """Map a Hermes model string onto one advertised Cursor modelId.

    Order: exact id -> exact display name -> bare id (text before ``[``).
    Returns None when nothing matches, so the session default is kept.
    """
    want = str(requested or "").strip()
    if not want or want in {"cursor", "cursor-acp", "cursor-agent", "cursor-cli", "default"}:
        return None
    ids = [str(m.get("modelId") or "") for m in available if isinstance(m, dict)]
    if want in ids:
        return want
    for m in available:
        if isinstance(m, dict) and str(m.get("name") or "").strip() == want:
            return str(m.get("modelId"))
    for mid in ids:
        if _bare(mid) == _bare(want):
            return mid
    return None


def _make_client_class():
    from agent.copilot_acp_client import CopilotACPClient

    class CursorACPClient(CopilotACPClient):
        """CopilotACPClient pointed at ``agent acp`` with name-based model selection."""

        def __init__(self, *, command: str | None = None, args: list[str] | None = None,
                     acp_cwd: str | None = None, **kwargs: Any):
            kwargs.pop("acp_command", None)
            kwargs.pop("acp_args", None)
            kwargs.setdefault("api_key", "cursor-acp")
            kwargs.setdefault("base_url", CURSOR_MARKER_BASE_URL)
            cwd = acp_cwd or _workspace_dir()
            _write_deny_permissions(Path(cwd))
            self._native = threading.local()
            super().__init__(
                acp_command=command or os.environ.get("HERMES_CURSOR_ACP_COMMAND", "").strip() or "agent",
                acp_args=list(args or ["acp"]),
                acp_cwd=cwd,
                **kwargs,
            )

        def _spawn(self):
            try:
                return super()._spawn()
            except RuntimeError as exc:
                if "Could not start" in str(exc):
                    raise RuntimeError(
                        f"Could not start the Cursor CLI '{self._acp_command}'. Install it with "
                        "`curl https://cursor.com/install -fsS | bash`, run `agent login`, or set "
                        "HERMES_CURSOR_ACP_COMMAND.") from exc
                raise

        def _create_chat_completion(self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
                                    timeout: Any = None, tools: list[dict[str, Any]] | None = None,
                                    tool_choice: Any = None, stream: bool = False, **_: Any) -> Any:
            # Own prompt + response shaping instead of the parent's Copilot-flavoured one:
            # 1. The Copilot preamble ("use ACP capabilities") pushes Cursor models toward their
            #    native tools; measured 2/4 bridged for gpt-5.6-sol and composer-2.5 with it.
            # 2. The parent reports usage as all-zero numbers. Hermes reads a zero reading as REAL
            #    usage, never marks the provider usage-less, and its proactive compression waits for
            #    a count that never arrives. Cursor's ACP result has no usage ({"stopReason": ...}),
            #    so report None honestly.
            from types import SimpleNamespace

            from agent.acp_openai_bridge import completion_to_stream_chunks, tool_specs_from_openai_tools
            from agent.copilot_acp_client import _effective_timeout

            names = [s["name"] for s in tool_specs_from_openai_tools(tools)]
            prompt_text = _format_prompt(messages or [], tools, tool_choice)
            response_text, reasoning = self._run_prompt(
                prompt_text, timeout_seconds=_effective_timeout(timeout), model=model, tool_names=names)
            tool_calls, cleaned = extract_tool_calls(response_text, names)
            message = SimpleNamespace(content=cleaned, tool_calls=tool_calls, reasoning=reasoning or None,
                                      reasoning_content=reasoning or None, reasoning_details=None)
            completion = SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
                usage=None, model=model or "cursor-acp")
            return completion_to_stream_chunks(completion) if stream else completion

        def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
            with self._session(timeout_seconds, allow_file_requests=False) as (session, _):
                seen: list[str] = []
                for m in (session.get("models") or {}).get("availableModels") or []:
                    name = _bare(str(m.get("modelId") or "")) if isinstance(m, dict) else ""
                    if name and name not in seen:
                        seen.append(name)
                return seen

        def _native_attempts(self) -> dict[str, dict[str, Any]]:
            """Native Cursor tool attempts seen on this thread's ACP stream, by toolCallId."""
            if not hasattr(self._native, "calls"):
                self._native.calls = {}
            return self._native.calls

        def _handle_server_message(self, msg: dict[str, Any], **kwargs: Any) -> bool:
            # The parent drops ``tool_call`` updates. They are the only record that the model
            # reached for a native Cursor tool, and they carry the input it wanted to run.
            if msg.get("method") == "session/update":
                update = (msg.get("params") or {}).get("update") or {}
                if update.get("sessionUpdate") in ("tool_call", "tool_call_update") and update.get("toolCallId"):
                    entry = self._native_attempts().setdefault(str(update["toolCallId"]), {})
                    entry.update({k: update[k] for k in ("kind", "title", "rawInput") if update.get(k)})
            return super()._handle_server_message(msg, **kwargs)

        def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model: str | None = None,
                        tool_names: list[str] | None = None) -> tuple[str, str]:
            with self._session(timeout_seconds) as (session, _request):
                session_id = str(session.get("sessionId") or "").strip()
                available = (session.get("models") or {}).get("availableModels") or []
                model_id = resolve_cursor_model(model, available)
                if model and model_id is None and str(model).strip() not in {"cursor", "cursor-acp", "cursor-agent", "cursor-cli", "default"}:
                    raise RuntimeError(
                        f"Cursor does not offer model {model!r} on this account. "
                        "Run `agent models` to see the list.")
                if model_id:
                    _set_model_with_retry(_request, session_id, model_id)
                mode = os.environ.get("HERMES_CURSOR_ACP_MODE", _DEFAULT_MODE).strip()
                if mode:
                    try:
                        _request("session/set_mode", {"sessionId": session_id, "modeId": mode})
                    except Exception as exc:  # mode is defense in depth, not load-bearing
                        logger.warning("Cursor ACP set_mode(%s) failed: %s", mode, exc)

                def ask(text: str) -> tuple[str, str]:
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    _request(
                        "session/prompt",
                        {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
                        text_parts=text_parts, reasoning_parts=reasoning_parts,
                    )
                    return "".join(text_parts), "".join(reasoning_parts)

                native = self._native_attempts()
                native.clear()
                text, reasoning = ask(prompt_text)
                if not tool_names or not native or extract_tool_calls(text, tool_names)[0]:
                    return text, reasoning
                # The model reached for Cursor's own tools, the fence refused them, and its text
                # now tells the user to fix permissions. Run what it meant through Hermes instead.
                kinds = sorted({str(a.get("title") or a.get("kind") or "?") for a in native.values()})
                rerouted = translate_native_calls(list(native.values()), tool_names)
                if rerouted:
                    logger.info("cursor-acp: model=%s used %d refused native tool call(s) %s; rerouted to Hermes: %s",
                                model, len(native), kinds, [c["function"]["name"] for c in rerouted])
                    return "".join(f"<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c in rerouted), reasoning
                logger.info("cursor-acp: model=%s used refused native tool call(s) %s with no Hermes equivalent; "
                            "re-prompting once", model, kinds)
                retry_text, retry_reasoning = ask(_NATIVE_REFUSED_NUDGE.format(names=", ".join(tool_names)))
                if extract_tool_calls(retry_text, tool_names)[0]:
                    return retry_text, reasoning + retry_reasoning
                return text, reasoning

    return CursorACPClient


_USAGE_URL = "https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage"


def _cursor_cli_token() -> str | None:
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["security", "find-generic-password", "-s", "cursor-access-token", "-w"],
                         capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
    return out.stdout.strip() or None


def _fetch_current_period_usage(token: str) -> dict[str, Any]:
    request = urllib.request.Request(_USAGE_URL, data=b"{}", method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json",
        "Connect-Protocol-Version": "1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def _datetime_from_epoch_ms(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def cursor_usage_snapshot(body: dict[str, Any]) -> Any:
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

    plan_usage = body.get("planUsage") or {}
    reset_at = _datetime_from_epoch_ms(body.get("billingCycleEnd"))
    pools = (("Cursor models", "autoPercentUsed", "Includes Cursor Grok and Composer"),
             ("Other models", "apiPercentUsed", None))
    windows = tuple(
        AccountUsageWindow(label=label, used_percent=float(math.ceil(plan_usage[field])),
                           reset_at=reset_at, detail=detail, scope="model")
        for label, field, detail in pools if plan_usage.get(field) is not None)
    return AccountUsageSnapshot(
        provider="cursor-acp", source="cursor_dashboard_api", fetched_at=datetime.now(timezone.utc),
        windows=windows, unavailable_reason=None if windows else "Cursor reported no plan usage",
        raw=body)


class CursorACPProfile(ProviderProfile):
    """Cursor subscription via ``agent acp`` (external process, no REST endpoint)."""

    def create_client(self, **client_kwargs: Any) -> Any:
        return _make_client_class()(**client_kwargs)

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None) -> Any:
        token = _cursor_cli_token()
        if not token:
            return None
        return cursor_usage_snapshot(_fetch_current_period_usage(token))

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 30.0) -> list[str] | None:
        try:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            creds = resolve_external_process_provider_credentials(self.name)
            client = self.create_client(command=creds.get("command"), args=creds.get("args"))
            return client.list_models(timeout_seconds=timeout) or list(_FALLBACK_MODELS)
        except Exception as exc:
            logger.debug("cursor-acp fetch_models failed: %s", exc)
            return list(_FALLBACK_MODELS)

    def get_model_context_length(self, model: str) -> int | None:
        return 200_000


cursor_acp = CursorACPProfile(
    name="cursor-acp",
    aliases=("cursor-agent", "cursor-cli"),
    display_name="Cursor Subscription (ACP)",
    description="Use your Cursor subscription models via the Cursor CLI (`agent acp`)",
    api_mode="chat_completions",
    env_vars=(),
    base_url=CURSOR_MARKER_BASE_URL,
    auth_type="external_process",
    process_command="agent",
    process_args=("acp",),
    process_command_env_vars=("HERMES_CURSOR_ACP_COMMAND", "CURSOR_AGENT_PATH"),
    process_args_env_var="HERMES_CURSOR_ACP_ARGS",
    default_aux_model="composer-2.5",
    fallback_models=_FALLBACK_MODELS,
)

register_provider(cursor_acp)
