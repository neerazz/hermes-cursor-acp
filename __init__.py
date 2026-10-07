"""Cursor subscription model provider for Hermes, via the Cursor CLI ACP server.

Cursor sells its models through an agent harness, not a chat-completions API, so this
plugin reuses Hermes' built-in ACP shim (the same one ``copilot-acp`` uses) against
``agent acp``. Hermes' tools are bridged as ``<tool_call>`` text and executed by Hermes;
Cursor's own permission requests are cancelled by the shim.

Isolation: Cursor runs in an empty private workspace and is switched to ``ask`` mode
(no edits, no command execution) so it cannot act on the repo by itself.

Models: pick by bare name (``claude-opus-5-5``, ``gpt-5.6-sol``, ``composer-2.5``) or by
the full advertised id (``claude-opus-5-5[context=300k,effort=high,fast=false]``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

CURSOR_MARKER_BASE_URL = "acp://cursor"
_DEFAULT_MODE = "ask"
# Cursor's harness prompt pushes its native tools (Shell, Read, Edit...). Those are
# cancelled by the shim, so the model must route every action through Hermes instead.
_HARNESS_OVERRIDE = (
    "HARNESS OVERRIDE — read first.\n"
    "You are the language model behind a different agent (Hermes). Every built-in Cursor tool "
    "(Shell, Read, Grep, Glob, Edit, Write, Search, MCP, todo) is DISABLED in this session and any "
    "call to one is automatically cancelled. Never call them.\n"
    "To act, emit one or more <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> blocks "
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
    """(Re)write ``<workspace>/.cursor/cli.json`` every time so drift can't loosen it."""
    import json

    cfg = workspace / ".cursor" / "cli.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_DENY_PERMISSIONS, indent=2) + "\n"
    if not cfg.exists() or cfg.read_text(encoding="utf-8") != body:
        cfg.write_text(body, encoding="utf-8")


def _bare(model_id: str) -> str:
    return str(model_id or "").split("[", 1)[0].strip()


def resolve_cursor_model(requested: str | None, available: list[dict[str, Any]]) -> str | None:
    """Map a Hermes model string onto one advertised Cursor modelId.

    Order: exact id -> exact display name -> bare id (text before ``[``).
    Returns None when nothing matches, so the session default is kept.
    """
    want = str(requested or "").strip()
    if not want or want in {"cursor", "cursor-acp", "default"}:
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

        def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
            with self._session(timeout_seconds, allow_file_requests=False) as (session, _):
                seen: list[str] = []
                for m in (session.get("models") or {}).get("availableModels") or []:
                    name = _bare(str(m.get("modelId") or "")) if isinstance(m, dict) else ""
                    if name and name not in seen:
                        seen.append(name)
                return seen

        def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model: str | None = None) -> tuple[str, str]:
            with self._session(timeout_seconds) as (session, _request):
                session_id = str(session.get("sessionId") or "").strip()
                available = (session.get("models") or {}).get("availableModels") or []
                model_id = resolve_cursor_model(model, available)
                if model and model_id is None and str(model).strip() not in {"cursor", "cursor-acp", "default"}:
                    raise RuntimeError(
                        f"Cursor does not offer model {model!r} on this account. "
                        "Run `agent models` to see the list.")
                if model_id:
                    _request("session/set_model", {"sessionId": session_id, "modelId": model_id})
                mode = os.environ.get("HERMES_CURSOR_ACP_MODE", _DEFAULT_MODE).strip()
                if mode:
                    try:
                        _request("session/set_mode", {"sessionId": session_id, "modeId": mode})
                    except Exception as exc:  # mode is defense in depth, not load-bearing
                        logger.warning("Cursor ACP set_mode(%s) failed: %s", mode, exc)
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                _request(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": [{"type": "text", "text": _HARNESS_OVERRIDE + prompt_text}]},
                    text_parts=text_parts, reasoning_parts=reasoning_parts,
                )
                return "".join(text_parts), "".join(reasoning_parts)

    return CursorACPClient


class CursorACPProfile(ProviderProfile):
    """Cursor subscription via ``agent acp`` (external process, no REST endpoint)."""

    def create_client(self, **client_kwargs: Any) -> Any:
        return _make_client_class()(**client_kwargs)

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
    aliases=("cursor", "cursor-agent", "cursor-subscription"),
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
