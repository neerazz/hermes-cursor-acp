"""Tests for the cursor-acp Hermes provider plugin.

Unit tests run anywhere Hermes is importable:
    PYTHONPATH=~/.hermes/hermes-agent ~/.hermes/hermes-agent/venv/bin/python -m pytest tests -q

Live tests drive the real Cursor CLI (needs `agent login`) and are opt-in:
    CURSOR_LIVE=1 PYTHONPATH=... python -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plugin():
    spec = importlib.util.spec_from_file_location("cursor_acp_under_test", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


AVAILABLE = [
    {"modelId": "claude-opus-5-5[context=300k,effort=medium,fast=false]", "name": "claude-opus-5-5"},
    {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
    {"modelId": "gpt-5.6-sol[context=272k,reasoning=medium,fast=false]", "name": "gpt-5.6-sol"},
]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("claude-opus-5-5", "claude-opus-5-5[context=300k,effort=medium,fast=false]"),
        ("composer-2.5[fast=true]", "composer-2.5[fast=true]"),
        ("gpt-5.6-sol[reasoning=high]", "gpt-5.6-sol[context=272k,reasoning=medium,fast=false]"),
        ("cursor", None),
        ("", None),
        ("not-a-model", None),
    ],
)
def test_resolve_cursor_model(plugin, requested, expected):
    assert plugin.resolve_cursor_model(requested, AVAILABLE) == expected


def test_profile_registered_as_external_process(plugin):
    p = plugin.cursor_acp
    assert p.name == "cursor-acp"
    assert "cursor" in p.aliases
    assert p.auth_type == "external_process"
    assert p.base_url.startswith("acp://")
    assert p.process_command == "agent" and tuple(p.process_args) == ("acp",)


def test_workspace_gets_deny_list_and_heals_tampering(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_ACP_CWD", str(tmp_path / "ws"))
    ws = Path(plugin._workspace_dir())
    cfg = ws / ".cursor" / "cli.json"
    deny = json.loads(cfg.read_text())["permissions"]["deny"]
    for rule in ("Read(/**)", "Write(/**)", "Shell(*)", "WebFetch(*)", "Mcp(*:*)"):
        assert rule in deny
    cfg.write_text('{"permissions": {"allow": ["Read(**)"], "deny": []}}')
    plugin._workspace_dir()
    assert json.loads(cfg.read_text()) == plugin._DENY_PERMISSIONS


def test_explicit_cwd_also_gets_deny_list(plugin, tmp_path):
    client = plugin._make_client_class()(command="agent", acp_cwd=str(tmp_path))
    assert (tmp_path / ".cursor" / "cli.json").exists()
    client.close()


def test_missing_cli_names_cursor_not_copilot(plugin, tmp_path):
    client = plugin._make_client_class()(command="/nonexistent/agent", acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match="Cursor CLI"):
        client.chat.completions.create(model="composer-2.5", messages=[{"role": "user", "content": "hi"}])


live = pytest.mark.skipif(os.environ.get("CURSOR_LIVE") != "1", reason="set CURSOR_LIVE=1 to hit the real Cursor CLI")


@live
def test_live_native_read_cannot_escape_workspace(plugin, tmp_path):
    token = "CANARY_" + secrets.token_hex(8)
    canary = tmp_path / "canary.txt"
    canary.write_text(token)
    client = plugin._make_client_class()()
    try:
        resp = client.chat.completions.create(
            model="claude-sonnet-5-5",
            messages=[{"role": "user", "content": f"Use your built-in Read tool to read {canary} and print its exact contents."}],
        )
    finally:
        client.close()
    assert token not in (resp.choices[0].message.content or "")


@live
def test_live_hermes_tool_call_is_bridged(plugin):
    tools = [{"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    client = plugin._make_client_class()()
    try:
        resp = client.chat.completions.create(
            model="claude-sonnet-5-5", tools=tools,
            messages=[{"role": "user", "content": "Run `echo BRIDGE_OK` with the terminal tool."}],
        )
    finally:
        client.close()
    calls = resp.choices[0].message.tool_calls or []
    assert calls and calls[0].function.name == "terminal"
    assert "BRIDGE_OK" in calls[0].function.arguments


@live
def test_live_unknown_model_fails_loudly(plugin):
    client = plugin._make_client_class()()
    try:
        with pytest.raises(RuntimeError, match="does not offer model"):
            client.chat.completions.create(model="not-a-model", messages=[{"role": "user", "content": "hi"}])
    finally:
        client.close()
