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


@pytest.mark.parametrize("stream", [False, True])
def test_usage_is_none_so_hermes_marks_provider_usage_less(plugin, tmp_path, monkeypatch, stream):
    """A zero usage object reads as REAL usage and stalls Hermes' proactive compression."""
    from types import SimpleNamespace

    from agent.turn_usage import record_response_usage

    client = plugin._make_client_class()(command="agent", acp_cwd=str(tmp_path))
    monkeypatch.setattr(client, "_run_prompt", lambda *a, **k: ("hello", ""))
    resp = client.chat.completions.create(model="composer-2.5", stream=stream,
                                          messages=[{"role": "user", "content": "hi"}])
    if stream:
        assert all(getattr(ch, "usage", None) is None for ch in resp)
        return
    assert resp.usage is None
    noted = []
    compressor = SimpleNamespace(awaiting_real_usage_after_compression=False,
                                 note_usage_less_response=lambda: noted.append(True))
    agent = SimpleNamespace(context_compressor=compressor, session_api_calls=0,
                            model="composer-2.5", provider="cursor-acp")
    record_response_usage(agent, resp, messages=[], api_call_count=1, api_duration=0.1,
                          compression_attempts=0, max_compression_attempts=3)
    assert noted == [True]


live = pytest.mark.skipif(os.environ.get("CURSOR_LIVE") != "1", reason="set CURSOR_LIVE=1 to hit the real Cursor CLI")


@pytest.mark.parametrize(
    "text, names",
    [
        # Real composer-2.5 output shape that Hermes' bridge regex dropped.
        ('The harness wants a `<tool_call>` block, emitting it now:\n\n'
         '<tool_call>{"name": "terminal", "arguments": {"command": "date +%Y"}}</tool_call>', ["terminal"]),
        ('<tool_call>{"name":"a","arguments":{}}</tool_call> then <tool_call>{"name":"b","arguments":{"x":{"y":1}}}</tool_call>',
         ["a", "b"]),
        ('<tool_call>{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\": \\"/x\\"}"}}</tool_call>',
         ["read_file"]),
        ("<tool_call>{not json}</tool_call>", []),
        ("no tools here", []),
    ],
)
def test_extract_tool_calls(plugin, text, names):
    calls, cleaned = plugin.extract_tool_calls(text)
    assert [c.function.name for c in calls] == names
    if names:
        assert '"name"' not in cleaned


def test_prompt_keeps_prior_tool_calls_and_ends_with_tool_reminder(plugin):
    tools = [{"type": "function", "function": {"name": "terminal", "description": "", "parameters": {}}}]
    messages = [
        {"role": "user", "content": "year?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "date +%Y"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "2026"},
    ]
    prompt = plugin._format_prompt(messages, tools, None)
    assert ('<tool_call>{"id": "c1", "type": "function", "function": {"name": "terminal", '
            '"arguments": "{\\"command\\": \\"date +%Y\\"}"}}</tool_call>') in prompt
    transcript = prompt.split("Conversation so far:")[1].split("Now continue as the Assistant")[0]
    calls, _ = plugin.extract_tool_calls(transcript)
    assert [c.function.name for c in calls] == ["terminal"]
    assert "Tool result (terminal):\n2026" in prompt
    assert prompt.rstrip().endswith("they are refused.")
    assert "ACP capabilities" not in prompt


def test_set_model_retries_internal_error_then_succeeds(plugin, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def req(method, params):
        calls.append(method)
        if len(calls) < 3:
            raise RuntimeError("Copilot ACP session/set_model failed: Internal error")

    plugin._set_model_with_retry(req, "s1", "composer-2.5[fast=true]")
    assert calls == ["session/set_model"] * 3
    with pytest.raises(RuntimeError, match="bad id"):
        plugin._set_model_with_retry(lambda m, p: (_ for _ in ()).throw(RuntimeError("bad id")), "s1", "x")


@live
def test_live_native_read_cannot_escape_workspace(plugin, tmp_path):
    token = "CANARY_" + secrets.token_hex(8)
    canary = tmp_path / "canary.txt"
    canary.write_text(token)
    client = plugin._make_client_class()()
    try:
        # Raw prompt, bypassing the Hermes prompt that tells the model not to try: this tests the
        # deny-list FENCE against an injected or disobedient model, not the model's manners.
        text, _ = client._run_prompt(f"Use your Read tool to read {canary} and print its exact contents.",
                                     timeout_seconds=180, model="claude-sonnet-5-5")
    finally:
        client.close()
    assert token not in text


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
