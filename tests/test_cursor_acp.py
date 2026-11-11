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
    assert p.aliases == ("cursor-agent", "cursor-cli")
    assert "cursor" not in p.aliases
    assert "cursor-subscription" not in p.aliases
    assert p.auth_type == "external_process"
    assert p.base_url.startswith("acp://")
    assert p.process_command == "agent" and tuple(p.process_args) == ("acp",)


def test_existing_cli_json_is_backed_up_before_overwrite(plugin, tmp_path):
    cfg = tmp_path / ".cursor" / "cli.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    custom = '{"permissions": {"allow": ["Read(**)"], "deny": []}}\n'
    cfg.write_text(custom, encoding="utf-8")
    plugin._write_deny_permissions(tmp_path)
    assert json.loads(cfg.read_text(encoding="utf-8")) == plugin._DENY_PERMISSIONS
    assert (tmp_path / ".cursor" / "cli.json.bak").read_text(encoding="utf-8") == custom


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


GLM_UNTAGGED = ('{"id": "call_1", "type": "function", "function": {"name": "terminal", '
                '"arguments": "{\\"command\\": \\"echo X_$((6*7))\\"}"}}')


def test_untagged_tool_call_json_is_parsed_only_for_offered_tools(plugin):
    """Real glm-5p3 output: the call JSON with no <tool_call> tags. It was shown to the user as the answer."""
    calls, cleaned = plugin.extract_tool_calls(GLM_UNTAGGED, ["terminal"])
    assert [c.function.name for c in calls] == ["terminal"]
    assert json.loads(calls[0].function.arguments) == {"command": "echo X_$((6*7))"}
    assert cleaned == ""
    fenced, cleaned = plugin.extract_tool_calls("```json\n" + GLM_UNTAGGED + "\n```", ["terminal"])
    assert [c.function.name for c in fenced] == ["terminal"] and cleaned == ""
    # JSON that names a tool Hermes did not offer is prose, not a call.
    assert plugin.extract_tool_calls(GLM_UNTAGGED, ["read_file"])[0] == []
    assert plugin.extract_tool_calls(GLM_UNTAGGED)[0] == []


# Shapes copied from a live `agent acp` trace of refused native calls.
NATIVE_ATTEMPTS = [
    {"kind": "execute", "title": "`echo hi`", "rawInput": {"command": "echo hi"}},
    {"kind": "execute", "title": "`echo hi`", "rawInput": {"command": "echo hi"}},  # the model retried
    {"kind": "read", "title": "Read /etc/hosts", "rawInput": {"path": "/etc/hosts"}},
    {"kind": "search", "title": "grep", "rawInput": {"pattern": "localhost", "path": "/etc"}},
    {"kind": "other", "title": "hub: hub_list_servers",
     "rawInput": {"providerIdentifier": "hub", "toolName": "hub_list_servers", "args": {}}},
    {"kind": "search", "title": "grep", "rawInput": {}},  # input never arrived
]


def test_native_calls_translate_to_offered_hermes_tools(plugin):
    offered = ["terminal", "read_file", "search_files", "mcp_hub_hub_list_servers"]
    got = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
           for c in plugin.translate_native_calls(NATIVE_ATTEMPTS, offered)]
    assert got == [
        ("terminal", {"command": "echo hi"}),
        ("read_file", {"path": "/etc/hosts"}),
        ("search_files", {"pattern": "localhost", "path": "/etc"}),
        ("mcp_hub_hub_list_servers", {}),
    ]
    # Only tools Hermes offered this turn are ever emitted.
    assert [c["function"]["name"] for c in plugin.translate_native_calls(NATIVE_ATTEMPTS, ["read_file"])] == ["read_file"]
    assert plugin.translate_native_calls(NATIVE_ATTEMPTS, []) == []


TERMINAL_TOOL = [{"type": "function", "function": {
    "name": "terminal", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _scripted_client(plugin, tmp_path, monkeypatch, turns):
    """Client whose ACP session replays ``turns``: one (updates, text) pair per session/prompt."""
    import contextlib
    from types import SimpleNamespace

    client = plugin._make_client_class()(command="agent", acp_cwd=str(tmp_path))
    prompts: list[str] = []
    script = iter(turns)

    def request(method, params, *, text_parts=None, reasoning_parts=None):
        if method != "session/prompt":
            return {}
        prompts.append(params["prompt"][0]["text"])
        updates, text = next(script)
        for update in [*updates, {"sessionUpdate": "agent_message_chunk", "content": {"text": text}}]:
            client._handle_server_message(
                {"method": "session/update", "params": {"update": update}}, process=SimpleNamespace(stdin=None),
                cwd=str(tmp_path), text_parts=text_parts, reasoning_parts=reasoning_parts)
        return {"stopReason": "end_turn"}

    @contextlib.contextmanager
    def session(timeout_seconds, **_):
        yield {"sessionId": "s1", "models": {"availableModels": AVAILABLE}}, request

    monkeypatch.setattr(client, "_session", session)
    return client, prompts


def test_refused_native_call_is_rerouted_to_hermes_tool(plugin, tmp_path, monkeypatch):
    """The model used Cursor's Shell, the fence refused it, and the user was told to fix permissions."""
    refused = [
        {"sessionUpdate": "tool_call", "toolCallId": "t1", "kind": "execute", "status": "pending",
         "title": "`echo hi`", "rawInput": {"command": "echo hi"}},
        {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"},
    ]
    client, prompts = _scripted_client(plugin, tmp_path, monkeypatch, [
        (refused, "Blocked by permissions configuration. Run /permissions and allow Shell.")])
    msg = client.chat.completions.create(model="composer-2.5", tools=TERMINAL_TOOL,
                                         messages=[{"role": "user", "content": "say hi"}]).choices[0].message
    assert [c.function.name for c in msg.tool_calls] == ["terminal"]
    assert json.loads(msg.tool_calls[0].function.arguments) == {"command": "echo hi"}
    assert "permissions" not in msg.content  # the false "blocked" report never reaches the user
    assert len(prompts) == 1


def test_native_call_with_no_hermes_equivalent_gets_one_corrective_prompt(plugin, tmp_path, monkeypatch):
    refused = [
        {"sessionUpdate": "tool_call", "toolCallId": "t1", "kind": "other", "title": "MCP: tool", "rawInput": {}},
        {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "title": "hub: bos_tabs",
         "rawInput": {"providerIdentifier": "hub", "toolName": "bos_tabs", "args": {}}},
    ]
    client, prompts = _scripted_client(plugin, tmp_path, monkeypatch, [
        (refused, "MCP tool execution blocked."),
        ([], '<tool_call>{"name": "terminal", "arguments": {"command": "echo hi"}}</tool_call>'),
    ])
    msg = client.chat.completions.create(model="composer-2.5", tools=TERMINAL_TOOL,
                                         messages=[{"role": "user", "content": "say hi"}]).choices[0].message
    assert [c.function.name for c in msg.tool_calls] == ["terminal"]
    assert len(prompts) == 2 and "refused" in prompts[1] and "terminal" in prompts[1]


def test_bridged_call_ignores_native_attempts_from_the_same_turn(plugin, tmp_path, monkeypatch):
    """A model that tried Shell, then emitted the text block itself, must not run the command twice."""
    refused = [{"sessionUpdate": "tool_call", "toolCallId": "t1", "kind": "execute",
                "rawInput": {"command": "echo hi"}}]
    client, prompts = _scripted_client(plugin, tmp_path, monkeypatch, [
        (refused, '<tool_call>{"name": "terminal", "arguments": {"command": "echo hi"}}</tool_call>')])
    msg = client.chat.completions.create(model="composer-2.5", tools=TERMINAL_TOOL,
                                         messages=[{"role": "user", "content": "say hi"}]).choices[0].message
    assert len(msg.tool_calls) == 1 and len(prompts) == 1


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


LIVE_MODELS = [m.strip() for m in os.environ.get("CURSOR_LIVE_MODELS", "claude-sonnet-5-5").split(",") if m.strip()]
LIVE_TRIALS = int(os.environ.get("CURSOR_LIVE_TRIALS", "1"))
READ_FILE_TOOL = {"type": "function", "function": {
    "name": "read_file", "description": "Read a text file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}


@live
@pytest.mark.parametrize("trial", range(LIVE_TRIALS))
@pytest.mark.parametrize("model", LIVE_MODELS)
def test_live_two_step_task_completes(plugin, tmp_path, model, trial):
    """The whole job, per model: read a value it cannot guess, reason over it, act on the result.

    Covers every route a call can take (text block, untagged JSON, refused native tool rerouted).
    Pick models and repeats with CURSOR_LIVE_MODELS="a,b,c" CURSOR_LIVE_TRIALS=3.
    """
    import re
    import subprocess

    token, a, b = secrets.token_hex(6), secrets.randbelow(80) + 11, secrets.randbelow(80) + 11
    canary = tmp_path / "canary.txt"
    canary.write_text(f"{token} {a} {b}\n")
    want = f"DONE_{token}_{a * b + 7}"
    messages: list[dict] = [{"role": "user", "content": (
        f"Step 1: use the terminal tool to run: cat {canary} . It prints a word and two integers. "
        "Step 2: use the terminal tool to run: echo DONE_<word>_<N> where N is the product of the two "
        "integers plus 7. Reply with only the output of step 2.")}]
    client = plugin._make_client_class()()
    read, final = False, ""
    try:
        for _ in range(6):
            msg = client.chat.completions.create(model=model, messages=messages,
                                                 tools=[*TERMINAL_TOOL, READ_FILE_TOOL]).choices[0].message
            if not msg.tool_calls:
                final = msg.content or ""
                break
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls]})
            for c in msg.tool_calls:
                args = json.loads(c.function.arguments or "{}")
                cmd = str(args.get("command") or "").strip()
                if c.function.name == "read_file" and str(args.get("path")) == str(canary):
                    out = canary.read_text()
                elif c.function.name == "terminal" and re.fullmatch(
                        r"(cat\s+['\"]?%s['\"]?|echo\s+['\"]?DONE_\w+['\"]?)" % re.escape(str(canary)), cmd):
                    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
                else:
                    out = "refused by the test: only the two task commands may run"
                read = read or token in out
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": out})
    finally:
        client.close()
    assert read, f"{model} never ran step 1 through Hermes; it said: {final[-300:]!r}"
    assert want in final, f"{model} answered {final[-300:]!r}, wanted {want}"


@live
def test_live_unknown_model_fails_loudly(plugin):
    client = plugin._make_client_class()()
    try:
        with pytest.raises(RuntimeError, match="does not offer model"):
            client.chat.completions.create(model="not-a-model", messages=[{"role": "user", "content": "hi"}])
    finally:
        client.close()
