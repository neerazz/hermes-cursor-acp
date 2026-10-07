# hermes-cursor-acp

Use your Cursor subscription models (Claude, GPT, Grok, Gemini, Composer) as the model behind [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Cursor doesn't expose a chat-completions API, but the Cursor CLI ships an ACP server (`agent acp`). This plugin points Hermes' built-in ACP shim at it. Hermes keeps its own tools, memory, and skills. Cursor only supplies the model.

## How it works

- Hermes tool schemas are sent to the model as `<tool_call>` text, and Hermes runs those calls itself.
- Cursor's native tools (Shell, Edit, Read, …) are declared disabled in the prompt. The shim cancels any permission request they make.
- Cursor runs in an empty private workspace (`$HERMES_HOME/cache/cursor-acp-workspace`) in `ask` mode. The plugin rewrites `.cursor/cli.json` in that workspace on every client start with a deny-all list (`Read`, `Write`, `Shell`, `WebFetch`, `Mcp`). This is load-bearing: Cursor's native tools run inside the CLI, not over ACP, and without the deny list its Read tool can open any absolute path on disk (caught by `test_live_native_read_cannot_escape_workspace`).
- Models can be chosen by bare name (`claude-opus-5-5`) or by full advertised id (`claude-opus-5-5[context=300k,effort=high,fast=false]`). An unknown name raises an error instead of silently using Cursor's default.

## Install

```bash
curl https://cursor.com/install -fsS | bash   # Cursor CLI
agent login                                   # sign in with your Cursor account
agent models                                  # confirm what your plan offers

git clone git@github.com:neerazz/hermes-cursor-acp.git ~/.hermes/plugins/cursor-acp
hermes plugins enable cursor-acp
```

## Use

```bash
hermes --provider cursor -m claude-opus-5-5
```

Or run `/model` inside Hermes and pick **Cursor Subscription (ACP)**.

## Settings (env)

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_CURSOR_ACP_COMMAND` | `agent` | Path to the Cursor CLI |
| `HERMES_CURSOR_ACP_ARGS` | `acp` | Argv tail |
| `HERMES_CURSOR_ACP_MODE` | `ask` | Cursor session mode (`ask`, `plan`, `agent`; empty to skip) |
| `HERMES_CURSOR_ACP_CWD` | `$HERMES_HOME/cache/cursor-acp-workspace` | Private workspace |

## Known limits

- Every turn starts a new ACP session, which adds about 10s of latency and roughly 17k tokens of Cursor's harness prompt.
- Cursor doesn't report token usage over ACP, so Hermes records 0 tokens and no cost for these calls.
- Tool calling depends on the model following the text protocol. Strong models (Claude, GPT-5.x) do. Smaller ones may not.

Tested with Hermes v0.21.5 and Cursor CLI 2026.10.01.

## License

MIT
