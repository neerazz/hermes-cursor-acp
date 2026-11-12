# hermes-cursor-acp

Use your Cursor subscription models (Claude, GPT, Grok, Gemini, Composer) as the model behind [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Cursor doesn't expose a chat-completions API, but the Cursor CLI ships an ACP server (`agent acp`). This plugin drives it through Hermes' ACP shim. Hermes keeps its own tools, memory, and skills. Cursor only supplies the model.

## How it works

- Hermes tool schemas go to the model as text. The model replies with `<tool_call>` blocks, and Hermes runs those calls itself. The plugin parses them with its own extractor, because Hermes' built-in bridge silently drops three shapes models emit: the flat `{"name", "arguments"}` form, a real block that follows a mention of `<tool_call>` in prose, and call JSON with no tags at all.
- Models still reach for Cursor's own Shell, Read, Grep, or MCP tools, most often the GLM, Kimi, and DeepSeek families. The fence refuses the call, and the model then tells you to fix permissions. The plugin reads the refused call from the ACP stream and runs the same action through the matching Hermes tool, so Hermes' approvals still apply. If no Hermes tool matches, it re-prompts the model once.
- The plugin rewrites `.cursor/cli.json` in an empty private workspace (`$HERMES_HOME/cache/cursor-acp-workspace`) every time a client starts, with a deny-all list (`Read`, `Write`, `Shell`, `WebFetch`, `Mcp`). This deny list is the security fence. Cursor's native tools run inside the CLI rather than over ACP, and without the list its Read tool can open any absolute path on disk.
- Models can be chosen by bare name (`claude-opus-5-5`) or by full advertised id (`claude-opus-5-5[context=300k,effort=high,fast=false]`). An unknown name raises an error instead of silently using Cursor's default.
- Responses report `usage=None`. Cursor sends no token counts, and a zero reading would stall Hermes' proactive context compression.
- `/usage` and `hermes usage` show the plan's two pools, "Cursor models" (Grok, Composer) and "Other models", with the same percentages as Cursor's dashboard. The plugin reads the Cursor CLI's login from the macOS Keychain and calls the endpoint Cursor's dashboard uses. That endpoint is undocumented and may change.

## Install

### Option 1: One-command install (recommended)

```bash
npx hermes-cursor-acp
```

This installs and enables the plugin in Hermes, verifies your Cursor CLI (`agent`) installation, and checks that your subscription models are accessible.

### Option 2: Manual install

```bash
curl https://cursor.com/install -fsS | bash   # Cursor CLI
agent login                                   # sign in with your Cursor account
agent models                                  # confirm what your plan offers

git clone git@github.com:neerazz/hermes-cursor-acp.git ~/.hermes/plugins/cursor-acp
hermes plugins enable cursor-acp
```

## Use

```bash
hermes --provider cursor-acp -m claude-opus-5-5
```

Or run `/model` inside Hermes and pick **Cursor Subscription (ACP)**.

## Settings (env)

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_CURSOR_ACP_COMMAND` | `agent` | Path to the Cursor CLI |
| `HERMES_CURSOR_ACP_ARGS` | `acp` | Argv tail |
| `HERMES_CURSOR_ACP_MODE` | `agent` | Cursor session mode. `ask` made gpt-5.6-sol refuse the bridge 2 times in 5. |
| `HERMES_CURSOR_ACP_CWD` | `$HERMES_HOME/cache/cursor-acp-workspace` | Private workspace |

## Tests

```bash
H=~/.hermes/hermes-agent
PYTHONPATH=$H uv run --no-project --python $H/venv/bin/python --with pytest python -m pytest tests -q -o addopts=
CURSOR_LIVE=1 ...   # adds live tests against the real CLI (needs `agent login`)
CURSOR_LIVE=1 CURSOR_LIVE_MODELS="glm-5p3,kimi-k3" CURSOR_LIVE_TRIALS=3 ... -k two_step   # two-step task per model
```

Live results on 2026-11-10 (Hermes v0.23.1, Cursor CLI 2026.10.22), driven through the plugin client:

- Two-step task (read a random value, compute from it, run a second command), 3 runs per model: deepseek-v41-flash-1017 3/3, glm-5p3 3/3, glm-5p3-flash 3/3, glm-5.2 3/3, kimi-k3 3/3, kimi-k2.7-code 2/3. Most GLM and DeepSeek runs passed only because a refused native call was rerouted.
- The same task, 2 runs per model: claude-sonnet-5-5, gpt-5.6-sol, grok-4.7, composer-2.5, gemini-4-carbon, and auto-smart all 2/2.

Live results on 2026-10-06 (Hermes v0.21.5, Cursor CLI 2026.10.01):

- Hermes tool-call bridging over 6 trials per model: claude-sonnet-5-5 6/6, claude-opus-5-5 6/6, composer-2.5 6/6, grok-4.7 6/6, gpt-5.6-sol 5/6. The one gpt miss answered from context without calling the tool.
- Fence probes sent as raw prompts, so the model was told to escape: Cursor's native Read, Grep, Shell, Write, and WebFetch were all blocked. The canary file content and a live GitHub SHA never came back.
- End-to-end multi-step runs through `hermes chat` (read_file, then terminal) executed both tools for real with composer-2.5 and gpt-5.6-sol.

## Known limits

- Every turn starts a new ACP session. That adds 7–16s of latency and about 17k tokens of Cursor's harness prompt.
- Hermes records no token or cost data for these calls, so the token counts in `/usage` and `/insights` undercount.
- Plan usage in `/usage` works on macOS only. The Linux lookup for the Cursor CLI's login is not implemented.
- When several sessions start at once, Cursor occasionally answers `set_model` with `Internal error`. The plugin retries up to 3 times.
- kimi-k2.7-code sometimes refuses the bridge in plain text without trying any tool. There is no call to reroute, so that turn ends with no action.

## License

MIT
