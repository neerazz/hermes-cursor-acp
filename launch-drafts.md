DRAFTS — hermes-cursor-acp launch (HELD, nothing posted)
Status: not sent. Each one needs Neeraj's go-ahead. Put source links in the first comment wherever the platform allows it.

──────────────────────────────────────────────
1. r/hermesagent (Showcase Thursday thread, or its own post flaired "Plugin")
Title: Use your Cursor subscription models in Hermes Agent with one npx command (and real tool calling)

Cursor doesn't have an OpenAI-compatible completions API, but the Cursor CLI includes an ACP server (`agent acp`). I built an official-CLI plugin that bridges Cursor models into Hermes:

npx hermes-cursor-acp

That installs and enables the plugin and verifies your Cursor CLI auth (`agent login`). Then:

hermes --provider cursor -m claude-opus-5-5
hermes --provider cursor -m composer-2.5

Key problems with standard wrappers that this solves:
- Zero-leak security fence: Cursor's native CLI tools run locally and can read any file on disk. The plugin rewrites a deny-all `.cursor/cli.json` (`Read`, `Write`, `Shell`, `WebFetch`, `Mcp`) inside an isolated workspace every run so Cursor can only provide the model.
- Tool-call bridging: Custom extractor catches both flat `{"name", "arguments"}` and inline `<tool_call>` shapes that default regexes drop.
- Compression protection: Cursor doesn't report token counts; this plugin emits `usage=None` so Hermes' proactive context compression doesn't stall.
- Automatic retries on concurrent `set_model` calls.

MIT licensed, 19 tests, npm provenance.
Feedback welcome!

──────────────────────────────────────────────
2. r/Cursor
Title: Run Hermes Agent on your Cursor subscription (via `agent acp`)

If you have a Cursor subscription and want to use its models (Claude Opus/Sonnet 5.5, Composer 2.5, GPT-5.6, Grok) inside Hermes Agent, you can connect them directly through Cursor's ACP server:

npx hermes-cursor-acp

hermes --provider cursor -m claude-opus-5-5

- Runs through Cursor CLI's native `agent acp` mode
- Isolated workspace with strict deny-list for native tools (Hermes runs all execution and approvals)
- Works with any model available on your plan (`agent models`)

Repo and details in the first comment.

──────────────────────────────────────────────
3. X / Twitter (single post; repo link in the reply)
Use your Cursor subscription models in Hermes Agent in one command:

npx hermes-cursor-acp

hermes --provider cursor -m claude-opus-5-5

Cursor only supplies the model. Hermes runs all tools, memory, and skills under an isolated deny-all fence so native CLI tools can't escape.

@NousResearch @cursor_run

──────────────────────────────────────────────
4. Hermes Discord (#plugins or #showcase)
cursor-acp: use your Cursor subscription models (Claude, Composer, GPT, Grok) in Hermes Agent via the Cursor CLI's ACP server.
`npx hermes-cursor-acp` → `hermes --provider cursor -m claude-opus-5-5`
Includes isolated workspace security deny-list, multi-format tool call parser, and set_model retries.
https://github.com/neerazz/hermes-cursor-acp

──────────────────────────────────────────────
First comment / reply (all platforms):
Repo: https://github.com/neerazz/hermes-cursor-acp
npm: https://www.npmjs.com/package/hermes-cursor-acp
License: MIT
