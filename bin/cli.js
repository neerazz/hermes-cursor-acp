#!/usr/bin/env node
// npx hermes-cursor-acp [install|status|uninstall]
// Thin wrapper over the official CLIs: `hermes plugins install` and Cursor's `agent`.
// No credentials or tokens pass through this script.
"use strict";

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
let REPO = pkg.repository?.url || "https://github.com/neerazz/hermes-cursor-acp";
if (REPO.startsWith("git+")) REPO = REPO.slice(4);
if (REPO.endsWith(".git")) REPO = REPO.slice(0, -4);
const PLUGIN = pkg.name.replace(/^hermes-/, "");
const CURSOR_INSTALLER = "https://cursor.com/install";
const HERMES_DOCS = "https://hermes-agent.nousresearch.com/docs";

const [, , cmd = "install", ...rest] = process.argv;
const flags = new Set(rest);
const yes = flags.has("--yes") || flags.has("-y");

function which(bin) {
  const r = spawnSync(process.platform === "win32" ? "where" : "which", [bin], { encoding: "utf8" });
  return r.status === 0 ? r.stdout.split(/\r?\n/)[0].trim() : null;
}

function run(bin, args, opts = {}) {
  console.log(`$ ${bin} ${args.join(" ")}`);
  const r = spawnSync(bin, args, { stdio: "inherit", shell: process.platform === "win32", ...opts });
  if (r.error) throw r.error;
  return r.status ?? 1;
}

function die(msg, code = 1) {
  console.error(`\n✗ ${msg}`);
  process.exit(code);
}

function requireHermes() {
  if (!which("hermes")) {
    die(`Hermes Agent not found on PATH. Install it first: ${HERMES_DOCS}`);
  }
}

async function ensureCursorCli() {
  if (which("agent")) return;
  if (!yes) {
    die(`Cursor CLI (agent) not found.\n  Install it with:\n    curl ${CURSOR_INSTALLER} -fsS | bash\n  or re-run with --yes to install it now.`);
  }
  console.log(`Installing Cursor CLI from ${CURSOR_INSTALLER}...`);
  const code = run("bash", ["-c", `curl ${CURSOR_INSTALLER} -fsS | bash`]);
  if (code !== 0 || !which("agent")) {
    die("Cursor CLI installation failed or `agent` is not on PATH. Open a new terminal and re-run.");
  }
}

async function install() {
  requireHermes();
  await ensureCursorCli();
  const ref = rest.find((a) => /^[0-9a-f]{40}$/.test(a));
  const args = ["plugins", "install", REPO, "--enable"];
  if (ref) args.push("--ref", ref);
  if (flags.has("--force")) args.push("--force");

  console.log(`Installing ${PLUGIN} into Hermes...`);
  if (run("hermes", args) !== 0) die("`hermes plugins install` failed (see output above).");

  console.log("\nVerifying Cursor CLI connection...");
  const modelsCheck = spawnSync("agent", ["models"], { encoding: "utf8" });
  if (modelsCheck.status !== 0) {
    console.log("\nNotice: Cursor CLI is installed but may not be signed in.");
    console.log("Run `agent login` in your terminal to authenticate your Cursor subscription.");
  } else {
    console.log("✓ Cursor CLI is authenticated and active.");
  }

  console.log(`\n✓ Done! Start Hermes with your Cursor subscription:\n`);
  console.log(`  hermes --provider cursor -m claude-opus-5-5`);
  console.log(`  hermes --provider cursor -m composer-2.5\n`);
  console.log(`Or select 'Cursor Subscription (ACP)' from /model inside Hermes.`);
}

function status() {
  requireHermes();
  console.log("Hermes: " + (which("hermes") || "not found"));
  console.log("Cursor CLI (`agent`): " + (which("agent") || "not found"));
  if (which("agent")) {
    console.log("\nChecking Cursor models:");
    run("agent", ["models"]);
  }
  console.log("\nHermes plugin status:");
  process.exit(run("hermes", ["plugins", "list"]));
}

function uninstall() {
  requireHermes();
  process.exit(run("hermes", ["plugins", "remove", PLUGIN]));
}

function help() {
  console.log(`hermes-cursor-acp — Cursor Subscription models for Hermes Agent

Usage: npx hermes-cursor-acp [command] [flags]

Commands:
  install     Install + enable the Hermes plugin, verify Cursor CLI (default)
  status      Show CLI and plugin status
  uninstall   Remove the plugin from Hermes

Flags:
  --yes, -y      Install the Cursor CLI automatically if missing
  --force        Reinstall over an existing copy
  <40-hex sha>   Pin the plugin to an exact commit

Platform: ${os.platform()} ${os.arch()}`);
}

const commands = { install, status, uninstall, help, "--help": help, "-h": help };
Promise.resolve((commands[cmd] || (() => { help(); process.exit(2); }))()).catch((err) => die(err.message || String(err)));
