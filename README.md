# adbagent

Let an AI agent use your Android phone. Tell it what you want in plain language —
"turn on airplane mode", "read my latest messages and tell me who wrote" — and it
reads the screen, taps, swipes and types until the job is done, checking that
every step actually worked.

There are two ways to drive it:

- **From the AI agent you already use** — Claude Code, Codex, Antigravity, Claude
  Desktop or any other MCP client. Your agent's model makes the decisions, on the
  plan you already pay for, so no API key is needed.
  → [Use it from your AI agent](#use-it-from-your-ai-agent-mcp)
- **On its own, with an API key** — `adbagent run` for one-off goals,
  `adbagent watch` to keep looking after an app (answering messages under rules
  you write, say), and a web UI for both.
  → [Use it with your own model](#use-it-with-your-own-model-api-key)

Either way, it asks you before anything it can't undo — sending a message,
buying, deleting — and the model never sees password, PIN or payment screens.

```
$ adbagent run "turn on airplane mode"
    1 tap #12 "Network & internet"
    2 tap #7 "Airplane mode"

  ── Result ──

  Airplane mode is on; the toggle reads Enabled and the status bar shows the
      aeroplane icon.

  SUCCESS  2 steps, 3 LLM calls, $0.0091, 11.4s
  trace: runs/8f21c0a4e1b9 (events.jsonl, run.log, step prompts)
```

## Contents

- [Install](#install)
- [Connect your phone](#connect-your-phone)
- [Use it from your AI agent (MCP)](#use-it-from-your-ai-agent-mcp) —
  [Claude Code](#claude-code) · [Claude Desktop](#claude-desktop) ·
  [Codex](#codex) · [Antigravity](#antigravity) · [others](#other-mcp-clients)
- [Use it with your own model (API key)](#use-it-with-your-own-model-api-key)
- [Web UI](#web-ui)
- [Watch an app](#watch-an-app)
- [Notifications on your phone](#notifications-on-your-phone)
- [App skills](#app-skills)
- [Safety](#safety)
- [How it works](#how-it-works)
- [Runs, reports and replay](#runs-reports-and-replay)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Install

You need **Python 3.10 or newer** and Android's **platform tools** (`adb`):

- any system: [download the platform tools](https://developer.android.com/tools/releases/platform-tools)
  and add the folder to your `PATH`
- macOS with Homebrew: `brew install --cask android-platform-tools`
- Debian or Ubuntu: `sudo apt install adb`

Then install adbagent and create your config file:

```bash
git clone https://github.com/ganganimaulik/adb-automator.git
cd adb-automator
pip install -e .
cp config.example.json config.json     # PowerShell: Copy-Item config.example.json config.json
adbagent doctor
```

`doctor` checks Python, the dependencies, `adb`, the phone, the model, the API
key and notifications, and says what needs attention. If you will only drive
the phone from an AI agent over MCP, ignore what it says about the API key and
the model.

`config.json` holds your settings, and any keys and tokens you add. It is
git-ignored, so they stay on your machine. adbagent looks for it in the current
folder, then at `~/.config/adbagent/config.json`; `-c path/to/config.json` names
one explicitly. See [Configuration](#configuration).

## Connect your phone

First, on the phone: turn on **Developer options** (*Settings → About phone*,
then tap *Build number* seven times), then turn on **USB debugging** inside
Developer options.

**USB.** Plug the phone in, accept the *Allow USB debugging?* prompt on its
screen, and check that it is listed:

```bash
adbagent devices
```

**Wi-Fi (Android 11+).** In *Developer options → Wireless debugging*, tap *Pair
device with QR code* and scan the code this prints:

```bash
adbagent pair-qr
```

Or tap *Pair device with pairing code* and pass the address and code it shows:

```bash
adbagent pair 192.168.1.50:37115 --code 123456
```

The pairing port and the connection port are different, and the connection port
changes every time Wireless debugging is switched on. `pair` finds it by itself
where it can; if it can't, add `--connect ip:port` with the address shown on the
Wireless debugging screen.

Pairing saves the phone in `config.json`, so later commands find it without
being told. With more than one phone or emulator attached, pick one with
`-d SERIAL`, or set `device.serial` in `config.json`.

**Emulators.** An Android Studio emulator is listed as `emulator-5554` with no
setup. For BlueStacks, switch on *Settings → Advanced → Android Debug Bridge*
and run `adb connect` with the address it shows (usually `127.0.0.1:5555`).

Keep the phone unlocked while adbagent works: it can't, and won't, enter your
PIN.

## Use it from your AI agent (MCP)

`adbagent mcp` makes your phone available to your AI agent as a set of
[MCP](https://modelcontextprotocol.io) tools. The agent decides each step with
its own model, and adbagent does the rest: it reads the screen, performs each
action and checks that it worked, refuses moves already known to lead nowhere,
asks **you** before sends and irreversible presses, and records every session
like any other run — in the web UI's History, and in `adbagent report`.

You need no API key and no model settings: only adbagent installed, your phone
connected (`adbagent devices` lists it) and a `config.json`.

### 1. Get the command right

Your agent starts the server itself — usually from some other folder, and often
without your terminal's `PATH` — so give it **full paths**:

- **The command.** Find it with `which adbagent` (macOS, Linux) or
  `(Get-Command adbagent).Source` (Windows PowerShell). If your agent can't run
  it, use Python instead: the command becomes the full path of `python` (or
  `python.exe`), and the arguments start with `-m adbagent`. Print that path with
  `python -c "import sys; print(sys.executable)"`.
- **Your config.** Always pass `-c` with the absolute path of your
  `config.json`. The `runs/`, `skills/` and `policies/` folders next to it are
  then the ones every session uses, so all of them land in one history and share
  the same app skills and policies, whatever folder your agent was opened in.
  Double-check the path: a config file that doesn't exist is skipped without a
  warning.

So the server command is:

```bash
adbagent mcp -c /path/to/adb-automator/config.json
```

Run it once in a terminal to check it: it should print nothing and sit waiting
for an agent to talk to it, which means it works (close the terminal when you
are done). Any error it shows is the error your agent would get. On Windows the
path looks like `C:\Users\you\adb-automator\config.json`; each client below
shows how to write one.

### 2. Add it to your agent

In Windows PowerShell, if `claude` or `codex` was installed with npm
(`Get-Command claude` shows a `.ps1` file), write the `--` below as `'--'`:
PowerShell drops a bare `--` before such a script sees it.

#### Claude Code

```bash
claude mcp add --scope user adbagent -- adbagent mcp -c /path/to/adb-automator/config.json
```

Everything after `--` is the server's command. `--scope user` makes adbagent
available in all your projects; leave it out to add it to the current project
only. On Windows, put the path in quotes: `-c "C:\Users\you\adb-automator\config.json"`.

Check it with `claude mcp list`, which should say *Connected*, or with `/mcp`
inside a session. adbagent's approval questions appear as a dialog in Claude
Code.

Claude Code also asks your permission before every tool call. To let adbagent's
tools run without that — adbagent will still ask you before any send or
irreversible press — add this to `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["mcp__adbagent__*"]
  }
}
```

#### Claude Desktop

Open *Settings → Developer → Edit Config*. That opens
`claude_desktop_config.json` (in `%APPDATA%\Claude\` on Windows, and in
`~/Library/Application Support/Claude/` on macOS). Add adbagent to it:

```json
{
  "mcpServers": {
    "adbagent": {
      "command": "/full/path/to/adbagent",
      "args": ["mcp", "-c", "/path/to/adb-automator/config.json"]
    }
  }
}
```

Then quit Claude Desktop completely and open it again. In JSON, a Windows path
needs every backslash doubled: `"C:\\Users\\you\\adb-automator\\config.json"`.

#### Codex

```bash
codex mcp add adbagent -- adbagent mcp -c /path/to/adb-automator/config.json
```

That writes to `~/.codex/config.toml` (`%USERPROFILE%\.codex\config.toml` on
Windows), which the Codex CLI, IDE extension and app all read. Then open the file
and add the settings `codex mcp add` can't write:

```toml
[mcp_servers.adbagent]
command = "adbagent"
args = ["mcp", "-c", "/path/to/adb-automator/config.json"]
required = true                          # wait for adbagent before the first message
startup_timeout_sec = 30
tool_timeout_sec = 300
default_tools_approval_mode = "approve"  # optional: no prompt before each tool call
```

- **`required` and the timeouts.** adbagent takes a few seconds to start (about
  six on a Windows laptop). Without `required`, Codex gives a slow server about a
  second before the first message, so a new session's first message may not see
  adbagent's tools. Older versions of Codex also had much shorter timeouts.
- **`default_tools_approval_mode`.** Codex asks before every call to a tool that
  doesn't declare itself read-only. `"approve"` stops that, and adbagent still
  asks you before any send or irreversible press. `codex exec`, which never asks,
  needs it to use adbagent's tools at all.
- **Approvals** appear as an Allow / Deny prompt in current versions of Codex
  (0.119 and later). **Don't use Full Access (`--yolo`) with adbagent:** in that
  mode Codex answers adbagent's approval questions itself, with yes. Where Codex
  never asks (`codex exec`), it declines them instead.
- **Environment.** Codex passes an MCP server only a few environment variables
  (`PATH`, your home folder and the like), so keep the phone's serial and your
  Telegram settings in `config.json` rather than in environment variables.
- **Windows.** Write paths as single-quoted TOML strings, which keep backslashes
  as they are: `args = ['mcp', '-c', 'C:\Users\you\adb-automator\config.json']`.
  Run Codex natively rather than in WSL, which can't see a USB phone without
  extra setup.

Type `/mcp` in a session to check that adbagent is connected. A running session
keeps the servers it started with, so after changing the config start a new one
(`/new`), or restart the IDE extension or the app.

#### Antigravity

Antigravity's App, IDE and CLI share one list of MCP servers,
`~/.gemini/config/mcp_config.json` (`%USERPROFILE%\.gemini\config\mcp_config.json`
on Windows). Open it from:

- **the App:** *Settings → Customizations → Installed MCP Servers*
- **the IDE:** the **…** menu at the top of the agent panel → *MCP Servers* →
  *Manage MCP Servers* → *View raw config*
- **the CLI:** `/mcp`

Add adbagent to it. This uses the `python -m adbagent` form, which works
whatever `PATH` Antigravity has:

```json
{
  "mcpServers": {
    "adbagent": {
      "command": "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["-m", "adbagent", "mcp", "-c", "C:\\Users\\you\\adb-automator\\config.json"]
    }
  }
}
```

Then refresh the server list, or restart Antigravity. (Antigravity also
documents a per-workspace `.agents/mcp_config.json`, but support for it varies
between its apps and versions; the global file is the dependable choice.)

Things to know:

- **Approvals.** Antigravity doesn't document support for MCP approval
  questions, so expect adbagent to decline sends and irreversible presses — or
  decide them up front with `--draft`, `--no-send-approval` or
  `--allow-destructive` (see [What needs your OK](#what-needs-your-ok)).
- **Tool permissions.** Antigravity asks before every MCP tool call until you
  allow it; the permission rule `mcp(adbagent/*)` allows all of adbagent's
  tools. A scheduled conversation (an *Automation*) needs it too, since nobody is
  there to click.
- **Screenshots.** Antigravity saves images that MCP tools return to a file
  instead of showing them inline, so its agent may only see adbagent's
  screenshots if it opens that file. It can still read every screen from the
  element list.
- **One server for every conversation.** Antigravity starts a single adbagent
  server and all its conversations share it. Each session belongs to the
  conversation that started it, so a second conversation — an overlapping
  scheduled run, say — is told the phone is busy instead of taking over.
- **New adbagent code needs a restart.** Antigravity keeps the server it
  started: after updating adbagent, choose **Restart** on the server's menu under
  Installed MCP Servers, or restart Antigravity. Policy edits need no restart —
  they are read at the start of every session.

#### Other MCP clients

Cursor, Windsurf, opencode and other MCP clients take the same thing: a local
(**stdio**) server whose command is `adbagent` (as a full path) with the
arguments `mcp -c /path/to/config.json`. Most of them use the same
`mcpServers` JSON as [Claude Desktop](#claude-desktop).

### 3. Ask for something

Mention adbagent or your phone, so the agent reaches for the right tools:

> Use adbagent to turn on dark mode on my phone.

> With adbagent, open my messages app and tell me who wrote to me today. Don't
> reply to anyone.

> Run one pass of my `whatsapp` policy with adbagent.

> Using adbagent, run my `whatsapp` policy and then my `instagram` one.

The last two run your [watch policies](#watch-an-app) — the rules, goal and send
caps you saved for an app — once each, in order. Ask the agent to list your
policies if you don't remember their names.

### What needs your OK

- **Sends and irreversible presses** — a message, a reply, a like, buying,
  deleting, uninstalling, signing out — stop and wait for you. The agent asks,
  and **you** answer in its window; the agent can't answer for you. (One client
  setting can: Codex's Full Access mode, see [Codex](#codex).)
- **Logins, one-time codes, captchas** — the agent can hand you the phone: you
  get a prompt, do it on the phone, and accept to give it back.
- **Password, PIN, one-time-code and payment screens** are never shown to the
  agent. The session ends there unless it has handed you the phone.

If your agent can't show adbagent's questions (it doesn't support MCP
*elicitation*), the action is declined — never taken — and the agent is told
which flag would allow it. Flags go after `mcp` in the server's arguments:

| flag | what it does |
|---|---|
| `--draft` | write sends but never send them — the safe way to try a policy |
| `--no-send-approval` | sends go out without asking you; `--draft` and a policy's `send_limits` still apply |
| `--allow-destructive` | irreversible presses go ahead without asking |
| `--idle-timeout MINUTES` | end a session after this long without a call (default 15) |
| `--max-steps N` | step budget for each session (default: `run.max_steps` from your config) |
| `-d SERIAL` | which phone, when more than one is attached |

**Unattended and scheduled sessions** have nobody to answer, so decide up front
what they may do: run with `--draft` until you like what it would send, then
with `--no-send-approval` and `send_limits` in your policies to cap what goes
out.

### One phone at a time

Only one thing drives the phone at a time: an MCP session, a run or a watch.
When something else has it, the agent is told what and since when, and should
stop and tell you. A session belongs to the conversation that started it, so two
conversations sharing one server can't act in, or end, each other's session.

A session with no call for 5 minutes lets go of the phone so something else can
use it (`device.let_go_after_s`), and its next call takes the phone back if it
is free. After 15 minutes without a call (`--idle-timeout`) the session ends.
Each session also has the step and time budget of a run — `run.max_steps` and
`run.max_wall_clock_s` in `config.json` — so raise those for long sessions.

### The tools your agent gets

| tool | what it does |
|---|---|
| `start_session` | takes the phone and returns the first screen — for a goal, a policy, or several policies in turn |
| `act` | does one or more actions, and returns what happened and the new screen |
| `look` | the current screen with a screenshot, optionally with a grid for tapping by position |
| `approve` | asks you whether a waiting send or irreversible press may go ahead |
| `hand_over` | gives you the phone for something only you can do, then takes it back |
| `end_session` | records the answer and gives the phone back |
| `list_policies` | your watch policies, with the goal and send caps of each |

The server comes with instructions that teach the agent how to read the screen
and use the actions, so your prompt only has to say what you want done.

## Use it with your own model (API key)

`adbagent run`, `adbagent watch`, skill generation and the web UI call a model
themselves, so they need an API key. Any OpenAI-compatible API works:
**Fireworks** is the default, **OpenAI** is built in, and any other service
works through `llm.base_url`.

### Choose a model

```bash
export FIREWORKS_API_KEY=fw_...        # PowerShell: $env:FIREWORKS_API_KEY = "fw_..."
adbagent models --vision --search kimi
```

`models` lists the Fireworks models you can chat with; `--vision` keeps the ones
that can read screenshots. Put the one you pick in `config.json` as `llm.model`
(`config.example.json` already names one), or pass `--model` to any command.
The key can live in `config.json` too, as `llm.api_key`.

For OpenAI, or any other OpenAI-compatible service:

```json
"llm": {
  "provider": "openai",
  "base_url": "https://api.example.com/v1",
  "api_key_env": "EXAMPLE_API_KEY",
  "model": "the-model-id"
}
```

For OpenAI itself, leave out `base_url` and use `"api_key_env": "OPENAI_API_KEY"`.

Each model setting has one job, and only `llm.model` is required:

| setting | used for | when unset |
|---|---|---|
| `llm.model` | deciding every step | — |
| `llm.model_image` | reading screenshots | `llm.model` |
| `llm.model_small` | double-checking "done" and every send | `llm.model` |
| `llm.model_skill` | exploring an app to write its skill | `llm.model` |
| `llm.model_skill_image` | reading screenshots while writing a skill | `llm.model_image` |

If `llm.model` can read images, set `llm.model_image` to the same model: each
screenshot then goes straight into the decision, one call instead of two.

### Run a goal

```bash
adbagent run "turn on dark mode" --dry-run    # decide every step, touch nothing
adbagent run "turn on dark mode"
adbagent run "check the loyalty app" --repeat inf --budget-usd 5
adbagent run --resume                         # continue the last failed or interrupted run
adbagent run --resume 8f21c0a4e1b9            # or a particular one
```

Name the app in the goal, and it finds and opens it. To see what is installed:

```bash
adbagent apps -s whatsapp
adbagent apps -3                 # third-party apps only
```

Every run ends with its answer under **Result**, so a "read X and tell me" goal
prints what it read. A run that fails or is interrupted keeps a checkpoint, and
`--resume` carries on from where it stopped, with a fresh step budget.

| flag | what it does |
|---|---|
| `--max-steps N` | step budget (default: `run.max_steps`) |
| `--budget-usd USD` | spend ceiling, across all repeats (default: `safety.budget_usd`) |
| `--repeat N` | do the goal N times, or `inf` for ever |
| `--dry-run` | decide every step but don't touch the phone |
| `--allow-destructive` | don't ask before irreversible actions |
| `--unattended` | never stop to ask — refuse instead (for runs nobody is watching) |
| `--no-learn` | don't update the app's skill from this run |
| `--always-screenshot`, `--never-screenshot` | a screenshot every step, or never |

### Tell it how to check success

The agent's own "done" is double-checked by a second model call. If success can
be checked mechanically, say how — it is free, instant and exact:

```bash
adbagent run "turn on airplane mode" \
  --assert-shell 'settings get global airplane_mode_on' --assert-equals 1

adbagent run "open the Wi-Fi screen" --assert-text "Forget network"
```

### See what the model sees

```bash
adbagent dump              # the element list the model is given for this screen
adbagent dump --detail 7   # every attribute of element 7
adbagent dump --raw        # the raw uiautomator XML
```

### Make it faster

Most of a run's time is the model thinking. With a reasoning model, let routine
steps think less and hard ones think properly:

```json
"llm": {
  "reasoning_effort": "low",
  "reasoning_effort_hard": "high"
}
```

A step counts as hard when the last action failed, the screen is new, or the run
is going in circles. Run `doctor` afterwards: it shows what each model will be
sent, and tells you when a model doesn't reason at all, in which case the
setting does nothing.

## Web UI

```bash
adbagent ui                 # http://127.0.0.1:8765
adbagent ui --port 9000
```

- **Work** — type a goal and start a run; watch it live, with the phone's screen
  beside the steps; read the answer. **Pause** and **Step** hold it between
  steps, and **Take the phone** lets you do something yourself (a login, a 2FA
  prompt) before **Resume**. Below is the **History** of every run, MCP sessions
  included, with search, outcome filters, cost per success, and **resume** for a
  run that stopped with a checkpoint. A run that stopped to ask you something
  gets a box to answer in.
- **Watch** — pick, write and edit policies, choose draft or live, tick several
  policies to run in order, and follow each pass live.
- **Setup** — the phone (choose a device, take a screenshot, list installed
  apps, see what the model sees, run doctor), the config editor (common
  settings first, the rest under *Advanced*, each one explained), and app skills
  (view, edit, generate).

A run started from the browser can't stop to ask you anything, so irreversible
actions are refused unless you tick **allow destructive**. Stop puts back the
phone's keyboard, animations and rotation, just like Ctrl+C. Run from a git
checkout, the UI picks up edits to its code, config, skills and policies by
itself; `--no-reload` turns that off.

## Watch an app

`run` pursues a goal and finishes. `watch` keeps looking after an app —
answering messages, working through a feed — and is meant to still be running
tomorrow. Between passes it only re-reads the screen, with no model call, so a
quiet inbox costs nothing.

```bash
adbagent watch "watch my instagram direct messages" --policy reply.md --draft
```

### Write a policy

A policy is a plain-language file of rules, given to the model word for word.
Start from [`policy.example.md`](policy.example.md):

```
Reply only to people I already follow. Keep it to one short sentence.
If someone asks to meet, say I'll check and get back to them — never commit
to a time. Anything about money, work, or an emergency: don't reply at all.
```

**Start with `--draft`.** Replies are written and logged but never sent, so a
policy that reads well but behaves badly costs you a wrong line in a log instead
of a wrong message in somebody's inbox. Drop `--draft` once the drafts look
right.

### Save the goal with the policy

Keep policies in `policies/` and give each a short header. The header is for
adbagent; only the rules below it reach the model.

```markdown
---
goal: work through unread chats and reply to anyone new
title: WhatsApp
---

# WhatsApp policy

- Never reply in a group chat…
```

Saved as `policies/whatsapp.md`, it starts by name:

```bash
adbagent watch --policy whatsapp --draft
```

The header can hold four things:

| key | example | what it does |
|---|---|---|
| `goal` | `work through unread chats and reply` | the goal the policy runs under, so it can be started by name |
| `title` | `WhatsApp` | the name the web UI shows |
| `send_limits` | `like=5, superlike=0` | caps for one pass, counted by adbagent: `like=5` allows five sends whose button mentions "like", and `0` forbids them |
| `snooze` | `You're all caught up = 30m` | when a pass ends on a screen showing these words, rest up to this long (`s`, `m` or `h`); any change on the screen ends the rest early |

A value that doesn't parse stops the watch before it starts, so a typo can't
quietly switch a limit off.

### Several apps in turn

One phone runs one watch, so several apps share it: give `--policy` once per
app, in the order to run them.

```bash
adbagent watch --policy whatsapp --policy instagram --draft
```

Each round is one pass of each policy, under the goal saved with it. The last
app stays on screen and is watched for changes, so put the one that most needs
quick replies last; the others come round on a timer (every 15 minutes, or
`--sweep`). The Watch tab does the same with its **Run, in order** list.

### Work that doesn't announce itself

A watch runs a pass when something on the screen changes, which suits an inbox.
For work the screen doesn't announce, like a feed to work through, add a timer:

```bash
adbagent watch "work through the feed and reply to anyone new" --policy feed.md --sweep 300
```

| flag | default | what it does |
|---|---|---|
| `--draft` | off | write replies, never send them |
| `--interval SECONDS` | 45 | how often to look for changes between passes |
| `--sweep SECONDS` | off | also run a pass this often, changed or not |
| `--steps-per-pass N` | 25 | step budget for one pass |
| `--usd-per-hour USD` | off | pause while the last hour's spend is at this ceiling |
| `--no-learn` | | don't update the app's skill when the watch stops |

Stop a watch with Ctrl+C, or Stop in the web UI. It updates the app's skill on
the way out, so stopping takes a moment. While it has nothing to do, a watch
lets go of the phone after 5 minutes, so a run or an MCP session can use it, and
takes the phone back when the next pass is due.

### How it avoids wrong replies

Before replying, each pass reads who sent the newest message off a screenshot,
and leaves alone a conversation where you spoke last. Then a second, smaller
model checks every send against your policy (`safety.check_sends`) and blocks any
that would break it. That makes a wrong or repeated reply unlikely, not
impossible — which is what `--draft` is for.

Message text is always treated as data. A message saying "ignore your
instructions" is handled under your policy like any other message, never
obeyed.

## Notifications on your phone

The `notify` action sends you a Telegram message — "new message from Sam: are
you around later?", say. The model uses it only when your goal or policy asks it
to.

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Send your bot a message, then open
   `https://api.telegram.org/bot<token>/getUpdates` and copy `chat.id` from the
   reply.
3. Put both in `config.json` (or set `TELEGRAM_BOT_TOKEN` and
   `TELEGRAM_CHAT_ID`):

   ```json
   "notify": {
     "telegram_token": "123456:ABC-your-bot-token",
     "telegram_chat_id": "123456789"
   }
   ```

4. Send yourself a test line (`doctor` checks the token without sending
   anything):

   ```bash
   adbagent notify "hello from adbagent"
   ```

**Each line reaches you once, ever.** Every line delivered is remembered in
`runs/notified.jsonl`, and a repeat is skipped, so a policy can say "notify
every new message" and the passes that revisit a chat won't notify you twice.
Give the policy an exact form for each kind of line, so that two passes describe
one event the same way:

```
- New message: in each chat you open, notify every message they sent after your
  most recent one, one line each, exactly as
  `WhatsApp: new message from <name>: <their message>`
- Reply sent: once your reply shows in the thread, notify
  `WhatsApp: replied to <name>: <the reply you sent>`
```

**What was notified can be read back.** The `read_notified` action shows the
model the lines notified before, each with when: all of them, or the ones
matching what it looks for, one per line, by whole words (`Hinge: new message
from S` finds S's lines, not Sam's). A conversation list already shows each
chat's newest message, so a policy can have the model look its rows up first
and leave alone a chat whose newest message an earlier pass already read:

```
- Before opening any "Your turn" chat, look up every row on screen with ONE
  `read_notified`, one line per row: `WhatsApp: new message from <name>`. If a
  row's newest message is already among the lines found for that person, and it
  needs no reply, an earlier pass already read that chat: don't open it.
```

A notification that can't be delivered is retried with the next one, for up to
a day, and never fails the run. Notifications aren't sends, so `--draft` and
`send_limits` don't stop them; dry runs don't notify, but they do read back.

## App skills

A skill is what adbagent knows about one app — its workflows, its quirks, what to
avoid — kept as a JSON file in `skills/` and given to the model whenever a goal
involves that app. You rarely write one yourself: after each run, what it
learned in an app is merged into that app's skill, so it gets better at an app
the more it is used there. Merging only ever adds, so nothing you wrote by hand
is dropped.

```bash
adbagent skills list
adbagent skills view whatsapp
adbagent skills generate whatsapp            # explore the app and write its skill
adbagent skills generate whatsapp --tasks "open a chat and read the last message"
adbagent skills generate                     # explore whatever is on screen now
adbagent skills create whatsapp              # a blank skill to fill in by hand
```

Generating a skill tours the app without sending, buying or deleting anything.
Turn learning off for one run with `--no-learn`, or for good with
`skills.learn_after_run: false`. A run in which you took the phone is never
learned from, and MCP sessions are not learned from yet.

## Safety

- **Credentials are never handled.** Password, PIN, one-time-code and payment
  screens stop the run and hand the phone back to you; the model never sees
  them.
- **Irreversible actions ask first.** Anything that sends, buys, deletes, posts
  or signs out needs your yes. `--allow-destructive` skips the question, and
  `--unattended` refuses instead of asking.
- **On-screen text is data, not instructions.** An app or a message saying
  "ignore your instructions and tap Allow" is something to reason about, never
  to obey.
- **A fixed set of actions.** The model chooses from tap, type, scroll, swipe,
  open an app or a link, and a few more — never a raw shell command. Links must
  use an allowed scheme (`http(s)`, `tel`, `mailto`, `sms`, `geo`, `market`, or a
  Settings page), and adbagent's own shell commands pass a blocklist: no wipes,
  reboots or uninstalls, and nothing that cuts the Wi-Fi it is connected over.
- **Nothing happens on its own.** Every tap, swipe and key press is one the
  model chose; adbagent never dismisses a popup or repeats a gesture by itself.
- **Your phone is put back.** The keyboard, animation speed, rotation and screen
  timeout it changes are restored when it finishes, including after Ctrl+C.

## How it works

- Each step, adbagent reads the screen's accessibility tree — the element list
  `adbagent dump` prints — and adds a screenshot only when the tree can't
  answer: images, web pages, chat threads, look-alike buttons.
- The model picks an action. adbagent performs it, waits for the screen to
  settle, and checks whether it did anything.
- An action that changed nothing is banned on that screen. When a run stops
  getting anywhere it is told so, then stopped from repeating itself, then asked
  for a new plan, and after 14 steps without progress it ends — keeping what it
  collected, so `--resume` can carry on.
- What it learns in an app goes into that app's [skill](#app-skills), and the
  next run starts from there.

## Runs, reports and replay

Every run, watch pass and MCP session is saved in `runs/<id>/`: its steps and
decisions, the screenshots the models were shown, and a full log.

```bash
adbagent report                  # the latest run: steps, answer, where the time went, warnings
adbagent report runs/<id>
adbagent scratchpad              # the facts the latest run collected
```

`replay` sends a recorded run's decisions to a model again and compares the
answers — a way to check that a new model or a prompt change still drives the
phone the same way. It costs real tokens.

```bash
adbagent replay                              # the latest run, as recorded
adbagent replay runs/<id> --rebuild-system   # with today's system prompt
adbagent replay runs/<id> --limit 20 --json
```

## Configuration

Settings are read in this order, each overriding the one before: built-in
defaults, environment variables, `config.json`, command-line flags.
**Setup → Config** in the web UI edits the same file and explains every setting.
Start from [`config.example.json`](config.example.json); these are the settings
you are most likely to change:

| setting | default | what it does |
|---|---|---|
| `llm.provider` | `fireworks` | `fireworks` or `openai`; for another OpenAI-compatible service, use `openai` with `llm.base_url` |
| `llm.base_url` | `""` | the API endpoint, when it isn't the provider's own |
| `llm.model` | `""` | the model that decides each step; there is no default, so pick one with `adbagent models` |
| `llm.api_key` | `""` | the API key itself; keep it here or in an environment variable |
| `llm.api_key_env` | `FIREWORKS_API_KEY` | the environment variable the API key is read from |
| `llm.vision_in_decider` | `false` | send screenshots straight to `llm.model`, when it can read images |
| `llm.reasoning_effort` | `""` | how hard a reasoning model thinks on routine steps: `none`, `low`, `medium`, `high`; empty sends nothing |
| `llm.reasoning_effort_hard` | `high` | how hard it thinks on steps that look difficult |
| `device.serial` | `""` | which phone to use; `pair` fills it in, and `-d` overrides it |
| `device.let_go_after_s` | `300` | seconds after which an idle watch or MCP session lets go of the phone for others; `0` never does |
| `safety.budget_usd` | `2.0` | spend ceiling for a run, across all its repeats |
| `safety.allow_destructive` | `false` | don't ask before irreversible actions |
| `safety.check_sends` | `true` | have `llm.model_small` check every send against the policy before it goes |
| `run.max_steps` | `60` | step budget for a run, and for an MCP session |
| `run.max_wall_clock_s` | `1800` | working-time budget in seconds for a run, and for an MCP session |
| `run.always_screenshot` | `false` | take a screenshot every step |
| `run.never_screenshot` | `false` | never take a screenshot |
| `skills.learn_after_run` | `true` | update the app's skill after every run |
| `watch.policies_dir` | `policies` | the folder policies are found in by name |
| `watch.interval_s` | `45` | seconds between looks while nothing changes |
| `watch.max_steps` | `25` | step budget for one watch pass |
| `watch.draft` | `false` | draft mode for every watch and MCP session: sends are written, never sent |
| `notify.telegram_token` | `""` | your Telegram bot's token |
| `notify.telegram_chat_id` | `""` | your Telegram chat id |

Useful environment variables: `FIREWORKS_API_KEY` (or whichever one
`llm.api_key_env` names), `ANDROID_SERIAL`, `ADBAGENT_MODEL`,
`ADBAGENT_MAX_STEPS`, `ADBAGENT_BUDGET_USD`, `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID`.

## Troubleshooting

**`doctor` says no device.** Check the cable, and accept the *Allow USB
debugging?* prompt on the phone. Over Wi-Fi, pair again: the connection port
changes each time Wireless debugging is switched on.

**"The phone is in use".** Something else is driving it — a run, a watch or an
MCP session — and the message says what, and since when. Wait for it, stop it,
or let it go quiet: an idle watch or session lets go of the phone after 5
minutes.

**The MCP server doesn't connect.** Run its exact command in a terminal (see
[step 1](#1-get-the-command-right)). Use full paths, double the backslashes in
JSON on Windows, and restart the client after editing its config. If the client
gives up while the server starts, raise its startup timeout.

**"approval … was declined: this client cannot ask the person".** Your agent
can't show adbagent's questions; [What needs your OK](#what-needs-your-ok) lists
the flags that decide instead.

**Changes to adbagent don't reach your agent.** The agent keeps the server it
started. Restart the server: reconnect it from `/mcp` in Claude Code, start a
new session in Codex, choose **Restart** on it in Antigravity, or restart the
app. Policy edits need no restart.

**"no API key".** `run`, `watch`, skill generation and the web UI's runs need one:
set `FIREWORKS_API_KEY`, or `llm.api_key` in `config.json`. MCP sessions don't.

**Irreversible steps are refused in the web UI.** A run started in the browser
can't stop to ask, so tick **allow destructive** to let them go ahead.

**A run gave up going in circles.** `adbagent report` shows what it tried.
Rephrase the goal, or add what it was missing to the app's skill, then continue
with `--resume`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests need no phone and no API key: `tests/fake.py` is a scripted Android
app that emits real uiautomator XML, so the whole loop runs in seconds.
`tests/test_readme.py` checks this README: every `adbagent` command in it must
parse, and the defaults in the configuration table must be the real ones.
