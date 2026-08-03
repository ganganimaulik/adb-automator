# adbagent

An Android automation agent. Give it a goal in plain language and it drives a real phone until the goal is met.

```
$ adbagent run "turn on airplane mode" --app com.android.settings
    1 [ LLM ] tap #12 "Network & internet"
    2 [ LLM ] tap #7 "Airplane mode"
  SUCCESS  2 steps, 3 LLM calls, $0.0091, 11.4s
```

## Features

- **Safety First:** Credentials (passwords/PINs) and payment screens are never logged. Destructive actions require explicit confirmation.
- **Goal Verification:** Verifies success with natural language checks or deterministic shell/text assertions.
- **App Exploration:** Explores app UI layouts safely without modifying state.
- **Run Reports:** Generate readable execution traces and reports for completed runs.
- **Carousel Sweeps:** Pages through galleries in code once the model has chosen to, replacing a reasoning turn per photo with a one-line vision read.
- **Replayable Runs:** Re-issues a recorded run's decisions against a changed prompt or model and diffs the results.

## Install

Requires **Python 3.10+** and **Android platform tools**.

```bash
pip install -e .
```

Verify your environment setup:

```bash
adbagent doctor
```

## Connecting a Phone

- **USB:** Connect device via USB with USB Debugging enabled.
- **Wi-Fi (Android 11+):**
  1. Open **Developer options → Wireless debugging → Pair device with pairing code**.
  2. Run pairing command:
     ```bash
     adbagent pair 192.168.1.50:37115
     ```

## Choosing a Model

```bash
export FIREWORKS_API_KEY=fw_...
adbagent models --vision
```

Specify your chosen model with `--model`, or set it in `config.json` (copy from `config.example.json`).

## Usage

```bash
# Preview actions without executing on device
adbagent run "turn on dark mode" --app com.android.settings --dry-run

# Run an action
adbagent run "turn on dark mode" --app com.android.settings

# Safe read-only exploration of app layouts
adbagent explore --app com.android.settings
```

### Assertions

Add instant, mechanical checks to confirm goal completion:

```bash
# Shell command assertion
adbagent run "turn on airplane mode" \
  --assert-shell 'settings get global airplane_mode_on' --assert-equals 1

# On-screen text assertion
adbagent run "open the Wi-Fi screen" --assert-text "Forget network"
```

## Reports

```bash
# Generate run reports, with a breakdown of where the wall clock went
adbagent report runs/<id>
```

Almost all of a run's wall clock is the model thinking — 26s median per step
against 3.4s to act and verify — so the report ends with the numbers that
actually move it:

```
  ── Cost of thinking ──
  latency/step     26.2s median     96.3s p90       5186s total
  prompt tokens     5500 median   698500 total       56% served from cache
  output tokens     4400 median   558800 total
  of which think    4200 median   533400 total       95% of output
```

## Replay

Every run records the messages it sent and the action each one produced, which
makes it a regression set. `adbagent replay` re-issues them and diffs the answers,
so a change to `prompts.py` or to the reasoning budget can be measured rather
than guessed at.

```bash
adbagent replay                              # the latest run, verbatim
adbagent replay runs/<id> --rebuild-system   # test an edit to prompts.py
adbagent replay runs/<id> --limit 20 --json
```

Verbatim holds the prompt fixed and varies the model or decoder; `--rebuild-system`
swaps in the system prompt `prompts.py` builds today and leaves the run's own
observations alone. Divergence is not treated as failure — each case carries the
grade its recorded action earned, so leaving a `no_change` step behind is reported
apart from changing a step that had worked, and only the latter sets the exit code.

## Tuning

Speed knobs, in `config.json`:

| setting | default | what it does |
|---|---|---|
| `run.pager_sweep` | `true` | After the model pages through a carousel and the item verifiably moves, keep paging in code — a vision read per item instead of a reasoning turn. Stops at either end of the set, on a hidden caption, on a dialog, or on an app switch. |
| `run.pager_sweep_max` | `12` | Items per sweep before control returns to the model. |
| `llm.vision_in_decider` | `false` | Set when `llm.model` itself accepts images: the screenshot then goes straight to the deciding call instead of being described first by `llm.model_image`, which is one round trip per screenshot turn instead of two. Leave off for a text-only model — it would fail the whole call. |

## Development

```bash
pip install -e ".[dev]"
pytest
```
