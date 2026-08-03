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
# Generate run reports
adbagent report runs/<id>
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
