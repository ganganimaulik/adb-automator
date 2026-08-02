"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class Out:
    """Terminal output. Colour when it is a terminal, plain when piped."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.colour = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def dim(self, text: str) -> str:
        return self._c("2", text)

    def bold(self, text: str) -> str:
        return self._c("1", text)

    def green(self, text: str) -> str:
        return self._c("32", text)

    def yellow(self, text: str) -> str:
        return self._c("33", text)

    def red(self, text: str) -> str:
        return self._c("31", text)

    def cyan(self, text: str) -> str:
        return self._c("36", text)

    def say(self, text: str = "") -> None:
        if not self.quiet:
            print(text)

    def ok(self, text: str) -> None:
        self.say(f"  {self.green('OK')}    {text}")

    def warn(self, text: str) -> None:
        self.say(f"  {self.yellow('WARN')}  {text}")

    def bad(self, text: str) -> None:
        self.say(f"  {self.red('FAIL')}  {text}")


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (
        logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    # These are chatty at DEBUG and drown out our own logs.
    for noisy in ("urllib3", "httpx", "httpcore", "openai", "adbutils", "PIL"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

#: CLI flag -> dotted config path.
OVERRIDES = {
    "model": "llm.model",
    "model_small": "llm.model_small",
    "model_image": "llm.model_image",
    "provider": "llm.provider",
    "rpm": "llm.rpm",
    "max_tokens": "llm.max_tokens",
    "device": "device.serial",
    "db": "memory.db",
    "budget_usd": "safety.budget_usd",
    "max_steps": "run.max_steps",
    "artifacts_dir": "run.artifacts_dir",
    "dry_run": "run.dry_run",
    "always_screenshot": "run.always_screenshot",
    "never_screenshot": "run.never_screenshot",
    "allow_destructive": "safety.allow_destructive",
    "unattended": "safety.unattended",
    "no_cache": "memory.enabled",
}


def build_config(args: argparse.Namespace):
    from .config import load_config

    overrides: Dict[str, Any] = {}
    for flag, dotted in OVERRIDES.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        if flag == "no_cache":
            overrides[dotted] = not value
        else:
            overrides[dotted] = value

    loaded = load_config(getattr(args, "config", None), overrides)
    for warning in loaded.warnings:
        print(f"  config: {warning}", file=sys.stderr)

    # Only the explore command uses --app to pin to a single package.
    app = getattr(args, "app", None)
    if app:
        loaded.config.safety.package_allowlist = [app]

    return loaded.config


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    from . import device as devmod

    out = Out()
    problems = 0
    out.say(out.bold(f"adbagent {__version__}"))
    out.say()

    out.say(out.bold("Environment"))
    out.ok(f"python {sys.version.split()[0]}")
    for name in ("uiautomator2", "adbutils", "openai", "pydantic", "PIL", "lxml"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "")
            if name == "uiautomator2":
                from uiautomator2.version import __version__ as v
                version = v
            out.ok(f"{name} {version}")
        except Exception as exc:  # noqa: BLE001
            out.bad(f"{name}: {exc}")
            problems += 1

    out.say()
    out.say(out.bold("adb"))
    try:
        path = devmod.adb_path()
        out.ok(f"adb at {path}")
    except Exception as exc:  # noqa: BLE001
        out.bad(f"adb not found: {exc}")
        return 1

    try:
        devices = devmod.list_devices()
    except Exception as exc:  # noqa: BLE001
        out.bad(f"could not list devices: {exc}")
        devices = []
        problems += 1

    if devices:
        for d in devices:
            out.ok(f"device {d.serial}")
    else:
        out.warn("no device attached")
        candidates = devmod.mdns_candidates()
        if candidates:
            out.say(f"        wireless debugging seen at: {', '.join(candidates)}")
            out.say("        try: adbagent pair <ip:pairing-port>")
        else:
            out.say("        plug in over USB, or enable Wireless debugging and run")
            out.say("        adbagent pair <ip:pairing-port>")
        problems += 1

    out.say()
    out.say(out.bold("LLM"))
    cfg = build_config(args)
    key = cfg.api_key()
    if key:
        out.ok(f"${cfg.llm.api_key_env} is set ({len(key)} chars)")
    else:
        out.bad(f"${cfg.llm.api_key_env} is not set")
        problems += 1
    if cfg.llm.model:
        out.ok(f"model {cfg.llm.model}")
        if cfg.llm.model_small:
            out.ok(f"small model {cfg.llm.model_small}")
        if cfg.llm.model_image:
            out.ok(f"vision model {cfg.llm.model_image}")
    else:
        out.warn("no model chosen -- run: adbagent models")
        problems += 1

    out.say()
    out.say(out.bold("Memory"))
    out.ok(f"database {cfg.db_path}")
    if cfg.db_path.exists():
        from .memory import Memory
        with Memory(cfg) as mem:
            summary = mem.stats_summary()
        out.ok(f"{summary['entries']} learned steps across {summary['apps']} app(s)")
        if summary["by_state"]:
            out.say(f"        {summary['by_state']}")
    else:
        out.say(out.dim("        (not created yet -- it appears on the first run)"))

    out.say()
    if problems:
        out.say(out.yellow(f"{problems} thing(s) need attention."))
    else:
        out.say(out.green("Ready."))
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# devices / pair
# ---------------------------------------------------------------------------

def cmd_devices(args) -> int:
    from . import device as devmod

    out = Out()
    devices = devmod.list_devices()
    if not devices:
        out.say("No devices. Connect over USB, or use `adbagent pair`.")
        for addr in devmod.mdns_candidates():
            out.say(f"  wireless debugging advertised at {addr}")
        return 1
    for d in devices:
        try:
            model = d.prop.model
            release = d.getprop("ro.build.version.release")
            out.say(f"  {d.serial:<28} {model} (Android {release})")
        except Exception:  # noqa: BLE001
            out.say(f"  {d.serial}")
    return 0


def cmd_pair(args) -> int:
    from . import device as devmod
    from .config import save_device_serial

    out = Out()
    code = args.code or input("  Pairing code shown on the phone: ").strip()
    try:
        out.say(devmod.pair(args.address, code))
    except Exception as exc:  # noqa: BLE001
        out.bad(str(exc))
        return 1

    connect_to = args.connect
    if not connect_to:
        candidates = devmod.mdns_candidates()
        if candidates:
            connect_to = candidates[0]
            out.say(f"  discovered {connect_to} over mDNS")
        else:
            out.warn("paired, but I do not know the connect port.")
            out.say("  It is the ip:port on the Wireless debugging screen itself,")
            out.say("  which is NOT the pairing port. Then run:")
            out.say("      adbagent pair --connect <ip:port> ...")
            return 0
    try:
        out.say(devmod.connect_wireless(connect_to))
    except Exception as exc:  # noqa: BLE001
        out.bad(str(exc))
        return 1

    # Persist the serial so subsequent commands find the device automatically.
    cfg_path = save_device_serial(connect_to, getattr(args, "config", None))
    out.say(f"  saved device serial {out.bold(connect_to)} to {cfg_path}")

    out.say(out.green("  Connected. Note the port changes whenever Wireless "
                      "debugging is toggled."))
    return 0


def cmd_pair_qr(args) -> int:
    from . import device as devmod
    from .config import save_device_serial

    out = Out()
    out.say(out.bold("  Scan this QR code on your phone:"))
    out.say("  Phone → Developer Options → Wireless Debugging → "
            "Pair device with QR code")
    out.say()

    try:
        serial = devmod.pair_qr(timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001
        out.bad(str(exc))
        return 1

    # Persist the serial so subsequent commands find the device automatically.
    cfg_path = save_device_serial(serial, getattr(args, "config", None))
    out.say(f"  saved device serial {out.bold(serial)} to {cfg_path}")

    out.say(out.green("  Connected and saved."))
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def cmd_models(args) -> int:
    from .llm import PROVIDERS, list_models

    out = Out()
    cfg = build_config(args)
    provider = PROVIDERS.get(cfg.llm.provider)
    if provider is None:
        out.bad(f"unknown provider {cfg.llm.provider}")
        return 1
    key = cfg.api_key()
    if not key:
        out.bad(f"${cfg.llm.api_key_env} is not set")
        return 1

    models = list_models(provider, key)
    if args.vision:
        models = [m for m in models if m.vision]
    if args.search:
        needle = args.search.lower()
        models = [m for m in models if needle in m.id.lower()
                  or needle in m.display_name.lower()]

    out.say(f"  {'MODEL':<58} {'CTX':>6}  CAPABILITIES")
    for model in models:
        out.say("  " + model.row())
    out.say()
    out.say(f"  {len(models)} model(s). Choose one with --model, or put it in "
            f"config.json under llm.model.")
    return 0


# ---------------------------------------------------------------------------
# auto-pair when no device is connected
# ---------------------------------------------------------------------------

def _ensure_device(args, cfg, out: Out) -> None:
    """If no device is reachable, run QR pairing automatically.

    Modifies ``cfg.device.serial`` in place so the caller's ``Device(cfg)``
    picks up the newly-paired serial.
    """
    from . import device as devmod
    from .config import save_device_serial

    serial = getattr(args, "device", None) or cfg.device.serial

    # USB serial — nothing we can do but hand it to Device and let it error.
    if serial and ":" not in serial:
        return

    # Wireless serial in config — try reconnecting.
    if serial and ":" in serial:
        try:
            devmod.connect_wireless(serial, timeout=5)
            return  # still reachable
        except Exception:  # noqa: BLE001
            out.warn(f"could not reach {serial}, looking for alternatives…")
            # Fall through — do NOT check list_devices() here because the
            # stale serial still appears in the list as "offline".

    else:
        # No serial at all — maybe a USB device is already plugged in.
        if devmod.list_devices():
            return

    # Try mDNS discovery first (fast, no user interaction).
    candidates = devmod.mdns_candidates()
    if candidates:
        addr = candidates[0]
        try:
            devmod.connect_wireless(addr)
            cfg.device.serial = addr
            out.ok(f"auto-connected to {addr} (mDNS)")
            save_device_serial(addr, getattr(args, "config", None))
            return
        except Exception:  # noqa: BLE001
            pass

    # Last resort: interactive QR pairing.
    out.say()
    out.say(out.yellow("  No device connected."))
    out.say(out.bold("  Starting QR pairing — scan with your phone:"))
    out.say("  Phone → Developer Options → Wireless Debugging → "
            "Pair device with QR code")
    out.say()

    serial = devmod.pair_qr(timeout=120)
    cfg.device.serial = serial
    cfg_path = save_device_serial(serial, getattr(args, "config", None))
    out.ok(f"paired and connected to {serial}")
    out.say(f"  saved to {cfg_path}")
    out.say()


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------

def cmd_dump(args) -> int:
    from .device import Device
    from .screen import element_detail, render

    out = Out()
    cfg = build_config(args)
    _ensure_device(args, cfg, out)
    with Device(cfg, args.device or "") as dev:
        screen = dev.observe()
        if args.detail is not None:
            element = screen.by_index(args.detail)
            if element is None:
                out.bad(f"no element #{args.detail}")
                return 1
            out.say(element_detail(element))
            return 0
        if args.raw:
            out.say(screen.xml)
            return 0
        rendered = render(screen)
        out.say(rendered)
        out.say()
        out.say(out.dim(
            f"  raw dump {len(screen.xml):,} chars -> {len(rendered):,} chars "
            f"(~{len(rendered) // 4:,} tokens), "
            f"{len(screen.elements)} elements, {len(screen.actionable)} actionable"))
        out.say(out.dim(f"  skeleton {screen.skeleton_id}  exact {screen.exact_id}"))
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _live_reporter(out: Out):
    def report(kind: str, **kw) -> None:
        if kind != "step":
            return
        state = kw["state"]
        action = kw["action"]
        source = kw["source"]
        tag = out.green("CACHE") if source == "cache" else out.cyan(" LLM ")
        out.say(f"  {state.step:>3} [{tag}] {action.describe()}")
        if source == "llm":
            if getattr(action, "observation", None):
                out.say(out.dim(f"        Obs:       {action.observation}"))
            if getattr(action, "reasoning", None):
                out.say(out.dim(f"        Reasoning: {action.reasoning}"))
    return report


def cmd_run(args) -> int:
    from .agent import Agent, Oracle
    from .device import Device
    from .llm import LLMClient
    from .memory import Memory

    out = Out()
    cfg = build_config(args)
    if not cfg.llm.model:
        out.bad("no model chosen. Run `adbagent models` and pass --model.")
        return 1

    _ensure_device(args, cfg, out)

    oracle = Oracle(shell=args.assert_shell or "", equals=args.assert_equals or "",
                    text=args.assert_text or "")

    repeats = args.repeat
    infinite = isinstance(repeats, str) and repeats == "inf"
    total = 0 if infinite else int(repeats or 1)

    exit_code = 0
    iteration = 0
    with Device(cfg, args.device or "") as dev, Memory(cfg) as mem:
        # One client for the whole session, so --budget-usd bounds the *session*
        # rather than resetting on every iteration (which would make it useless
        # with --repeat inf).
        llm = LLMClient(cfg, run_id=f"run-{int(time.time())}")
        while infinite or iteration < total:
            iteration += 1
            llm.run_id = f"run-{int(time.time())}-{iteration}"
            spent_before = llm.ledger.total_usd
            agent = Agent(dev, mem, llm, cfg, oracle=oracle,
                          on_event=_live_reporter(out))
            if infinite or total > 1:
                out.say(out.bold(f"\n  iteration {iteration}"))
            started = time.monotonic()
            outcome, state = agent.run(args.goal)
            elapsed = time.monotonic() - started

            colour = out.green if outcome == "success" else (
                out.yellow if outcome == "needs_user" else out.red)
            spent = llm.ledger.total_usd - spent_before
            tilde = "~" if llm.ledger.estimated else ""
            out.say()
            if state.scratchpad:
                out.say()
                out.say(out.bold("  ── Collected Data ──"))
                out.say()
                for chunk in state.scratchpad:
                    out.say(f"  {chunk}")
                out.say()
            out.say(f"  {colour(outcome.upper())}  "
                    f"{state.step} steps, {state.llm_calls} LLM calls "
                    f"({state.cache_rate():.0%} from cache), "
                    f"{tilde}${spent:.4f}, {elapsed:.1f}s")
            agreement = state.audit_agreement()
            if agreement is not None:
                note = f"  cache audited {state.audits}x, model agreed {agreement:.0%}"
                out.say(out.dim(note) if agreement == 1.0 else out.yellow(note))
            if outcome != "success":
                exit_code = 1
            if outcome in ("aborted", "needs_user"):
                break
        if iteration > 1:
            out.say(out.dim(f"  session total: {tilde}${llm.ledger.total_usd:.4f} "
                            f"over {iteration} iteration(s)"))
    return exit_code


# ---------------------------------------------------------------------------
# explore
# ---------------------------------------------------------------------------

def cmd_explore(args) -> int:
    from .agent import explore
    from .device import Device
    from .llm import LLMClient
    from .memory import Memory

    out = Out()
    cfg = build_config(args)
    if not cfg.llm.model:
        out.bad("no model chosen. Run `adbagent models` and pass --model.")
        return 1

    out.say(out.dim("  Explore is read-only: it navigates and scrolls, but never "
                    "types and never presses anything that changes state."))
    _ensure_device(args, cfg, out)
    with Device(cfg, args.device or "") as dev, Memory(cfg) as mem:
        llm = LLMClient(cfg, run_id=f"explore-{int(time.time())}")
        result = explore(dev, mem, llm, cfg, package=args.app or "",
                         max_screens=args.max_screens)
    out.say()
    out.say(f"  Visited {result['screens']} screen(s) in {result['steps']} steps, "
            f"{result['llm_calls']} LLM calls, ${result['usd']:.4f}")
    if result["blocked"]:
        out.say(out.yellow(f"  Skipped {len(result['blocked'])} action(s) that "
                           f"would have changed something:"))
        for item in result["blocked"][:10]:
            out.say(f"    - {item}")
    return 0


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def cmd_memory(args) -> int:
    from .memory import Memory

    out = Out()
    cfg = build_config(args)
    with Memory(cfg) as mem:
        if args.memory_command == "ls":
            entries = mem.entries(app_key=args.app or "", state=args.state or "")
            if not entries:
                out.say("  (nothing learned yet)")
                return 0
            out.say(f"  {'ID':>5}  {'STATE':<12} {'V':>2} {'OK':>4} {'BAD':>4}  "
                    f"{'APP':<32} ACTION")
            for e in entries:
                out.say(f"  {e.id:>5}  {e.state:<12} {e.version:>2} "
                        f"{e.stats.n_success:>4.0f} {e.stats.n_failure:>4.0f}  "
                        f"{e.app_key[:32]:<32} {e.action.describe()}")
            out.say()
            out.say(f"  {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")
            return 0

        if args.memory_command == "show":
            entry = mem.get(args.id)
            if entry is None:
                out.bad(f"no entry {args.id}")
                return 1
            out.say(json.dumps({
                "id": entry.id, "app": entry.app_key, "state": entry.state,
                "version": entry.version, "skeleton": entry.skeleton_id,
                "intent": entry.intent_id, "visit": entry.visit_ordinal,
                "action": entry.action.model_dump(),
                "anchor": entry.anchor.model_dump() if entry.anchor else None,
                "postcondition": (entry.postcondition.model_dump()
                                  if entry.postcondition else None),
                "required_tokens": entry.required_tokens,
                "forbidden_tokens": entry.forbidden_tokens,
                "next_screen": entry.next_skeleton_id,
                "alt_successors": entry.alt_successors,
                "successes": entry.stats.n_success,
                "failures": entry.stats.n_failure,
                "wilson": round(entry.stats.wilson(), 3),
            }, indent=2, default=str))
            out.say()
            out.say("  recent outcomes:")
            for row in mem.db.execute(
                    "SELECT grade, reason, match_distance, anchor_score, at "
                    "FROM entry_outcome WHERE entry_id=? ORDER BY at DESC LIMIT 10",
                    (entry.id,)):
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["at"]))
                out.say(f"    {when}  {row['grade']:<10} d={row['match_distance']} "
                        f"score={row['anchor_score']:.2f}  {row['reason'] or ''}")
            return 0

        if args.memory_command == "forget":
            n = mem.forget(entry_id=args.id or 0, app_key=args.app or "",
                           state=args.state or "")
            out.say(f"  forgot {n} entr{'y' if n == 1 else 'ies'}")
            return 0

        if args.memory_command == "gc":
            n = mem.gc()
            out.say(f"  retired {n} entr{'y' if n == 1 else 'ies'}")
            return 0

        summary = mem.stats_summary()
        out.say(json.dumps(summary, indent=2))
        return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args) -> int:
    out = Out()
    path = Path(args.run).expanduser()
    events_file = path / "events.jsonl" if path.is_dir() else path
    if not events_file.exists():
        out.bad(f"no events at {events_file}")
        return 1

    events = [json.loads(line) for line in
              events_file.read_text().splitlines() if line.strip()]
    start = next((e for e in events if e["kind"] == "run_start"), {})
    end = next((e for e in events if e["kind"] == "run_end"), {})

    out.say(out.bold(f"  goal: {start.get('goal', '?')}"))
    out.say(f"  model: {start.get('model', '?')}")
    out.say()
    for event in events:
        if event["kind"] == "decide":
            action = event.get("action", {})
            tag = "CACHE" if event.get("source") == "cache" else " LLM "
            shot = " +img" if event.get("screenshot") else ""
            out.say(f"  {event.get('step', 0):>3} [{tag}]{shot} "
                    f"{action.get('action')} {action.get('target') or ''}")
            if event.get("source") == "llm":
                if action.get("observation"):
                    out.say(out.dim(f"        Obs:       {action.get('observation')}"))
                if action.get("reasoning"):
                    out.say(out.dim(f"        Reasoning: {action.get('reasoning')}"))
        elif event["kind"] == "verify":
            out.say(f"      -> {event.get('grade')} {event.get('reason') or ''}")
        elif event["kind"] in ("dismiss", "refused", "loop_break", "sensitive",
                               "judge", "error", "gave_up"):
            out.say(f"      [{event['kind']}] "
                    + " ".join(f"{k}={v}" for k, v in event.items()
                               if k not in ("t", "kind")))
    out.say()
    if end:
        out.say(f"  {end.get('outcome', '?').upper()}: {end.get('steps')} steps, "
                f"{end.get('llm_calls')} LLM calls, "
                f"{end.get('cache_hits')} cache hits, ${end.get('usd', 0):.4f}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", help="path to config.json")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--provider", help="llm provider (default fireworks)")
    parser.add_argument("--model", help="model id")
    parser.add_argument("--model-small", dest="model_small",
                        help="cheaper model for judging and repair")
    parser.add_argument("--model-image", dest="model_image",
                        help="model for vision calls with screenshots")
    parser.add_argument("--rpm", type=int, help="client-side request throttle")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int,
                        help="max completion tokens for LLM calls")
    parser.add_argument("--db", help="path to the memory database")


def _add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-d", "--device", help="adb serial (or ip:port)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adbagent",
        description="A self-improving Android automation agent.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the environment")
    _add_common(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("devices", help="list attached devices")
    _add_common(p)
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("pair", help="pair with a phone over Wi-Fi")
    p.add_argument("address", help="ip:port from 'Pair device with pairing code'")
    p.add_argument("--code", help="the 6-digit code (prompted if omitted)")
    p.add_argument("--connect", help="ip:port from the Wireless debugging screen")
    _add_common(p)
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("pair-qr",
                       help="pair by displaying a QR code to scan on the phone")
    p.add_argument("--timeout", type=int, default=120,
                   help="seconds to wait for the phone to scan (default 120)")
    _add_common(p)
    p.set_defaults(func=cmd_pair_qr)

    p = sub.add_parser("models", help="list models you can choose from")
    p.add_argument("--vision", action="store_true", help="only multimodal models")
    p.add_argument("--search", help="filter by substring")
    _add_common(p)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("dump", help="show what the model would see")
    p.add_argument("--raw", action="store_true", help="print the raw XML instead")
    p.add_argument("--detail", type=int, metavar="N",
                   help="print every attribute of element #N")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("run", help="pursue a goal on the device")
    p.add_argument("goal", help="what to accomplish, in plain language")
    p.add_argument("--repeat", default="1",
                   help="how many times to repeat the goal ('inf' for forever)")
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--budget-usd", dest="budget_usd", type=float)
    p.add_argument("--assert-shell", dest="assert_shell",
                   help="shell command whose output proves success")
    p.add_argument("--assert-equals", dest="assert_equals",
                   help="expected output of --assert-shell")
    p.add_argument("--assert-text", dest="assert_text",
                   help="text that must be on screen for success")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="decide every step but execute nothing")
    p.add_argument("--no-cache", dest="no_cache", action="store_true",
                   help="ignore learned steps (the slow, expensive baseline)")
    p.add_argument("--always-screenshot", dest="always_screenshot",
                   action="store_true")
    p.add_argument("--never-screenshot", dest="never_screenshot",
                   action="store_true")
    p.add_argument("--allow-destructive", dest="allow_destructive",
                   action="store_true",
                   help="do not ask before irreversible actions")
    p.add_argument("--unattended", action="store_true",
                   help="never prompt; refuse instead of asking")
    p.add_argument("--artifacts-dir", dest="artifacts_dir")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("explore", help="learn an app's layout, read-only")
    p.add_argument("--app", help="package to explore")
    p.add_argument("--max-screens", dest="max_screens", type=int, default=40)
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--budget-usd", dest="budget_usd", type=float)
    p.add_argument("--unattended", action="store_true")
    p.add_argument("--allow-destructive", dest="allow_destructive",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--artifacts-dir", dest="artifacts_dir")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("memory", help="inspect and curate what has been learned")
    msub = p.add_subparsers(dest="memory_command")
    for name, help_text in (("ls", "list learned steps"),
                            ("show", "everything about one entry"),
                            ("forget", "delete entries"),
                            ("gc", "retire stale and quarantined entries"),
                            ("stats", "summary")):
        sp = msub.add_parser(name, help=help_text)
        if name in ("show", "forget"):
            sp.add_argument("id", nargs="?", type=int)
        if name in ("ls", "forget"):
            sp.add_argument("--app")
            sp.add_argument("--state",
                            choices=["probation", "active", "trusted",
                                     "quarantined", "retired"])
        _add_common(sp)
        sp.set_defaults(func=cmd_memory, memory_command=name)
    _add_common(p)
    p.set_defaults(func=cmd_memory, memory_command="stats")

    p = sub.add_parser("report", help="summarise a recorded run")
    p.add_argument("run", help="path to runs/<id> or its events.jsonl")
    _add_common(p)
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", 0))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "verbose", 0) >= 2:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
