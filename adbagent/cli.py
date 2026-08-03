"""Command line interface."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__, scratchpad

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

log = logging.getLogger("adbagent.cli")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _notes_text(notes: Any) -> str:
    """One line for the records an action wrote.

    `notes` is a list of ``{key, value}`` now, from either a live `Note` or a
    recorded dict, and printing the list itself puts a pydantic repr on the
    user's terminal.
    """
    pairs = scratchpad.as_records(notes)
    return "; ".join(f"{key}: {value}" if value else str(key)
                     for key, value in pairs) or ""


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

    def write(self, text: str = "") -> None:
        if not self.quiet:
            sys.stdout.write(text)
            sys.stdout.flush()

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
    "model_skill": "llm.model_skill",
    "provider": "llm.provider",
    "service_tier": "llm.service_tier",
    "rpm": "llm.rpm",
    "max_tokens": "llm.max_tokens",
    "device": "device.serial",
    "db": "memory.db",
    "budget_usd": "safety.budget_usd",
    "max_steps": "run.max_steps",
    "artifacts_dir": "run.artifacts_dir",
    "skills_dir": "skills.skills_dir",
    "dry_run": "run.dry_run",
    "always_screenshot": "run.always_screenshot",
    "never_screenshot": "run.never_screenshot",
    "allow_destructive": "safety.allow_destructive",
    "unattended": "safety.unattended",
}


def build_config(args: argparse.Namespace):
    from .config import load_config

    overrides: Dict[str, Any] = {}
    for flag, dotted in OVERRIDES.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        overrides[dotted] = value

    loaded = load_config(getattr(args, "config", None), overrides)
    for warning in loaded.warnings:
        print(f"  config: {warning}", file=sys.stderr)

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
        if cfg.llm.service_tier:
            out.ok(f"service tier {cfg.llm.service_tier}")
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
        out.ok("database exists")
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

def _tail_rows(text: str, max_rows: int, width: int) -> tuple:
    """The newest `max_rows` display rows of `text`, and whether older rows were cut.

    Rows, not lines: reasoning usually streams as a handful of very long
    paragraphs, so counting newlines lets one logical line overrun the panel.
    Once the live region is taller than the terminal, rich keeps its *top* and
    the visible text stops moving -- which reads as the stream having died.
    Hard-wrapping here also pins the panel height, so the maths below holds.
    """
    if max_rows <= 0:
        return [], bool(text)
    w = max(1, width)
    lines = text.splitlines() or [""]
    rows: List[str] = []
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        wrapped = [line[j:j + w] for j in range(0, len(line), w)] or [""]
        room = max_rows - len(rows)
        if len(wrapped) > room:
            rows[:0] = wrapped[len(wrapped) - room:] if room else []
            return rows, True
        rows[:0] = wrapped
    return rows, False


def _row_count(text: str, width: int) -> int:
    """How many display rows `text` needs at `width` columns."""
    w = max(1, width)
    return sum(max(1, (len(line) + w - 1) // w)
               for line in (text.splitlines() or [""]))


def _stream_panel(thinking: str, content: str, width: int, height: int) -> Any:
    """The LLM stream panel, tailed to fit a `width` x `height` terminal."""
    # A panel spends 2 rows on borders and 4 columns on borders plus padding.
    # Leave one row spare so the live region never outgrows the screen.
    body_rows = max(3, height - 3)
    text_width = max(20, width - 4)

    thinking = thinking.strip()
    content = content.strip()

    # (label, label style, text, body style, row budget)
    sections: List[tuple] = []
    if thinking and content:
        avail = max(2, body_rows - 3)  # two headers plus a blank separator
        t_want = _row_count(thinking, text_width)
        c_want = _row_count(content, text_width)
        if t_want + c_want <= avail:
            t_budget = t_want
        elif t_want <= avail // 2:
            t_budget = t_want  # thinking fits; the response can have the rest
        elif c_want < avail - avail // 2:
            t_budget = avail - c_want  # response fits; thinking takes the rest
        else:
            t_budget = max(1, avail // 2)  # both are long, so split evenly
        sections.append(("[Thinking]", "dim italic", thinking, "dim italic", t_budget))
        sections.append(("[Response]", "cyan bold", content, "dim", avail - t_budget))
    elif thinking:
        sections.append(("[Thinking]", "dim italic", thinking, "dim italic",
                         max(1, body_rows - 1)))
    elif content:
        sections.append(("[Response]", "cyan bold", content, "dim",
                         max(1, body_rows - 1)))

    if not sections:
        body: Any = Text("thinking...", style="dim")
    else:
        # Built as Text, never markup: model output is full of bracketed
        # fragments, and rich either swallows them as styles or raises
        # MarkupError from inside the refresh thread, killing the panel.
        body = Text(no_wrap=True, overflow="crop", end="")
        for i, (label, label_style, text, body_style, budget) in enumerate(sections):
            if i:
                body.append("\n\n")
            rows, cut = _tail_rows(text, budget, text_width)
            body.append(label, style=label_style)
            if cut:
                body.append("  (earlier output scrolled off)", style="dim")
            body.append("\n")
            body.append("\n".join(rows), style=body_style)

    return Panel(body, title="[cyan]LLM Stream[/cyan]", border_style="dim", expand=False)


#: Live panel frame rate. Chunks arriving between frames only accumulate text;
#: rebuilding the panel for a frame that is never drawn is wasted work on the
#: thread reading the LLM stream.
_STREAM_FPS = 12


def _live_reporter(out: Out, max_steps: Optional[int] = None):
    stream_state: Dict[str, Any] = {
        "active": False,
        "type": None,
        "thinking_text": "",
        "content_text": "",
        "live": None,
        "console": None,
        "using_rich": False,
        "rich_broken": False,
        "last_frame": 0.0,
    }

    def _render_live_panel() -> Any:
        if not _HAS_RICH:
            return ""
        console = stream_state.get("console")
        # Read the size every render so a mid-stream resize is picked up.
        width = getattr(console, "width", None) or 80
        height = getattr(console, "height", None) or 24
        return _stream_panel(stream_state["thinking_text"],
                             stream_state["content_text"], width, height)

    def report(kind: str, **kw) -> None:
        if kind != "llm_stream" and stream_state["active"]:
            if stream_state["live"] is not None:
                try:
                    stream_state["live"].stop()
                except Exception:
                    pass
                stream_state["live"] = None
                stream_state["console"] = None
            elif not stream_state.get("using_rich"):
                out.write("\n")
            stream_state["active"] = False
            stream_state["type"] = None
            stream_state["thinking_text"] = ""
            stream_state["content_text"] = ""
            stream_state["using_rich"] = False

        step = kw.get("step")
        if step is None and "state" in kw:
            step = kw["state"].step
        step_hdr = f"  [{step:>2}/{max_steps}]" if (step and max_steps) else (f"  [{step:>2}]" if step else "     ")

        if kind == "perceive":
            elapsed = kw.get("elapsed", 0.0)
            out.say(f"{step_hdr} Perceiving screen... {out.dim(f'({elapsed:.2f}s)')}")

        elif kind == "llm_start":
            purpose = kw.get("purpose", "decide")
            model = kw.get("model", "")
            shot = " +img" if kw.get("screenshot") else ""
            if purpose == "judge":
                label = "LLM judge"
            elif purpose == "analyze_image":
                label = "LLM image analyzer"
            else:
                label = "LLM"
            out.say(out.cyan(f"        calling {label} ({model}{shot})..."))

        elif kind == "llm_stream":
            stream_type = kw.get("stream_type", "content")
            text = kw.get("text", "")
            if not text:
                return

            use_rich_live = (_HAS_RICH and sys.stdout.isatty() and not out.quiet
                             and not stream_state.get("rich_broken"))

            if use_rich_live:
                if stream_type == "thinking":
                    stream_state["thinking_text"] += text
                else:
                    stream_state["content_text"] += text

                stream_state["active"] = True
                stream_state["type"] = stream_type
                stream_state["using_rich"] = True

                # A panel that fails to render must not take the run with it:
                # give up on the live view and keep streaming as plain text.
                now = time.monotonic()
                try:
                    if stream_state["live"] is None:
                        console = Console()
                        stream_state["console"] = console
                        stream_state["live"] = Live(
                            _render_live_panel(),
                            console=console,
                            transient=True,
                            # Draw from this thread on our own clock. A rich
                            # refresh thread would redraw stale text and, if a
                            # render ever raised, die there unnoticed.
                            auto_refresh=False,
                            vertical_overflow="crop",
                        )
                        stream_state["live"].start(refresh=True)
                        stream_state["last_frame"] = now
                    elif now - stream_state["last_frame"] >= 1.0 / _STREAM_FPS:
                        stream_state["live"].update(_render_live_panel(), refresh=True)
                        stream_state["last_frame"] = now
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("live LLM panel failed (%s); falling back to plain output", exc)
                    with contextlib.suppress(Exception):
                        if stream_state["live"] is not None:
                            stream_state["live"].stop()
                    stream_state["live"] = None
                    stream_state["console"] = None
                    stream_state["using_rich"] = False
                    stream_state["active"] = False
                    stream_state["type"] = None
                    stream_state["rich_broken"] = True
            else:
                if stream_state["type"] != stream_type:
                    if stream_state["active"]:
                        out.write("\n")
                    prefix = "        [Thinking] " if stream_type == "thinking" else "        [Response] "
                    out.write(out.dim(prefix))
                    stream_state["type"] = stream_type
                    stream_state["active"] = True
                out.write(out.dim(text))

        elif kind == "llm_end":
            if stream_state["live"] is not None:
                try:
                    stream_state["live"].stop()
                except Exception:
                    pass
                stream_state["live"] = None
                stream_state["console"] = None
                stream_state["active"] = False
                stream_state["type"] = None
                stream_state["thinking_text"] = ""
                stream_state["content_text"] = ""
                stream_state["using_rich"] = False

            elapsed = kw.get("elapsed", 0.0)
            call = kw.get("call")
            purpose = kw.get("purpose", "decide")
            tokens_info = ""
            if call and getattr(call, "prompt_tokens", 0):
                tokens_info = f" ({call.prompt_tokens} prompt tokens, {call.completion_tokens} completion tokens)"
            if purpose == "judge":
                tag = "LLM judge responded"
            elif purpose == "analyze_image":
                tag = "LLM image analyzer responded"
            else:
                tag = "LLM responded"
            out.say(out.dim(f"        {tag} in {elapsed:.2f}s{tokens_info}"))

        elif kind == "image_analysis":
            result = kw.get("result", "")
            model = kw.get("model", "")
            m_str = f" ({model})" if model else ""
            out.say(out.dim(f"        Vision{m_str}: {result}"))

        elif kind == "step":
            action = kw["action"]
            screenshot = kw.get("screenshot", False)
            shot = " +img" if screenshot else ""
            conf = " (confidence: low)" if getattr(action, "confidence", None) == "low" else ""
            out.say(f"{step_hdr}{shot} {out.bold(action.describe())}{conf}")
            if getattr(action, "observation", None):
                out.say(out.dim(f"        Obs:       {action.observation}"))
            if getattr(action, "reasoning", None):
                out.say(out.dim(f"        Reasoning: {action.reasoning}"))
            if getattr(action, "progress", None):
                out.say(out.dim(f"        Progress:  {action.progress}"))
            if getattr(action, "notes", None):
                out.say(out.dim(f"        Notes:     {_notes_text(action.notes)}"))

        elif kind == "act_end":
            elapsed = kw.get("elapsed", 0.0)
            out.say(out.dim(f"        executed action in {elapsed:.2f}s"))

        elif kind == "sweep_step":
            label = kw.get("label") or "(caption hidden)"
            total = kw.get("total") or 0
            of = f"/{total}" if total else ""
            moved = "" if kw.get("moved") else "  (did not move)"
            out.say(f"{step_hdr} {out.cyan('sweep')} {kw.get('direction', '')} "
                    f"-> {label}  [read {kw.get('read_count', 0)}{of}]{moved}")

        elif kind == "item_reading":
            out.say(out.dim(f"        Read:      {kw.get('reading', '')}"))

        elif kind == "sweep_end":
            out.say(out.dim(
                f"        swept {kw.get('swept', 0)} item(s) "
                f"{kw.get('direction', '')}, read {kw.get('read', 0)} "
                f"— stopped because {kw.get('reason', 'it stopped')}"))

        elif kind == "settle_start":
            budget = kw.get("budget", 2.0)
            out.say(out.dim(f"        waiting for settle (budget max {budget:.1f}s)..."))

        elif kind == "verify_end":
            elapsed = kw.get("elapsed", 0.0)
            grade = kw.get("grade", "")
            reason = kw.get("reason", "")
            r_str = f": {reason}" if reason else ""
            out.say(out.dim(f"        settled & verified in {elapsed:.2f}s -> grade: {grade}{r_str}"))

        elif kind == "loop_warning":
            msg = kw.get("message", "")
            out.say(out.yellow(f"        [Loop Warning] {msg}"))

        elif kind == "safety_warning":
            msg = kw.get("message", "")
            out.say(out.yellow(f"        [Safety Warning] {msg}"))

        elif kind == "skill_loaded":
            name = kw.get("name", "")
            pkg = kw.get("package", "")
            pkg_str = f" ({pkg})" if pkg else ""
            out.say(out.cyan(f"        [Skill Loaded] Active App Skill: '{name}'{pkg_str}"))

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
                          on_event=_live_reporter(out, max_steps=cfg.run.max_steps))
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
                for line in state.scratchpad.plain().splitlines():
                    out.say(f"  {line}")
                out.say()
            out.say(f"  {colour(outcome.upper())}  "
                    f"{state.step} steps, {state.llm_calls} LLM calls, "
                    f"{tilde}${spent:.4f}, {elapsed:.1f}s")
            if outcome != "success":
                exit_code = 1
            if outcome in ("aborted", "needs_user"):
                break
        if iteration > 1:
            out.say(out.dim(f"  session total: {tilde}${llm.ledger.total_usd:.4f} "
                            f"over {iteration} iteration(s)"))
    return exit_code








# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def resolve_run(run_arg: Optional[str], artifacts_dir: str = "runs") -> Optional[Path]:
    """A run directory from a path, or the most recent one for "latest"/omitted."""
    if run_arg and run_arg != "latest":
        return Path(run_arg).expanduser()
    runs_dir = Path(artifacts_dir).expanduser()
    if not runs_dir.is_dir():
        return None
    candidates = sorted((d for d in runs_dir.iterdir() if d.is_dir()),
                        key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * pct))]


def _cost_summary(out: Out, events: List[Dict[str, Any]]) -> None:
    """Where the wall clock and the tokens went.

    Recorded per step by `agent.step_metrics`; runs from before that landed have
    no `llm` block and simply print nothing.
    """
    # Reported per kind of call, not pooled. A swept item is a real vision call
    # and belongs in the totals, but it is ~25 output tokens against a reasoning
    # turn's ~4,400 -- pool the two and the median step becomes a vision read,
    # which is the opposite of what this block exists to show.
    groups = (("decisions", ("decide", "judge")),
              ("sweep reads", ("sweep_step",)))
    buckets = [(label, [e for e in events
                        if e.get("kind") in kinds and e.get("llm")])
               for label, kinds in groups]
    buckets = [(label, rows) for label, rows in buckets if rows]
    if not buckets:
        return

    out.say()
    out.say(out.bold("  ── Cost of thinking ──"))
    advice: List[str] = []

    for label, rows in buckets:
        def col(key: str) -> List[float]:
            return [float(e["llm"].get(key) or 0) for e in rows]

        latency = [float(e.get("wall_s") or e["llm"].get("latency_s") or 0)
                   for e in rows]
        prompt, cached = col("prompt_tokens"), col("cached_tokens")
        completion, reasoning = col("completion_tokens"), col("reasoning_tokens")
        reasoning_chars = col("reasoning_chars")
        # Providers do not all report `reasoning_tokens`; the characters we saw
        # streamed are measured either way, so fall back to those at 4 chars/token.
        if not any(reasoning) and any(reasoning_chars):
            reasoning = [c / 4 for c in reasoning_chars]
            estimated = " (est. from streamed text)"
        else:
            estimated = ""

        total_prompt, total_cached = sum(prompt), sum(cached)
        total_completion, total_reasoning = sum(completion), sum(reasoning)
        hit_rate = (total_cached / total_prompt * 100) if total_prompt else 0.0
        think_share = ((total_reasoning / total_completion * 100)
                       if total_completion else 0.0)

        if len(buckets) > 1:
            out.say(f"  {label} ({len(rows)})")
        out.say(f"  latency/step   {_median(latency):6.1f}s median   "
                f"{_percentile(latency, 0.9):6.1f}s p90   "
                f"{sum(latency):8.0f}s total")
        out.say(f"  prompt tokens  {_median(prompt):6.0f} median   "
                f"{total_prompt:8.0f} total   "
                f"{hit_rate:4.0f}% served from cache")
        out.say(f"  output tokens  {_median(completion):6.0f} median   "
                f"{total_completion:8.0f} total")
        if total_reasoning:
            out.say(f"  of which think {_median(reasoning):6.0f} median   "
                    f"{total_reasoning:8.0f} total   "
                    f"{think_share:4.0f}% of output{estimated}")

        # Advice is keyed to the reasoning turns only. A one-shot vision read has
        # no prefix worth caching, so scolding it for a cold cache is noise.
        if label != "decisions":
            continue
        if think_share >= 50:
            advice.append("Reasoning tokens dominate output, so they dominate "
                          "latency: try a lower reasoning effort.")
        if total_prompt and hit_rate < 40:
            advice.append("Low prompt-cache hit rate: something near the top of "
                          "the prompt is changing every turn.")

    for line in advice:
        out.say(out.dim(f"  {line}"))


def cmd_report(args) -> int:
    out = Out()
    path = resolve_run(getattr(args, "run", None),
                       getattr(args, "artifacts_dir", None) or "runs")
    if path is None:
        out.bad("no runs found")
        return 1
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
    last_notes = None
    for event in events:
        if event["kind"] == "decide":
            action = event.get("action", {})
            shot = " +img" if event.get("screenshot") else ""
            out.say(f"  {event.get('step', 0):>3}{shot} "
                    f"{action.get('action')} {action.get('target') or ''}")
            if action.get("observation"):
                out.say(out.dim(f"        Obs:       {action.get('observation')}"))
            if action.get("reasoning"):
                out.say(out.dim(f"        Reasoning: {action.get('reasoning')}"))
            if action.get("progress"):
                out.say(out.dim(f"        Progress:  {action.get('progress')}"))
            if action.get("notes"):
                notes_text = _notes_text(action.get("notes"))
                last_notes = notes_text
                out.say(out.dim(f"        Notes:     {notes_text}"))
        elif event["kind"] == "image_analysis":
            result = event.get("result", "")
            model = event.get("model", "")
            m_str = f" ({model})" if model else ""
            out.say(out.dim(f"        Vision{m_str}: {result}"))
        elif event["kind"] == "verify":
            out.say(f"      -> {event.get('grade')} {event.get('reason') or ''}")
        elif event["kind"] in ("dismiss", "refused", "loop_break", "sensitive",
                               "judge", "error", "gave_up"):
            # `llm` and `wall_s` are per-call metrics; they are summarised in the
            # cost block below rather than dumped inline as a dict.
            out.say(f"      [{event['kind']}] "
                    + " ".join(f"{k}={v}" for k, v in event.items()
                               if k not in ("t", "kind", "llm", "wall_s")))
    if last_notes:
        out.say()
        out.say(out.bold("  ── Collected Data ──"))
        out.say()
        out.say(f"  {last_notes}")
    _cost_summary(out, events)
    out.say()
    if end:
        out.say(f"  {end.get('outcome', '?').upper()}: {end.get('steps')} steps, "
                f"{end.get('llm_calls')} LLM calls, ${end.get('usd', 0):.4f}")
    return 0


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def cmd_replay(args) -> int:
    """Re-issue a recorded run's decisions and diff them against what it did."""
    from .actions import AgentAction
    from .llm import LLMClient
    from . import replay as rp

    # `--json` puts a document on stdout, so the narration has to go quiet --
    # otherwise the header lines land in whatever is parsing it. Failures still
    # print, because a bare exit code is not a diagnosis.
    out = Out()
    say = Out(quiet=bool(args.json))

    cfg = build_config(args)
    if not cfg.llm.model:
        out.bad("no model chosen. Run `adbagent models` and pass --model.")
        return 1

    path = resolve_run(getattr(args, "run", None), cfg.run.artifacts_dir)
    if path is None:
        out.bad("no runs found")
        return 1

    steps = getattr(args, "steps", None)
    try:
        cases = rp.load_cases(path, purpose=args.purpose, steps=steps,
                              limit=args.limit or 0)
    except rp.ReplayError as exc:
        out.bad(str(exc))
        return 1
    if not cases:
        out.bad(f"no replayable {args.purpose} cases in {path}")
        return 1

    llm = LLMClient(cfg, run_id=f"replay-{path.name}")
    schema_cls = AgentAction if args.purpose == "decide" else None
    if schema_cls is None:
        out.bad(f"replaying {args.purpose!r} is not supported yet "
                f"(only 'decide' has a schema to diff)")
        return 1

    def decide(messages: List[Dict[str, Any]]):
        mark = llm.ledger.mark()
        action = llm.structured(messages, AgentAction, model=cfg.llm.model,
                               purpose="replay")
        calls = llm.ledger.since(mark)
        from .agent import step_metrics
        return action, step_metrics(calls, detail=False)

    mode = "rebuilt system prompt" if args.rebuild_system else "verbatim"
    say.say(say.bold(f"  replaying {len(cases)} {args.purpose} case(s) from "
                     f"{path.name}"))
    say.say(say.dim(f"  mode: {mode}   model: {cfg.llm.model}"))
    say.say()

    colour = {"match": say.green, "same_action": say.yellow,
              "differs": say.red, "error": say.red}

    def show(result: rp.Result) -> None:
        tag = colour[result.verdict](f"{result.verdict:<11}")
        line = (f"  {result.step:>4}  {tag} {result.recorded:<28} "
                f"{result.replayed or result.error:<28}")
        if result.verdict != "match" and result.grade:
            line += say.dim(f" (recorded: {result.grade})")
        say.say(line)

    report = rp.replay(cases, decide,
                       rebuild_system_prompt=args.rebuild_system,
                       on_result=None if args.json else show)
    risky = report.regressions

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 1 if risky else 0

    say.say()
    say.say(say.bold(f"  {report.count('match')}/{report.n} identical "
                     f"({report.agreement * 100:.0f}%)"))
    for verdict in ("same_action", "differs", "error"):
        n = report.count(verdict)
        if n:
            say.say(f"  {n} {verdict.replace('_', ' ')}")
    if report.skipped:
        say.say(say.dim(f"  {len(report.skipped)} skipped (dumped images are "
                        f"placeholders, not replayable)"))
    if risky:
        say.warn(f"{len(risky)} diverged from a step that had worked: "
                 + ", ".join(str(r.step) for r in risky[:12]))
    else:
        say.ok("no divergence from a step that had worked")
    thinking = report.median("reasoning_tokens") or report.median("reasoning_chars") / 4
    tilde = "~" if llm.ledger.estimated else ""
    say.say(say.dim(
        f"  median {report.median('latency_s'):.1f}s/case, "
        f"{report.median('completion_tokens'):.0f} output tokens "
        f"({thinking:.0f} thinking), "
        f"{tilde}${llm.ledger.total_usd:.4f} spent"))
    return 1 if risky else 0


# ---------------------------------------------------------------------------
# scratchpad
# ---------------------------------------------------------------------------

def cmd_scratchpad(args) -> int:
    out = Out()
    path = resolve_run(getattr(args, "run", None))
    if path is None:
        out.bad("no runs found")
        return 1

    events_file = path / "events.jsonl" if path.is_dir() else path
    if not events_file.exists():
        out.bad(f"no events at {events_file}")
        return 1

    # `notes` in an event is a delta, not the whole ledger, so the records are
    # replayed rather than read off the last decide -- see `scratchpad.replay`.
    events = []
    last_vision = None
    for line in events_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            events.append(event)
            if event.get("kind") == "image_analysis" and event.get("result"):
                last_vision = event.get("result")
        except Exception:
            pass
    collected = scratchpad.replay(events).plain()

    if collected or last_vision:
        out.say(out.bold(f"  ── Scratchpad ({path.name}) ──"))
        if collected:
            out.say()
            for line in collected.splitlines():
                out.say(f"  {line}")
        if last_vision:
            out.say()
            out.say(f"  Latest Vision Analysis: {last_vision}")
        out.say()
    else:
        out.say(out.dim(f"  No scratchpad data collected in {path.name}"))

    return 0


# ---------------------------------------------------------------------------
# apps
# ---------------------------------------------------------------------------

def cmd_apps(args) -> int:
    from .device import Device

    out = Out()
    cfg = build_config(args)
    _ensure_device(args, cfg, out)

    with Device(cfg, args.device or "") as dev:
        query = getattr(args, "search", "") or ""
        third_party = getattr(args, "third_party", False)
        pkgs = dev.list_apps(query=query, third_party_only=third_party)
        title = "Installed Apps" if not third_party else "Installed 3rd-Party Apps"
        if query:
            title += f" matching {query!r}"
        out.say(out.bold(f"  {title} ({len(pkgs)})"))
        out.say()
        for pkg in pkgs:
            out.say(f"  - {pkg}")
        if not pkgs:
            out.say("  (no matching apps found)")
    return 0


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

def cmd_skills(args) -> int:
    from .skills import SkillRegistry, SkillGenerator, Skill
    from .llm import LLMClient
    from .device import Device

    out = Out()
    cfg = build_config(args)
    registry = SkillRegistry(cfg.skills.skills_dir)

    action = getattr(args, "skills_action", "") or "list"

    if action == "list":
        skills = registry.list_skills()
        out.say(out.bold(f"  App Skills in {registry.skills_dir} ({len(skills)})"))
        out.say()
        for sk in skills:
            pkgs = f" [{', '.join(sk.packages)}]" if sk.packages else ""
            out.say(f"  - {out.bold(sk.name)}{pkgs}: {sk.description[:60] if sk.description else 'No description'}")
        if not skills:
            out.say("  (no skills found in skills directory)")
        return 0

    if action == "view":
        name_or_pkg = getattr(args, "target", "")
        if not name_or_pkg:
            out.bad("Please specify an app name or package for 'view'. Example: adbagent skills view whatsapp")
            return 1
        skill = registry.find_by_name_or_alias(name_or_pkg) or registry.find_by_package(name_or_pkg)
        if not skill:
            out.bad(f"No skill found for '{name_or_pkg}'. Run 'adbagent skills list' to view available skills.")
            return 1
        out.say(skill.to_markdown())
        return 0

    if action == "create":
        name = getattr(args, "target", "")
        if not name:
            out.bad("Please specify an app name for 'create'. Example: adbagent skills create MyApp")
            return 1
        skill = Skill(
            name=name,
            packages=[f"com.example.{name.lower()}"],
            aliases=[name.lower()],
            description=f"App skill for {name}.",
            workflows=[],
            nuances=["First nuance or UI quirk."],
            recommendations=["First action recommendation."]
        )
        saved_path = registry.save_skill(skill)
        out.ok(f"Created new skill template: {saved_path}")
        return 0

    if action == "generate":
        from .agent import Agent
        from .memory import Memory

        app_target = getattr(args, "app", "") or getattr(args, "target", "")
        user_tasks = getattr(args, "tasks", "") or "Explore key screens and workflows in the app"

        if not app_target:
            out.bad("Please specify an app name or package via --app or argument. Example: adbagent skills generate --app com.whatsapp --tasks 'search contact, send message'")
            return 1

        api_key = cfg.api_key()
        model_name = cfg.llm.skill()
        llm = LLMClient(cfg, api_key=api_key) if api_key else None

        out.say(out.bold(f"  Exploring app '{app_target}' live on device & generating Skill using model '{model_name}'..."))
        out.say(out.dim(f"  Tasks to perform and verify: {user_tasks}"))
        out.say()

        screen_summaries: List[str] = []
        actions_taken: List[str] = []
        screenshots: List[bytes] = []

        try:
            with Device(cfg, getattr(args, "device", "") or "") as dev, Memory(cfg) as mem:
                dev.open_app(app_target)
                screen_init = dev.observe()
                try:
                    shot_init = dev.screenshot()
                    if shot_init:
                        screenshots.append(shot_init)
                except Exception:  # noqa: BLE001
                    pass

                screen_summaries.append(f"Initial Package: {screen_init.package}, Title/Elements: {[e.best_text for e in screen_init.elements[:15] if e.best_text]}")
                actions_taken.append(f"Opened target app {app_target}")

                if llm:
                    live_reporter = _live_reporter(out, max_steps=cfg.run.max_steps)

                    def exploration_tracer(kind: str, **kw: Any) -> None:
                        live_reporter(kind, **kw)
                        if kind == "step":
                            s = kw.get("screen")
                            act = kw.get("action")
                            if s:
                                elems = [e.best_text for e in s.elements[:15] if e.best_text]
                                screen_summaries.append(f"Package: {s.package}, Elements: {elems}")
                            if act:
                                desc = act.describe() if hasattr(act, "describe") else str(act)
                                obs = getattr(act, "observation", "")
                                actions_taken.append(f"Executed action: {desc}" + (f" (Observed: {obs})" if obs else ""))
                            try:
                                shot = dev.screenshot()
                                if shot and len(screenshots) < 10:
                                    screenshots.append(shot)
                            except Exception as shot_exc:  # noqa: BLE001
                                log.warning("Could not capture screenshot during exploration step: %s", shot_exc)

                    goal_text = f"Explore app {app_target} and perform the following tasks: {user_tasks}"
                    agent = Agent(dev, mem, llm, cfg, on_event=exploration_tracer)
                    out.say(out.bold("  ── Live App Exploration Run ──"))
                    agent.run(goal_text)
        except Exception as exc:  # noqa: BLE001
            out.warn(f"Live device interaction encounters warning: {exc}. Proceeding with LLM synthesis based on available trace.")

        generator = SkillGenerator(registry)
        skill = generator.generate_from_exploration(
            app_target, user_tasks, screen_summaries, actions_taken, llm or cfg, screenshots=screenshots
        )
        saved_path = registry.save_skill(skill)
        out.say()
        out.ok(f"Verified live actions and saved skill for '{skill.name}' to {saved_path}")
        out.say()
        out.say(skill.to_markdown())
        return 0

    out.bad(f"Unknown skills action '{action}'. Use list, view, create, or generate.")
    return 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", help="path to config.json")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--provider", help="llm provider (default fireworks)")
    parser.add_argument("--service-tier", dest="service_tier",
                        help="service tier for LLM requests (e.g. priority)")
    parser.add_argument("--model", help="model id")
    parser.add_argument("--model-small", dest="model_small",
                        help="cheaper model for judging and repair")
    parser.add_argument("--model-image", dest="model_image",
                        help="model for vision calls with screenshots")
    parser.add_argument("--model-skill", dest="model_skill",
                        help="dedicated model for app skill generation and exploration")
    parser.add_argument("--skills-dir", dest="skills_dir",
                        help="directory for app skills (default ./skills)")
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



    p = sub.add_parser("report", help="summarise a recorded run")
    p.add_argument("run", nargs="?", default="latest",
                   help="path to runs/<id> or its events.jsonl "
                        "(default: the most recent run)")
    _add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("replay",
                       help="re-run a recorded run's decisions and diff them")
    p.add_argument("run", nargs="?", default="latest",
                   help="path to runs/<id> (default: the most recent run)")
    p.add_argument("--rebuild-system", dest="rebuild_system",
                   action="store_true",
                   help="swap in the system prompt prompts.py builds today, "
                        "to test a prompt edit (default: send the recording "
                        "verbatim, to test a model or decoder change)")
    p.add_argument("--limit", type=int, default=0,
                   help="replay at most N cases, sampled evenly across the run")
    p.add_argument("--steps", type=int, nargs="+",
                   help="replay only these step numbers")
    p.add_argument("--purpose", default="decide",
                   help="which recorded calls to replay (default: decide)")
    p.add_argument("--json", action="store_true",
                   help="machine-readable report on stdout")
    _add_common(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("scratchpad", help="show latest or specified run scratchpad / collected data")
    p.add_argument("run", nargs="?", default="latest", help="path to run directory or 'latest' (default)")
    _add_common(p)
    p.set_defaults(func=cmd_scratchpad)

    p = sub.add_parser("apps", help="list or search installed app packages on device")
    p.add_argument("-s", "--search", help="filter packages by substring")
    p.add_argument("-3", "--third-party", dest="third_party", action="store_true",
                   help="only show third-party installed apps")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_apps)

    p = sub.add_parser("skills", help="manage app skills (list, view, create, generate)")
    p.add_argument("skills_action", nargs="?", choices=["list", "view", "create", "generate"],
                   default="list", help="action to perform (default: list)")
    p.add_argument("target", nargs="?", help="app name or package for view, create, or generate")
    p.add_argument("--app", help="app package or name for skill generation")
    p.add_argument("--tasks", help="user-defined task instructions to perform in app during exploration")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_skills)

    return parser


@contextlib.contextmanager
def prevent_sleep():
    """Prevent macOS system sleep while adbagent runs."""
    proc = None
    if sys.platform == "darwin":
        try:
            proc = subprocess.Popen(
                ["caffeinate", "-w", str(os.getpid()), "-d", "-i", "-s"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    try:
        yield
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", 0))
    with prevent_sleep():
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
