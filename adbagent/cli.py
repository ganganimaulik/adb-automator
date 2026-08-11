"""Command line interface."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__, runlog, scratchpad

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
            try:
                sys.stdout.write(text)
                sys.stdout.flush()
            except OSError:
                pass

    def say(self, text: str = "") -> None:
        if not self.quiet:
            try:
                print(text)
            except OSError:
                pass

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
    # The verbosity belongs on the console *handler*, not on the logger: a run
    # writes every debug record to `runs/<id>/run.log` regardless of what the
    # terminal was asked to show, and a logger left at WARNING drops those
    # records before any handler -- including that one -- can see them. See
    # `runlog.attach`, which lowers the logger and puts it back.
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
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
    "model_skill_image": "llm.model_skill_image",
    "provider": "llm.provider",
    "service_tier": "llm.service_tier",
    "rpm": "llm.rpm",
    "max_tokens": "llm.max_tokens",
    "max_tokens_image": "llm.max_tokens_image",
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
    "learn_after_run": "skills.learn_after_run",
    # `watch` only. Distinct dest names because `watch.max_steps` and
    # `run.max_steps` are different budgets and sharing `--max-steps` between
    # them would silently drop whichever one lost -- see `Watch.__init__`.
    "watch_interval": "watch.interval_s",
    "watch_sweep": "watch.sweep_s",
    "watch_max_steps": "watch.max_steps",
    "watch_draft": "watch.draft",
    "watch_policy": "watch.policy",
    "watch_ledger": "watch.ledger",
    "watch_cooldown": "watch.thread_cooldown_s",
    "watch_replies_per_hour": "watch.max_replies_per_hour",
    "watch_replies_per_thread": "watch.max_replies_per_thread_per_hour",
    "watch_usd_per_hour": "watch.max_usd_per_hour",
    "watch_fail_closed": "watch.fail_closed",
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

#: Every job a run gives a model, the config field it comes from, and the depths
#: that job asks for. Deciding is the only job with two, because it is the only one
#: whose difficulty varies from turn to turn -- and the skill model decides for the
#: whole of `skills generate`, so it asks for both as well.
_REASONING_JOBS = (
    ("deciding", "model", (("decide", False), ("decide", True))),
    ("judging", "small", (("judge", True),)),
    ("vision", "image", (("analyze_image", False),)),
    ("skills", "skill", (("decide", False), ("decide", True))),
)


def _check_vision(out: Out, cfg) -> int:
    """Confirm the models that will be handed images can actually take them.

    Both ways this can be wrong fail quietly, which is why it is worth a network
    call here rather than a surprise mid-run:

    * `llm.model_image` pointing at a text-only model. `analyze_image` swallows
      the 400 so the step is not lost, so every screenshot turn spends a call,
      gets nothing, and carries on -- blind on exactly the turns
      `needs_screenshot` decided the element tree could not answer.
    * A deciding model that gets the frame itself but cannot take it. There the
      400 takes the whole turn with it, and there are two ways in: setting
      `llm.vision_in_decider`, and naming one model for `llm.model` and
      `llm.model_image`, which means the same thing (`llm.decider_sees`).

    The catalogue answers both: `supportsImageInput` per model. `llm.model_image`
    used to be documented as uncheckable for want of exactly this flag; it is
    there now, so the guesswork can stop.

    Returns the number of problems found. A catalogue that cannot be reached is
    not one of them -- `doctor` runs on aeroplanes.
    """
    from .config import same_model
    from .llm import PROVIDERS, list_models, qualify

    # Said before the catalogue is fetched, so it is still said on the aeroplane.
    # Nobody typed this one, and a saving nobody asked for is one nobody thinks
    # to check for. Both pairs, because `skills generate` resolves its own
    # (`skills.use_skill_model`) and the config can match on one and not the
    # other -- which is the common case, one model for the tour and another for
    # the run.
    if cfg.llm.decider_sees() and not cfg.llm.vision_in_decider:
        out.say(out.dim("        (llm.model and llm.model_image name one model, "
                        "so the screenshot goes straight into the deciding call "
                        "-- one round trip per screenshot turn, not two)"))
    if ((cfg.llm.model_skill or cfg.llm.model_skill_image)
            and same_model(cfg.llm.skill(),
                           cfg.llm.model_skill_image or cfg.llm.model_image)):
        out.say(out.dim("        (llm.model_skill and llm.model_skill_image name "
                        "one model, so a `skills generate` tour goes straight "
                        "into the deciding call the same way)"))

    provider = PROVIDERS.get(cfg.llm.provider)
    key = cfg.api_key()
    if provider is None or not key or not provider.catalogue_url:
        return 0

    # (config field, the value, why it gets an image)
    wants_vision = [("llm.model_image", cfg.llm.image(),
                     "every screenshot turn goes to it")]
    if cfg.llm.decider_sees():
        if same_model(cfg.llm.model, cfg.llm.image()):
            # One model, both jobs. Checking it twice reads as two problems.
            wants_vision[0] = (
                "llm.model_image", cfg.llm.image(),
                "every screenshot turn goes to it -- and it is the deciding "
                "model too, so the frame rides in the deciding call")
        else:
            wants_vision.append(
                ("llm.model", cfg.llm.model,
                 "llm.vision_in_decider puts the frame in the deciding call"))
    if cfg.llm.model_skill_image:
        wants_vision.append(("llm.model_skill_image", cfg.llm.skill_image(),
                             "the screenshot pass of skill synthesis"))

    try:
        catalogue = {qualify(provider, m.id): m for m in list_models(provider, key)}
    except Exception as exc:  # noqa: BLE001 - offline is not a misconfiguration
        out.say(out.dim(f"        (could not check vision support: {exc})"))
        return 0

    problems = 0
    for field, name, why in wants_vision:
        if not name:
            continue
        info = catalogue.get(qualify(provider, name))
        if info is None:
            out.warn(f"{field} = {name} is not in the catalogue -- "
                     f"cannot confirm it takes images")
        elif not info.vision:
            out.bad(f"{field} = {name} does not take images, and {why}")
            problems += 1
        else:
            out.ok(f"{field} takes images ({name.rsplit('/', 1)[-1]})")
    if problems:
        out.say("        pick one from: adbagent models --vision")
    return problems


def _report_reasoning(out: Out, cfg) -> None:
    """Print the exact request fields the reasoning setting will send.

    Two conventions exist for this on the OpenAI wire protocol, most models take
    neither, and nothing in the catalogue says which a given model wants. A wrong
    guess either 400s -- survivable, since `_post` drops the field and carries on
    -- or is silently ignored, which is worse: the bill and the clock say nothing
    changed while the config says it was capped. Printing the body is what makes
    that checkable before a long run depends on it.

    Reported per model, because a run uses up to four and they need not agree. A
    reasoning decider alongside a vision model that does not think is normal, not
    a mistake.
    """
    from .llm import (PROVIDERS, known_non_reasoning, qualify, reasoning_body,
                      reasoning_style_for)

    if not cfg.llm.reasoning_effort:
        out.say(out.dim("        reasoning depth left to the model "
                        "(set llm.reasoning_effort to cap it)"))
        return

    provider = PROVIDERS.get(cfg.llm.provider)
    style = cfg.llm.reasoning_style

    # One model commonly serves several jobs, so group by model and collect every
    # depth it will actually be asked for.
    per_model: Dict[str, Tuple[List[str], List[str]]] = {}
    for label, field, calls in _REASONING_JOBS:
        name = getattr(cfg.llm, field)() if field != "model" else cfg.llm.model
        if not name:
            continue
        model = qualify(provider, name) if provider else name
        labels, depths = per_model.setdefault(model, ([], []))
        labels.append(label)
        for purpose, hard in calls:
            depth = cfg.llm.effort_for(purpose, hard=hard)
            if depth not in depths:
                depths.append(depth)

    unknown: List[Tuple[str, str]] = []
    printed_a_body = False
    for model, (labels, depths) in per_model.items():
        short = model.rsplit("/", 1)[-1]
        used = "/".join(labels)
        bodies = [(depth, reasoning_body(model, depth, style)) for depth in depths]
        if not any(body for _, body in bodies):
            # Two different silences, and only one wants doing something about.
            if known_non_reasoning(model):
                out.ok(f"{short} ({used}) does not reason -- nothing to cap")
            else:
                unknown.append((short, used))
            continue
        resolved = style if style in ("effort", "thinking", "off") else \
            reasoning_style_for(model)
        out.ok(f"{short} ({used}): {resolved} convention"
               f"{', from the model name' if style == 'auto' else ', configured'}")
        for depth, body in bodies:
            out.say(out.dim(f"        {depth:<6} sends {json.dumps(body)}"))
            printed_a_body = True

    for short, used in unknown:
        # Telling someone to force a style on a model that does not think would
        # break every call it makes, so both readings are offered.
        out.warn(f"{short} ({used}) is not a model this knows a reasoning "
                 f"convention for, so nothing will be sent")
        out.say(out.dim("        if it does reason, set llm.reasoning_style to "
                        "'effort' or 'thinking'; if it does not, nothing to fix"))
    if printed_a_body:
        out.say(out.dim("        confirm the bodies above against your provider's "
                        "docs -- an ignored field looks like a working one"))


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
        out.ok(f"API key is set ({len(key)} chars)")
    else:
        out.bad(f"no API key: set llm.api_key in config.json or ${cfg.llm.api_key_env}")
        problems += 1
    if cfg.llm.model:
        out.ok(f"model {cfg.llm.model}")
        if cfg.llm.service_tier:
            out.ok(f"service tier {cfg.llm.service_tier}")
        if cfg.llm.model_small:
            out.ok(f"small model {cfg.llm.model_small}")
        if cfg.llm.model_image:
            out.ok(f"vision model {cfg.llm.model_image}")
        problems += _check_vision(out, cfg)
        _report_reasoning(out, cfg)
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
        out.bad(f"no API key: set llm.api_key in config.json or ${cfg.llm.api_key_env}")
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
            elif purpose == "locate":
                label = "LLM locator"
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
            elif purpose == "locate":
                tag = "LLM locator responded"
            else:
                tag = "LLM responded"
            out.say(out.dim(f"        {tag} in {elapsed:.2f}s{tokens_info}"))

        elif kind == "image_analysis":
            result = kw.get("result", "")
            model = kw.get("model", "")
            m_str = f" ({model})" if model else ""
            out.say(out.dim(f"        Vision{m_str}: {result}"))

        elif kind == "vision_unavailable":
            # Loud, and yellow, because the failure is survivable but the turn is
            # not what it looks like: the screenshot was taken, paid for, and read
            # by nobody. `adbagent doctor` names the usual cause.
            out.say(out.yellow(
                f"        [Vision] {kw.get('model', '')} did not read the "
                f"screenshot ({kw.get('error', 'no answer')}). Deciding from the "
                f"element list alone -- check: adbagent doctor"))

        elif kind == "vision_skipped":
            # Dim, not yellow: unlike `vision_unavailable` nothing went wrong
            # here. The frame was read -- by a percentile, not a model -- and it
            # held nothing.
            out.say(out.dim(
                f"        Vision: frame is blank, {kw.get('purpose', 'read')} "
                f"answered without a model call"))

        elif kind == "vision_reread":
            out.say(out.dim(
                f"        Vision: value reported unreadable, re-reading at "
                f"{kw.get('long_edge', 0)}px"))

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

        elif kind == "reasoning_unsupported":
            # Said once, and said out loud: continuing quietly would leave the
            # run looking capped when it is not.
            out.say(out.yellow(
                f"        [Reasoning] {kw.get('model', '')} rejected "
                f"{', '.join(kw.get('fields') or [])} -- it does not take a "
                f"reasoning setting. Continuing without one."))

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


def _learn(out: Out, traces, llm, cfg, goal: str) -> None:
    """Fold what the finished run learned into each app's skill it worked in.

    `traces` is one trace per app the run toured -- a price check across two
    apps teaches both, not just the one more steps happened to land in.

    Reported rather than done quietly: it spends a call and rewrites a file the
    next run will obey, and a silent rewrite of the agent's own instructions is
    not something to discover later. A failure here never fails the run -- the
    goal was already met or missed before this point.
    """
    from .skills import SkillRegistry, learn_from_run

    try:
        registry = SkillRegistry(cfg.skills.skills_dir)
    except Exception as exc:  # noqa: BLE001
        out.warn(f"could not update the app skill: {exc}")
        return
    learned = []
    for trace in traces:
        try:
            skill = learn_from_run(trace, llm, registry, goal=goal, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            out.warn(f"could not update the {trace.package or 'app'} skill: {exc}")
            continue
        if skill is not None:
            learned.append(skill)
    if not learned:
        main = traces[0] if traces else None
        out.say(out.dim(f"  learned nothing new about "
                        f"{(main.package if main else '') or 'this run'} "
                        f"({main.steps if main else 0} steps, "
                        f"{len(main.screens) if main else 0} screens)"))
        return
    for skill in learned:
        out.say(out.cyan(f"  skill '{skill.name}' updated from this run "
                         f"({len(skill.workflows)} workflows, {len(skill.nuances)} nuances) "
                         f"-> {registry.path_for(skill)}"))


#: How wide the result paragraph is allowed to get before it is wrapped. Read
#: off the terminal, clamped: a maximised window would otherwise print the
#: answer as one 300-column line, which is the format it was already effectively
#: in when it was buried in the step feed.
_RESULT_WIDTH = 96


def _wrapped(text: str, indent: str = "  ") -> List[str]:
    """`text` as indented lines that fit the terminal, keeping its own breaks.

    A collection goal answers with a list -- one line per conversation, per
    price, per item -- so the newlines the model wrote are structure, not
    accidents, and reflowing the whole thing into one paragraph would lose it.
    """
    import shutil
    import textwrap

    width = max(min(shutil.get_terminal_size((80, 24)).columns,
                    _RESULT_WIDTH), len(indent) + 24)
    out: List[str] = []
    for para in text.splitlines():
        if not para.strip():
            out.append("")
            continue
        # Continuations are indented further than the line they continue, so a
        # wrapped list item cannot be misread as the next item -- which on the
        # list-shaped answers this exists for is most of them.
        out.extend(textwrap.wrap(para, width=width, initial_indent=indent,
                                 subsequent_indent=indent + "    "))
    return out


def _result_block(out: Out, outcome: str, result: str, evidence: str = "",
                  *, progress: str = "", problem: str = "") -> None:
    """What the run concluded, in its own block at the bottom.

    The answer to a "read X and tell me" goal is the text of the `done` action,
    and it used to appear in exactly one place: inside the last line of the step
    feed, in the same shape as the forty tap lines above it. On anything longer
    than a couple of steps that is off the top of the terminal by the time the
    run ends, so the visible ending was the outcome word and the bill.

    A run can also stop without answering -- the step budget, a lost device, an
    abort. Rather than print an empty heading, this says so and falls back to
    the last two things the run knew: where it had got to, and what was going
    wrong when it stopped.
    """
    out.say()
    out.say(out.bold("  ── Result ──"))
    out.say()
    if result:
        # Undimmed and uncoloured, alone in that: everything around it -- the
        # step feed above, the evidence below -- is dim, and the outcome word
        # under it is already carrying the green or the red.
        for line in _wrapped(result):
            out.say(line)
    else:
        out.say(out.dim("  the run ended without an answer "
                        f"({_NO_ANSWER.get(outcome, 'it stopped early')})"))
        for label, text in (("last progress", progress), ("last problem", problem)):
            if not text:
                continue
            for line in _wrapped(f"{label}: {' '.join(text.split())}",
                                 indent="  "):
                out.say(out.dim(line))
    if evidence:
        out.say()
        for line in _wrapped(" ".join(evidence.split()), indent="  "):
            out.say(out.dim(line))


#: Why there is no answer, per outcome, so the empty case still explains itself.
_NO_ANSWER = {
    # Reachable: `--assert-text`/`--assert-shell` end a run at the top of the
    # loop, before the model is asked for anything, so there is no summary to
    # print and the evidence line below carries the whole ending.
    "success": "the success check settled it before the agent reported one",
    "failed": "it ran out of steps or gave up without saying why",
    "aborted": "it was interrupted, or the device or the budget gave out",
    "needs_user": "it stopped for a person",
}


def _resolve_resume(target: str, artifacts_dir: str) -> Optional[Path]:
    """The run directory a `--resume` value points at, or None."""
    from . import checkpoint as ckpt

    if target == "latest":
        return ckpt.latest_resumable(Path(artifacts_dir).expanduser())
    path = Path(target).expanduser()
    if path.is_dir():
        return path
    candidate = Path(artifacts_dir).expanduser() / target
    return candidate if candidate.is_dir() else None


def cmd_run(args) -> int:
    from . import checkpoint as ckpt
    from . import skills as skillmod
    from .agent import Agent, Oracle
    from .device import Device
    from .llm import LLMClient
    from .memory import Memory

    out = Out()
    cfg = build_config(args)

    # -- where does the goal come from? ------------------------------------
    # Normally the command line. With --resume it comes from the checkpoint,
    # because everything the run remembers -- its history, its dead ends, its
    # collected data -- is keyed to the goal it started with.
    resume_data: Optional[Dict[str, Any]] = None
    run_id = ""
    goal = args.goal or ""
    if args.resume is not None:
        run_dir = _resolve_resume(args.resume, cfg.run.artifacts_dir)
        if run_dir is None:
            if args.resume == "latest":
                out.bad("no resumable run found -- a run leaves a checkpoint "
                        "when it fails or is interrupted")
            else:
                out.bad(f"no such run: {args.resume}")
            return 1
        resume_data = ckpt.load(run_dir)
        if resume_data is None:
            out.bad(f"run {run_dir.name} has no checkpoint to resume from "
                    f"-- it either finished or predates checkpoints")
            return 1
        run_id = resume_data.get("run_id") or run_dir.name
        ckpt_goal = resume_data.get("goal") or ""
        if goal and goal != ckpt_goal:
            out.warn(f"resuming run {run_id} -- keeping its original goal, "
                     f"not the one just given")
        goal = ckpt_goal
        # The budget is for *additional* steps: the steps already taken are
        # spent, so the ceiling moves up by that many. --max-steps still says
        # how much more work this sitting may do.
        cfg.run.max_steps += int(resume_data.get("step") or 0)
    if not goal:
        out.bad("no goal given -- pass one, or continue a failed run with "
                "--resume")
        return 1

    if not cfg.llm.model:
        out.bad("no model chosen. Run `adbagent models` and pass --model.")
        return 1

    _ensure_device(args, cfg, out)

    oracle = Oracle(shell=args.assert_shell or "", equals=args.assert_equals or "",
                    text=args.assert_text or "")

    repeats = args.repeat
    infinite = isinstance(repeats, str) and repeats == "inf"
    total = 0 if infinite else int(repeats or 1)

    if resume_data:
        out.say(out.bold(f"  resuming run {run_id} from step "
                         f"{resume_data.get('step') or 0}"))
        out.say(out.dim(f"  goal: {goal}"))

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
            # The trace wraps the reporter, so the run pays nothing for it beyond
            # a screenshot of each new screen -- and a new Agent per iteration
            # reads back whatever the last one learned.
            trace = skillmod.TraceCollector(
                dev, skillmod.AppTrace(tasks=goal),
                on_event=_live_reporter(out, max_steps=cfg.run.max_steps))
            agent = Agent(dev, mem, llm, cfg, oracle=oracle, on_event=trace)
            if infinite or total > 1:
                out.say(out.bold(f"\n  iteration {iteration}"))
            started = time.monotonic()
            outcome, state = agent.run(goal, run_id=run_id, resume=resume_data)
            # A repeat after a resume is a fresh run of the same goal: the
            # checkpoint was spent on the first iteration.
            resume_data, run_id = None, ""
            elapsed = time.monotonic() - started

            colour = out.green if outcome == "success" else (
                out.yellow if outcome == "needs_user" else out.red)
            spent = llm.ledger.total_usd - spent_before
            tilde = "~" if llm.ledger.estimated else ""
            if state.scratchpad:
                out.say()
                out.say(out.bold("  ── Collected Data ──"))
                out.say()
                for line in state.scratchpad.plain().splitlines():
                    out.say(f"  {line}")
            # Last, under the data it was drawn from, because this is the line
            # the person came for -- see `_result_block`.
            _result_block(out, outcome, state.result, state.evidence,
                          progress=(state.progress_log or [""])[0]
                                   or state.last_progress,
                          problem=state.last_failure)
            out.say()
            out.say(f"  {colour(outcome.upper())}  "
                    f"{state.step} steps, {state.llm_calls} LLM calls, "
                    f"{tilde}${spent:.4f}, {elapsed:.1f}s")
            run_path = runlog.run_dir(cfg, state.run_id)
            # Said on every run, not just a failing one: the run you want the
            # trace of is rarely the one you thought to note the id of.
            out.say(out.dim(f"  trace: {run_path} "
                            f"(events.jsonl, {runlog.LOG_NAME}, step prompts)"))
            if cfg.skills.enabled and cfg.skills.learn_after_run:
                # Inside the run's own log: this spends a call and rewrites the
                # file the next run obeys, on this run's behalf, and it happens
                # after the loop has closed its log.
                with runlog.capture(run_path):
                    trace.finish(outcome, state)
                    _learn(out, trace.app_traces(), llm, cfg, goal)
            if outcome != "success":
                exit_code = 1
            if outcome in ("aborted", "needs_user"):
                break
        if iteration > 1:
            out.say(out.dim(f"  session total: {tilde}${llm.ledger.total_usd:.4f} "
                            f"over {iteration} iteration(s)"))
    return exit_code








# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------

def _watch_banner(out: Out, cfg, goal: str, policy: str, ledger) -> None:
    """Say exactly what is about to happen, in the loudest terms available.

    A watch is unattended and it sends messages to real people. The one thing
    nobody should ever have to guess is whether this invocation is going to put
    words in somebody's inbox, so that line is first, coloured, and unambiguous.
    """
    w = cfg.watch
    out.say()
    out.say(f"  {out.bold('WATCH')}  {goal}")
    if w.draft:
        out.say(f"  {out.green('DRAFT MODE')} -- replies are composed and "
                f"recorded, and never sent.")
    else:
        out.say(f"  {out.red('LIVE')} -- replies WILL be sent to real people "
                f"from this device.")
    out.say(out.dim(f"  policy: {w.policy} ({len(policy)} chars)"))
    out.say(out.dim(f"  ledger: {ledger.path} "
                    f"({len(ledger)} repl(ies) already recorded)"))
    out.say(out.dim(
        f"  every {w.interval_s:g}s | <={w.max_steps} steps/pass | "
        f"<={w.max_replies_per_hour}/h | "
        f"<={w.max_replies_per_thread_per_hour}/conversation/h | "
        f"{w.thread_cooldown_s:g}s cooldown | "
        f"fail_{'closed' if w.fail_closed else 'OPEN'}"))
    if w.sweep_s > 0:
        # Said on its own line because it changes what the loop costs: without
        # it a quiet app spends nothing, with it a pass runs on the clock
        # whether or not anything arrived.
        out.say(out.dim(f"  sweep: a pass every {w.sweep_s:g}s even when "
                        f"nothing has changed"))
    if w.max_usd_per_hour:
        out.say(out.dim(f"  spend ceiling: ${w.max_usd_per_hour:g}/h"))
    out.say(out.dim("  Ctrl-C to stop"))
    out.say()


#: Actions one watch keeps in its accumulated trace. A watch runs for days and
#: repeats the same handful of gestures every pass, so the tail is representative
#: and an uncapped list is a leak. The screens are deduped and complete
#: regardless, and they are what carries the coverage.
WATCH_TRACE_ACTIONS = 400


def cmd_watch(args) -> int:
    from . import skills as skillmod
    from .device import Device
    from .ledger import ReplyLedger
    from .llm import LLMClient
    from .memory import Memory
    from .watch import Watch, load_policy

    out = Out()
    cfg = build_config(args)

    goal = args.goal or ""
    if not goal:
        out.bad("no goal given -- say what to watch, e.g. "
                "\"watch my instagram direct messages\"")
        return 1
    if not cfg.llm.model:
        out.bad("no model chosen. Run `adbagent models` and pass --model.")
        return 1
    if not cfg.watch.policy:
        out.bad("a watch needs --policy FILE: the instructions that decide what "
                "gets replied to and what it says. There is no default -- a "
                "default policy is one nobody wrote.")
        return 1
    try:
        policy = load_policy(cfg.watch.policy)
    except (OSError, ValueError) as exc:
        out.bad(str(exc))
        return 1

    # A watch is unattended by definition: it runs for days with nobody at the
    # terminal. Left as it comes, `safety.confirm` would reach `input()` the
    # first time the agent touched a destructive control and block the loop
    # forever -- and under the web UI there is not even a tty to block on.
    # Refusing is the only answer that keeps a watch watching. `allow_destructive`
    # in config still wins, for anyone who has decided otherwise.
    if not cfg.safety.allow_destructive and not cfg.safety.unattended:
        cfg.safety.unattended = True

    _ensure_device(args, cfg, out)

    ledger = ReplyLedger(cfg.watch.ledger)
    _watch_banner(out, cfg, goal, policy, ledger)

    # Per-step reporting only under -v. One line per pass is what a loop meant to
    # run for days should print; the full step trace is megabytes by morning.
    reporter = _live_reporter(out, max_steps=cfg.watch.max_steps) \
        if args.verbose else None

    with Device(cfg, args.device or "") as dev, Memory(cfg) as mem:
        # One client for the whole watch, so the rolling ceilings and the ledger
        # both see the session rather than a single pass.
        llm = LLMClient(cfg, run_id=f"watch-{int(time.time())}")
        # One trace across every pass, folded into the app's skill once when the
        # watch stops -- not per pass, which would rewrite the file the next pass
        # obeys every 45 seconds, mostly from passes that did nothing. Fifty
        # passes over an inbox and its threads is a far better tour of the app
        # than any single pass, so the trade is all upside.
        trace = skillmod.TraceCollector(
            dev, skillmod.AppTrace(tasks=goal), on_event=reporter,
            max_actions=WATCH_TRACE_ACTIONS)
        watch = Watch(dev, mem, llm, cfg, policy=policy, ledger=ledger,
                      say=out.say, on_event=trace)
        try:
            watch.run(goal)
        except KeyboardInterrupt:
            out.say()
            out.say("  stopped")
        out.say(out.dim(f"  {watch.status()}"))
        if cfg.skills.enabled and cfg.skills.learn_after_run:
            # After the status line, because it spends a call and can take a
            # minute: the numbers should already be on screen when it starts.
            trace.finish("stopped", watch.last_state)
            _learn(out, trace.app_traces(), llm, cfg, goal)
    return 0


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
        # How the reasoning budget was actually spent, when it was capped at all.
        # A run that escalated on every turn has a policy that is not helping;
        # one that never escalated may have been shallow through a failure.
        efforts = [e.get("effort") for e in rows if e.get("effort")]
        shallow = sum(1 for e in efforts if e == "none")
        if efforts:
            out.say(out.dim(
                f"  thinking depth  {len(efforts) - shallow} of {len(efforts)} "
                f"turn(s) escalated"
                + (f", {shallow} at the floor" if shallow else "")))
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
        if think_share >= 50 and not efforts:
            advice.append("Reasoning tokens dominate output, so they dominate "
                          "latency: try setting llm.reasoning_effort.")
        elif think_share >= 50 and efforts and len(efforts) - shallow > len(efforts) / 2:
            # Already capped, and escalating anyway on most turns -- so the
            # thinking is not the setting's fault, it is the run's difficulty.
            advice.append("Most turns escalated, so the depth cap is buying "
                          "little: look at why they were graded hard.")
        if total_prompt and hit_rate < 40:
            advice.append("Low prompt-cache hit rate: something near the top of "
                          "the prompt is changing every turn.")

    for line in advice:
        out.say(out.dim(f"  {line}"))


#: Warnings shown inline by `report` before it stops listing them. Enough to see
#: what kind of trouble a run had; the file has the rest, in context.
_MAX_PROBLEMS = 8


def _log_summary(out: Out, run_dir: Path) -> None:
    """Point at the run log, and say up front whether anything went wrong in it.

    A trace nobody can find is a trace nobody reads, and the warnings that
    explain a bad run -- the settle timeout, the retargeted swipe, the dropped
    request field -- are a few dozen lines among many thousands.
    """
    path = runlog.log_path(run_dir)
    if not path.is_file():
        return
    out.say()
    size = path.stat().st_size
    shown = f"{size / 1024:.0f} KB" if size >= 1024 else f"{size} B"
    found = runlog.problems(path)
    out.say(f"  {out.bold('── Run log ──')}  {path} ({shown}, "
            f"{len(found)} warning(s) or worse)")
    for line in found[:_MAX_PROBLEMS]:
        out.say(out.dim(f"  {line}"))
    if len(found) > _MAX_PROBLEMS:
        out.say(out.dim(f"  ... {len(found) - _MAX_PROBLEMS} more in the file"))


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
    # The LAST run_end: a run continued with --resume has one per sitting, and
    # only the final one carries the outcome the run actually reached.
    end = next((e for e in reversed(events) if e["kind"] == "run_end"), {})

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
        elif event["kind"] in ("dismiss", "refused", "scroll_refused",
                               "loop_break", "sensitive",
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
    _log_summary(out, events_file.parent)
    if end:
        # Last, next to the outcome line, so the two things a person opens a
        # report for are together at the bottom rather than either side of the
        # token arithmetic. Runs recorded before `run_end` carried the answer
        # fall back to the terminal action in the feed, where it always was.
        _result_block(out, end.get("outcome", "?"),
                      end.get("result") or _terminal_text(events),
                      end.get("evidence", ""),
                      progress=_last_progress(events),
                      problem=_last_problem(events))
    out.say()
    if end:
        out.say(f"  {end.get('outcome', '?').upper()}: {end.get('steps')} steps, "
                f"{end.get('llm_calls')} LLM calls, ${end.get('usd', 0):.4f}")
    return 0


def _terminal_text(events: List[Dict[str, Any]]) -> str:
    """The text of the action that ended a run, read back out of its feed."""
    from .actions import TERMINAL_ACTIONS

    for event in reversed(events):
        action = event.get("action") or {}
        if event.get("kind") == "decide" \
                and action.get("action") in TERMINAL_ACTIONS:
            return " ".join((action.get("text") or "").split())
    return ""


def _last_progress(events: List[Dict[str, Any]]) -> str:
    """The last note the model wrote about where it had got to."""
    for event in reversed(events):
        progress = (event.get("action") or {}).get("progress")
        if isinstance(progress, str) and progress.strip():
            return progress
    return ""


def _last_problem(events: List[Dict[str, Any]]) -> str:
    """The last thing that went wrong, for a run that stopped without saying."""
    for event in reversed(events):
        kind = event.get("kind")
        if kind == "gave_up" and event.get("reason"):
            return str(event["reason"])
        if kind == "error" and event.get("error"):
            return str(event["error"])
        if kind == "judge" and not event.get("satisfied") and event.get("evidence"):
            return f"a claim of completion was rejected: {event['evidence']}"
    return ""


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
        from . import skills as skillmod
        from .memory import Memory

        # The positional form is the documented one -- `skills generate whatsapp`
        # -- and `--app` stays as an alias for anyone who already types it.
        app_target = (getattr(args, "target", "") or getattr(args, "app", "") or "").strip()
        user_tasks = (getattr(args, "tasks", "") or "").strip()

        # The whole command is skill work, so `llm.model_skill` drives it: the
        # tour as well as the write-up. Done before the check below because an
        # empty `llm.model` is no obstacle when `llm.model_skill` is set.
        blinded = skillmod.use_skill_model(cfg)

        if not cfg.llm.model:
            out.bad("no model chosen. Run `adbagent models` and pass --model.")
            return 1
        if not cfg.api_key():
            out.bad(f"no API key: exploring the app needs one. Set llm.api_key in config.json or ${cfg.llm.api_key_env}.")
            return 1

        # An open-ended "look around" must not inherit a step budget sized for a
        # collection run; --max-steps still wins when it is given.
        if getattr(args, "max_steps", None) is None:
            cfg.run.max_steps = skillmod.DEFAULT_EXPLORE_STEPS

        # A tour is never the moment to confirm an irreversible action. The brief
        # tells it not to reach for one, and a prompt half way through either
        # stalls an unattended run or asks about a control nobody meant to press,
        # so it refuses instead. `safety.allow_destructive` still overrides.
        if not cfg.safety.allow_destructive:
            cfg.safety.unattended = True

        _ensure_device(args, cfg, out)

        out.say(out.bold(f"  Exploring {app_target or 'the app your tasks name'} live on the phone"))
        out.say(out.dim(f"  tasks:     {user_tasks or skillmod.DEFAULT_EXPLORE_TASKS}"))
        out.say(out.dim(f"  budget:    up to {cfg.run.max_steps} steps, ${cfg.safety.budget_usd:.2f}"))
        out.say(out.dim(f"  model:     {cfg.llm.skill()} explores and writes the skill"))
        out.say(out.dim(f"  screens:   {cfg.llm.skill_image()} reads them, "
                        "exploring and writing"))
        if blinded:
            out.say(out.dim("             (vision_in_decider off for this run: a skill "
                            "model that cannot see would fail every step it was "
                            "handed a screenshot)"))
        elif cfg.llm.decider_sees():
            out.say(out.dim("             (one model for both, so it explores looking "
                            "at the screenshots itself -- no separate vision call)"))
        out.say()

        llm = LLMClient(cfg, run_id=f"skill-{int(time.time())}")
        try:
            with Device(cfg, getattr(args, "device", "") or "") as dev, Memory(cfg) as mem:
                exp = skillmod.explore_app(
                    dev, mem, llm, cfg, query=app_target, tasks=user_tasks,
                    on_event=_live_reporter(out, max_steps=cfg.run.max_steps))
        except skillmod.ExplorationBlocked as exc:
            out.bad(f"nothing was explored: {exc}")
            return 1

        colour = out.green if exp.outcome == "success" else out.yellow
        chose = {"tasks": " (picked from your --tasks)",
                 "foreground": " (the app that was in front)"}.get(exp.chosen_by, "")
        out.say()
        out.say(f"  explored {out.bold(exp.package)}{chose}: {exp.steps} steps, "
                f"{len(exp.screens)} distinct screens, {len(exp.screenshots)} screenshots, "
                f"{colour(exp.outcome.upper())}, ~${llm.ledger.total_usd:.4f}")
        if exp.notes:
            out.say()
            out.say(out.bold("  ── What it found ──"))
            out.say()
            for line in exp.notes.splitlines():
                out.say(f"  {line}")
        if not exp.looked_around:
            out.warn("the run never left the first screen, so the skill below rests on "
                     "one screen. Give --tasks something concrete to do in the app.")
        out.say()

        # Synthesis belongs in the tour's own log: it is one more call spent on
        # that run, and when the skill comes back thin the reason is in the
        # exploration above it rather than anywhere else.
        trace_dir = runlog.run_dir(cfg, exp.run_id) if exp.run_id else None
        with runlog.capture(trace_dir):
            skill = SkillGenerator(registry).generate_from_exploration(
                app_target or exp.package, exp.tasks, exp.screens, exp.actions, llm,
                screenshots=exp.screenshots, package=exp.package,
                notes=exp.notes, outcome=exp.outcome,
                history=skillmod.run_history(cfg, exp.package))
        if trace_dir:
            out.say(out.dim(f"  trace: {trace_dir}"))
        out.ok(f"saved skill '{skill.name}' to {registry.path_for(skill)}")
        out.say()
        out.say(skill.to_markdown())
        return 0

    out.bad(f"Unknown skills action '{action}'. Use list, view, create, or generate.")
    return 1


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------

def _ui_live_reload(args, out: Out):
    """The file watcher behind the UI's live reload, or None when it is off.

    On by default in a source checkout and off in an installed copy, because
    that is where the answer is obvious either way: files in the repo are being
    edited, files in site-packages are not, and re-execing a server nobody is
    working on is a surprise rather than a convenience. `--reload` and
    `--no-reload` say so explicitly.

    Started here rather than by the server: the phone-busy question it has to
    ask, and the restart it has to perform, both live on this side.
    """
    from .web.reload import for_ui, in_source_checkout

    package_dir = Path(__file__).resolve().parent
    want = getattr(args, "reload", None)
    if want is None:
        want = in_source_checkout(package_dir)
    if not want:
        return None

    from .config import load_config

    loaded = load_config(getattr(args, "config", None))
    cfg = loaded.config
    policy = (cfg.watch.policy or "").strip()
    reloader = for_ui(
        package_dir,
        # A config file that does not exist yet is still worth watching: the UI
        # writes exactly this path the first time the config form is saved.
        config_path=Path(loaded.path) if loaded.path else Path.cwd() / "config.json",
        skills_dir=Path(cfg.skills.skills_dir).expanduser(),
        policy_path=Path(policy).expanduser() if policy else None)
    out.say(out.dim("  live reload: code, static, config, skills, policy "
                    "(--no-reload to turn it off)"))
    return reloader


def _reexec() -> int:
    """Start this same command again, in place of this process.

    Replacing the process is the only way to pick up an edited module: this one
    imported them all at startup and will not import them again. argv is reused
    verbatim so the new server is the one that was asked for -- same port, same
    config, same artifacts directory.
    """
    argv = list(sys.argv)
    if not sys.executable:  # an embedded interpreter has nothing to re-run
        print("  cannot restart: no interpreter to re-run", file=sys.stderr)
        return 1
    if Path(argv[0]).name == "__main__.py":
        argv = ["-m", "adbagent", *argv[1:]]   # started as `python -m adbagent`
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *argv])
    return 0  # not reached


def cmd_ui(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: the web UI needs uvicorn -- pip install uvicorn fastapi",
              file=sys.stderr)
        return 1

    from .web.server import create_app

    out = Out()
    kwargs: Dict[str, Any] = {}
    if getattr(args, "artifacts_dir", None):
        kwargs["artifacts_dir"] = args.artifacts_dir
    if getattr(args, "config", None):
        kwargs["config_path"] = args.config
    reloader = _ui_live_reload(args, out)
    if reloader is not None:
        kwargs["live_reload"] = reloader
    app = create_app(**kwargs)
    out.say(f"  adbagent ui on http://{args.host}:{args.port}")
    # Flushed rather than left to the buffer. Piped into a supervisor rather
    # than a terminal, stdout is block-buffered, so the line saying the server
    # is up sits unread until something else fills the buffer -- and after a
    # live-reload restart nothing does, which reads as a server that never
    # came back.
    with contextlib.suppress(OSError):
        sys.stdout.flush()

    # Run the server by hand rather than through `uvicorn.run` so the interrupt
    # can be seen as it lands. The live views are Server-Sent Events streams that
    # follow a run for as long as it lasts -- for a watch, days -- and uvicorn
    # waits for in-flight requests before it sends the lifespan shutdown. Left to
    # itself it therefore waits forever (`timeout_graceful_shutdown` is None by
    # default), gives up on the connection, and prints a CancelledError raised
    # out of the middle of a StreamingResponse. Telling the app at the top of the
    # shutdown lets those streams end themselves, so the wait is a moment and
    # there is nothing to cancel.
    config = uvicorn.Config(app, host=args.host, port=args.port,
                            log_level="warning",
                            # A backstop for a shutdown that never went through
                            # the handler below, so it cannot hang indefinitely.
                            timeout_graceful_shutdown=20)
    server = uvicorn.Server(config)
    signalled = server.handle_exit

    def leave() -> None:
        leaving = getattr(app.state, "shutting_down", None)
        if leaving is not None:
            leaving.set()

    def handle_exit(sig, frame):  # type: ignore[no-untyped-def]
        leave()
        signalled(sig, frame)

    server.handle_exit = handle_exit  # type: ignore[method-assign]

    restarting = threading.Event()
    if reloader is not None:
        # A restart while an agent holds the phone would orphan it: the run and
        # watch children are in their own process groups on purpose, so they
        # outlive this process, and the server that comes back has no handle on
        # the thing still tapping the screen. So the reloader asks first, and
        # waits -- the browser is told it is waiting, and what for.
        reloader.busy = app.state.phone_busy

        def restart() -> None:
            restarting.set()
            leave()             # so the live streams end themselves, as on Ctrl+C
            server.should_exit = True

        reloader.on_restart = restart
        reloader.start()

    try:
        server.run()
    finally:
        if reloader is not None:
            reloader.stop()
    if restarting.is_set():
        out.say(out.dim("  code changed — restarting"))
        return _reexec()
    return 0


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
    parser.add_argument("--model-skill-image", dest="model_skill_image",
                        help="multimodal model for the screenshot pass of skill "
                             "synthesis (falls back to --model-image)")
    parser.add_argument("--skills-dir", dest="skills_dir",
                        help="directory for app skills (default ./skills)")
    parser.add_argument("--rpm", type=int, help="client-side request throttle")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int,
                        help="max completion tokens for LLM calls")
    parser.add_argument("--max-tokens-image", dest="max_tokens_image", type=int,
                        help="max completion tokens for image model calls "
                             "(falls back to --max-tokens)")
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
    p.add_argument("goal", nargs="?",
                   help="what to accomplish, in plain language "
                        "(optional with --resume: the checkpoint's goal is kept)")
    p.add_argument("--resume", nargs="?", const="latest", metavar="RUN",
                   help="continue a failed or interrupted run where it stopped: "
                        "a run id or directory, or the most recent resumable "
                        "run when no value is given")
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
    p.add_argument("--no-learn", dest="learn_after_run", action="store_false",
                   default=None,
                   help="do not update the app's skill from what this run learned")
    p.add_argument("--artifacts-dir", dest="artifacts_dir")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_run)



    p = sub.add_parser("watch",
                       help="monitor an app and reply, continuously")
    p.add_argument("goal", nargs="?",
                   help="what to watch, in plain language, e.g. "
                        "\"watch my instagram direct messages\"")
    p.add_argument("--policy", dest="watch_policy", metavar="FILE",
                   help="required: file holding the reply instructions -- what "
                        "to answer, what to ignore, and what to say")
    p.add_argument("--draft", dest="watch_draft", action="store_true",
                   default=None,
                   help="compose and record replies but never send them; run "
                        "this first whenever the policy has changed")
    p.add_argument("--interval", dest="watch_interval", type=float,
                   metavar="SECONDS",
                   help="how often to look when nothing has changed "
                        "(default 45)")
    p.add_argument("--sweep", dest="watch_sweep", type=float,
                   metavar="SECONDS",
                   help="run a pass this often even when nothing on screen has "
                        "changed. For goals whose work does not announce itself "
                        "-- a feed to work through, a queue to drain, a periodic "
                        "check. Off by default, which watches for new messages "
                        "and spends nothing while there are none")
    p.add_argument("--steps-per-pass", dest="watch_max_steps", type=int,
                   metavar="N",
                   help="step budget for one pass over the inbox (default 25)")
    p.add_argument("--replies-per-hour", dest="watch_replies_per_hour",
                   type=int, metavar="N",
                   help="circuit breaker on total replies (default 12)")
    p.add_argument("--replies-per-conversation",
                   dest="watch_replies_per_thread", type=int, metavar="N",
                   help="circuit breaker per conversation, per hour (default 2)")
    p.add_argument("--cooldown", dest="watch_cooldown", type=float,
                   metavar="SECONDS",
                   help="minimum gap between two replies to the same "
                        "conversation (default 600)")
    p.add_argument("--ledger", dest="watch_ledger", metavar="FILE",
                   help="where the record of sent replies lives "
                        "(default watch-replies.jsonl). Deleting it allows "
                        "every conversation to be answered again")
    p.add_argument("--usd-per-hour", dest="watch_usd_per_hour", type=float,
                   metavar="USD",
                   help="pause the loop when spend in the last hour reaches "
                        "this (default 0, meaning no ceiling)")
    p.add_argument("--no-learn", dest="learn_after_run", action="store_false",
                   default=None,
                   help="do not update the app's skill when the watch stops")
    p.add_argument("--fail-open", dest="watch_fail_closed",
                   action="store_false", default=None,
                   help="send even when the conversation on screen cannot be "
                        "identified. Off by default, and off is the safe setting: "
                        "an unidentifiable conversation is one where a duplicate "
                        "cannot be ruled out")
    _add_common(p)
    _add_device(p)
    p.set_defaults(func=cmd_watch)

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

    from .skills import DEFAULT_EXPLORE_STEPS

    p = sub.add_parser("ui", help="serve the web UI")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    p.add_argument("--artifacts-dir", dest="artifacts_dir",
                   help="directory of recorded runs (default: config's run.artifacts_dir)")
    # Tri-state on purpose: unset means "decide by whether this is a checkout".
    p.add_argument("--reload", dest="reload", action="store_true", default=None,
                   help="apply edits to code, static files, config, skills and "
                        "policy without restarting by hand (default: on when "
                        "running from a source checkout)")
    p.add_argument("--no-reload", dest="reload", action="store_false",
                   help="never watch files or restart on a change")
    _add_common(p)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("skills", help="manage app skills (list, view, create, generate)")
    p.add_argument("skills_action", nargs="?", choices=["list", "view", "create", "generate"],
                   default="list", help="action to perform (default: list)")
    p.add_argument("target", nargs="?",
                   help="app name or package to view, create, or explore. "
                        "'generate' with no app explores whatever is on screen")
    p.add_argument("--app", help="alias for the positional app argument")
    p.add_argument("--tasks", help="what to do in the app while exploring "
                                   "(default: tour its main screens)")
    p.add_argument("--max-steps", dest="max_steps", type=int,
                   help=f"step budget for the exploration (default {DEFAULT_EXPLORE_STEPS})")
    p.add_argument("--budget-usd", dest="budget_usd", type=float)
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


def _catch_windows_break() -> None:
    """Make CTRL_BREAK raise KeyboardInterrupt, as SIGINT does everywhere else.

    The web UI stops a child by signalling it, and on Windows that signal has to
    be CTRL_BREAK_EVENT: a child spawned into its own process group -- which it
    must be, or stopping a run would take the server down with it -- cannot be
    sent anything else. Left alone, SIGBREAK's default action is to terminate the
    process where it stands. Nothing below would run: the phone would keep the
    keyboard, animations, rotation and screen timeout the agent changed, and a
    watch would never write up what its passes learned.

    Everything that makes a stop orderly hangs off `KeyboardInterrupt`, so the
    fix is to raise one -- with the handler Python already uses for SIGINT.
    """
    if sys.platform != "win32" or not hasattr(signal, "SIGBREAK"):
        return
    try:
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    except (ValueError, OSError):
        # Not the main thread, or a platform that will not have it. A stop is
        # still a stop; it is just the abrupt kind again.
        log.debug("could not install a SIGBREAK handler", exc_info=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", 0))
    _catch_windows_break()
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
