"""Configuration: dataclasses plus a loader.

Precedence, lowest to highest: built-in defaults < environment < config.json < CLI flags.

The API key may be set in config.json (``llm.api_key``) or read from the
environment variable named by ``llm.api_key_env``. config.json is gitignored,
so the key is safe there; the env var remains as a fallback. The key is redacted
in ``to_dict()`` so it can never leak into a run artifact or the web UI.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_NAMES = ("config.json", "adbagent.json")


def same_model(a: str, b: str) -> bool:
    """True when two model settings name one model.

    Compares on the last segment, because the same id has two accepted forms --
    ``kimi-k3`` and ``accounts/fireworks/models/kimi-k3``, which ``llm.qualify``
    treats as one -- and a config that mixes them is still naming one model.

    Empty is never a match. An unset setting falls back to another one, and a
    fallback is not evidence of anything: ``model_image`` left empty resolves to
    ``model`` in every config there is, text-only ones included.
    """
    return bool(a) and bool(b) and a.rsplit("/", 1)[-1] == b.rsplit("/", 1)[-1]


@dataclass
class LLMConfig:
    provider: str = "fireworks"
    model: str = ""
    #: Cheaper model used for bounded side-calls (JSON repair, completion judging).
    #: Falls back to `model` when empty.
    model_small: str = ""
    #: Multimodal model used when a screenshot is provided.
    #: Falls back to `model` when empty.
    model_image: str = ""
    #: Dedicated model used for app skill generation and exploration.
    #: Falls back to `model` when empty.
    model_skill: str = ""
    #: Multimodal model for the screenshot pass of skill synthesis. The model
    #: that writes a skill well is not always one that can see; a text-only
    #: `model_skill` handed an image part fails the whole call. Falls back to
    #: `model_image`, then `model`, when empty.
    model_skill_image: str = ""
    temperature: float = 0.0
    #: Output ceiling for every call the agent makes -- deciding, judging,
    #: describing a screenshot, reading one item. The single source: no call site
    #: clamps it further, so a truncated reply means raise this, not hunt for a
    #: literal. Reasoning models spend most of their output thinking, so a
    #: ceiling that fits a bare answer truncates a thought.
    max_tokens: int = 10000
    #: Output ceiling for image model calls (analyze_image, read_item). The
    #: image model is often a different family with a different output limit,
    #: and a single ceiling that fits a reasoning model's chain of thought can
    #: truncate a vision transcription. Set to 0 to fall back to `max_tokens`.
    max_tokens_image: int = 4096
    #: Client-side request throttle. Fireworks free accounts are capped at 10 RPM;
    #: paid accounts go to 6000. 120 is a safe default for a paid account.
    rpm: int = 120
    base_url: str = ""
    #: The API key itself. config.json is gitignored, so the key is safe here;
    #: when set, it takes precedence over the env var below. Empty falls back to
    #: the environment variable named by ``api_key_env``.
    api_key: str = ""
    api_key_env: str = "FIREWORKS_API_KEY"
    #: Seconds. Agentic workloads are long; Fireworks recommends 5-30 min.
    read_timeout: float = 300.0
    #: How many times to retry a failed LLM call before giving up.
    max_retries: int = 5
    #: Service tier for request (e.g. "priority").
    service_tier: str = ""
    #: True when `model` itself accepts images. The screenshot then goes straight
    #: to the deciding call and the separate `model_image` description is skipped
    #: -- one round trip per screenshot turn instead of two. Left off by default
    #: because a text-only model given an image part fails the whole call, and it
    #: is still not consulted at run time: the run would discover it as a 400 on
    #: the first screenshot turn, which may be ninety steps in.
    #:
    #: Only ever *asserts* the saving. Naming one model for both `model` and
    #: `model_image` asserts it too, without being asked -- see `decider_sees`,
    #: which is what the run reads. Nothing reads this field directly.
    #:
    #: `adbagent doctor` now settles it up front, for this and for `model_image`,
    #: against the catalogue's `supportsImageInput` -- see `cli._check_vision`.
    #: Check it after changing any of the three model settings.
    vision_in_decider: bool = False
    #: How hard to think on a routine turn: "", "none", "low", "medium", "high".
    #: Empty leaves the model's own default alone and switches the whole feature
    #: off. This is the largest single lever on latency there is -- a reasoning
    #: model spends ~4,200 of its ~4,400 output tokens thinking, and on a step
    #: whose answer is "swipe left again" almost all of that is waste.
    reasoning_effort: str = ""
    #: How hard to think on a turn the agent is struggling with. Falls back to
    #: `reasoning_effort`. The point of the pair is that "think less" is only safe
    #: when the turn is easy, and the loop already knows when it is not.
    reasoning_effort_hard: str = "high"
    #: How to say it on the wire: "auto" picks by model family, "effort" sends
    #: `reasoning_effort`, "thinking" sends `chat_template_kwargs`, "off" sends
    #: nothing. See `llm.reasoning_body`.
    reasoning_style: str = "auto"

    #: Calls that read a frame and report what is on it. Transcription, not
    #: reasoning: "what does this scale read" and "where is the send button" have
    #: no chain of thought worth paying for, and the answer is four short strings
    #: or a coordinate pair.
    VISION_PURPOSES = ("analyze_image", "read_item", "locate")

    def effort_for(self, purpose: str = "decide", hard: bool = False) -> str:
        """Reasoning depth for one call. Empty means "send nothing".

        Vision calls are pinned to the floor, and pinned *before* the switch is
        consulted, because "send nothing" is not the same as "do not think". A
        hybrid model thinks by default, so leaving its own default alone buys a
        chain of thought nobody asked for -- and the ratio it buys is not
        marginal. In ``runs/a7ef4e0e45e9`` (`reasoning_effort` unset, so this
        method returned "" for every purpose and the pin below never ran)
        `analyze_image` spent 6,165 characters of thinking per call to fill four
        fields with 200, 38 chars of thought per char of answer; `locate` ran
        35:1 for a coordinate pair. That was 95% of the run's `analyze_image`
        output tokens and 153 of its 739 seconds.

        Reaching it through the switch was not an option: `reasoning_effort` is
        one setting for every purpose, so the config could not say "vision does
        not think" without also opting `decide`, `judge` and `goal_check` into
        depths they had not asked for -- `judge` is called with ``hard=True``
        and would have gone straight to "high".

        Costs nothing on a model that does not reason: `llm.reasoning_body`
        sends a field only for a family known to take one, and `_post` drops it
        for good if the provider rejects it anyway.
        """
        if purpose in self.VISION_PURPOSES:
            return "none"
        if not self.reasoning_effort:
            return ""
        if hard:
            return self.reasoning_effort_hard or self.reasoning_effort
        return self.reasoning_effort

    def small(self) -> str:
        return self.model_small or self.model

    def image(self) -> str:
        return self.model_image or self.model

    def image_max_tokens(self) -> int:
        return self.max_tokens_image or self.max_tokens

    def skill(self) -> str:
        return self.model_skill or self.model

    def skill_image(self) -> str:
        return self.model_skill_image or self.model_image or self.model

    def decider_sees(self) -> bool:
        """True when the screenshot rides in the deciding call itself.

        `vision_in_decider` says so outright. Naming one model for `model` and
        for `model_image` says the same thing without being asked: the frame was
        going to that model either way, so describing it first spends a whole
        round trip having a model tell itself what it is already looking at.
        Setting the pair and then hunting for the checkbox that makes it pay off
        is a tax on knowing the internals, and forgetting the checkbox costs a
        call on every screenshot turn of every run, silently.

        Only an explicit `model_image` counts, not `image()`: an empty one falls
        back to `model`, which would read every text-only config as a matching
        pair and fail its first screenshot turn outright.
        """
        return bool(self.vision_in_decider) or same_model(self.model, self.model_image)


@dataclass
class DeviceConfig:
    serial: str = ""
    #: u2's own default is 50, which silently truncates deep Compose/RN trees.
    max_depth: int = 40
    #: Drop nodes not marked important-for-accessibility. Much smaller XML.
    compressed: bool = True
    #: Hard ceiling on one adaptive settle. It bounds how long the loop will keep
    #: re-dumping; it is *not* the thing that decides a screen has settled -- see
    #: `settle_quiet_s`.
    #:
    #: This was 2.0, which is smaller than a single observation over wireless adb
    #: (~1.2s for the dump alone), so the comparison the settle loop exists to
    #: make was reached at most once and usually not at all: 95 of ~100 settling
    #: observations across ``runs/`` logged "screen never settled". Raising it
    #: costs nothing on a screen that is already still, because that screen
    #: returns on its first comparison.
    settle_budget_s: float = 6.0
    settle_interval_s: float = 0.18
    #: How long two dumps must agree before the screen counts as settled.
    #:
    #: Equality alone cannot answer the question. A screen that has drawn its
    #: chrome and not yet its content is *stably* half-rendered, so two dumps
    #: 0.18s apart agree on a frame that is not the frame -- which is why the
    #: model spent 13 of 103 turns (12.6%, ~254s across ``runs/``) choosing a
    #: `wait` action to re-read a screen the harness had already declared
    #: settled. Agreement across a wider window is much stronger evidence, and
    #: measuring it in wall clock makes it self-calibrating: over a slow link the
    #: dumps themselves span the window and it costs nothing, while over a fast
    #: one a few more cheap dumps are taken.
    settle_quiet_s: float = 0.5
    #: How long `open_app` waits for the package to actually reach the foreground.
    #: `app_start` returns before the window exists, and a cold start on a loaded
    #: phone can take several seconds; observing before then reads the launch, not
    #: the app.
    launch_timeout_s: float = 8.0
    launch_poll_s: float = 0.25
    #: Hard ceiling on any single device round trip. u2's own `timeout=` argument
    #: is inert -- the underlying socket defaults to 600s -- so we enforce our own.
    watchdog_s: float = 60.0
    #: Zero the animator scales for the duration of the run (restored on exit).
    disable_animations: bool = True
    #: Disable auto-rotate and force portrait mode for the duration of the run (restored on exit).
    disable_auto_rotate: bool = True


@dataclass
class MemoryConfig:
    db: str = "memory.db"


@dataclass
class SafetyConfig:
    # A `package_allowlist` used to live here, along with an `allowed_packages()`
    # that composed it with a list of system packages. Nothing ever consulted
    # either, so setting it bought exactly nothing while reading like a sandbox.
    # An inert safety control is worse than an absent one, so it is gone rather
    # than documented. The guards that do work are in `safety.py`, and they key
    # off what is on screen rather than off which package drew it.
    budget_usd: float = 2.0
    #: Skip the interactive confirmation on irreversible actions.
    allow_destructive: bool = False
    #: Never prompt; abort instead of asking. For unattended runs.
    unattended: bool = False


@dataclass
class RunConfig:
    max_steps: int = 60
    max_consecutive_failures: int = 4
    max_wall_clock_s: float = 1800.0
    artifacts_dir: str = "runs"
    #: Force a screenshot on every LLM call (expensive), or never (XML-only).
    always_screenshot: bool = False
    never_screenshot: bool = False
    dry_run: bool = False
    #: Ceiling on the collected-data block in the prompt. The model sends one
    #: record at a time and the harness keeps the union, so this bounds how much
    #: of that union is rendered back -- the oldest records go first, and the
    #: block says how many it left out.
    scratchpad_max_chars: int = 50_000
    #: After the model swipes through a carousel and the item verifiably moves,
    #: keep swiping and reading in code rather than paying a full reasoning turn
    #: to be told "swipe left" again. 71 of 127 steps in the run that motivated
    #: this were exactly that one gesture.
    pager_sweep: bool = True
    #: Items per sweep before control returns to the model. The sweep stops on
    #: its own at an edge, a hidden caption or a full ledger; this is the cap for
    #: when none of those arrive, so an endless feed cannot run away with the run.
    pager_sweep_max: int = 12

    # -- the stall ladder --------------------------------------------------
    #
    # `consecutive_failures` counts actions that did not work, and for most of
    # this project's life it was the only give-up signal there was. It cannot
    # see the dominant failure: an agent whose every action succeeds and whose
    # run goes nowhere. `runs/2521862d7a23` navigated between two screens for
    # twenty steps with `consecutive_failures` pinned at zero.
    #
    # So the loop also counts steps since it last learned anything -- a screen
    # it had not seen, a record written, content that moved, a state it changed
    # (see `Agent._progress_made`) -- and escalates on that count. The tiers go
    # cheap to expensive on purpose: say something, then refuse something, then
    # spend a call rethinking, then stop.
    #
    # Set any of them to 0 to switch that tier off.

    #: Tell the model it is stalling, take a screenshot, and think harder.
    stall_nudge_at: int = 3
    #: Stop asking: mechanically refuse an action already tried on this screen.
    stall_block_at: int = 5
    #: Spend one call on a fresh strategy, from outside the decide history.
    stall_replan_at: int = 8
    #: Give up. The collected data survives -- the CLI prints it, and the
    #: checkpoint keeps it for `--resume`.
    stall_give_up_at: int = 14

    # -- the goal check ----------------------------------------------------
    #
    # The ladder above measures whether the run is getting *anywhere*. It cannot
    # measure whether the run is already *finished*, and nothing else could
    # either: the completion judge is reachable only through a terminal action
    # the model volunteers, and `Oracle` needs a condition supplied at launch.
    # A run whose every action succeeds while the goal has already been met is
    # invisible to every guard in the file -- ``runs/963a4f4ae96c`` answered its
    # goal at step 14, ran 24 more steps and 471s, and was killed by hand.
    #
    # So every `goal_check_every` steps the harness asks a model, in as many
    # words, whether the goal is already satisfied. The call is issued while the
    # loop is blocked on the device anyway, so it costs no wall clock.

    #: Steps between goal checks. 0 switches the check off.
    goal_check_every: int = 5
    #: Consecutive satisfied verdicts before the run is ended. Two, not one: this
    #: is the only guard that ends a run on a model's say-so without the model
    #: having asked to stop, and a single sample of anything is how a run that
    #: still had work to do gets cut off. The second opinion is free -- it lands
    #: in the next step's device round trip.
    goal_check_hits: int = 2


@dataclass
class WatchConfig:
    """Settings for `adbagent watch` -- the unbounded monitor-and-reply loop.

    A watch is not a long run. A run is bounded, may fail, and is over; a watch
    is expected to outlive transient failures and to still be going tomorrow.
    Every ceiling here is therefore per *iteration* or per *rolling hour*, never
    per lifetime, and none of them ends the loop -- they pause it.
    """

    #: Seconds to wait after an iteration that found nothing to do. The device
    #: is dumped once per interval (no LLM call), so this is the resolution at
    #: which a new message is noticed, not a cost dial.
    interval_s: float = 45.0
    #: Run a pass at least this often even when the screen has not changed.
    #: 0 leaves the loop purely reactive.
    #:
    #: The novelty probe answers "did anything arrive?", and for an inbox that is
    #: the whole question -- no new message, nothing to do. It is the wrong
    #: question for a goal whose work is not announced on the screen: a feed with
    #: more items below, a queue to drain a few at a time, anything the operator
    #: wants done on a period. There the last pass's screen is unchanged *and*
    #: there is work, and a purely reactive loop runs once and then sleeps
    #: forever. Which of the two a goal is cannot be read off the app -- it is a
    #: property of what was asked for -- so the operator declares it here.
    sweep_s: float = 0.0
    #: Steps one iteration may spend before it is abandoned and re-anchored. An
    #: iteration is "look at the inbox, handle what is new, come back" -- if that
    #: takes 25 steps something is wrong, and the fix is a fresh iteration rather
    #: than more budget for a confused one.
    max_steps: int = 25
    #: Compose replies and record them, but never tap Send. The first thing to
    #: run when a policy changes: the failure mode becomes a wrong draft in the
    #: log instead of a wrong message in somebody's inbox.
    draft: bool = False
    #: Seconds before the same conversation may be written to again, whatever the
    #: content digests say. The backstop for the one crash window digests cannot
    #: close -- see `ledger`.
    thread_cooldown_s: float = 600.0
    #: Rolling ceilings on sends. These are circuit breakers, not budgets: a loop
    #: that has started replying to everything is the thing they exist to stop.
    max_replies_per_hour: int = 12
    max_replies_per_thread_per_hour: int = 2
    #: The reply ledger. Relative paths are relative to the working directory,
    #: not to the artifacts dir: it must outlive any one run.
    ledger: str = "watch-replies.jsonl"
    #: File holding the reply instructions, injected verbatim into the prompt.
    #: Required by `adbagent watch` -- there is no default policy, because a
    #: default policy is one nobody wrote and everybody would be surprised by.
    #:
    #: One of possibly several: this is the one a bare `adbagent watch` uses and
    #: the one the UI opens on. The rest live in `policies_dir`, and a policy
    #: carries the goal it was written for in its own front matter -- see
    #: `adbagent.policies`.
    policy: str = ""
    #: Where the other policies live. Listed in the UI's policy picker, and what
    #: a bare `--policy hinge` is resolved against. Only ever read: nothing here
    #: decides which policy a watch uses -- `policy` above and `--policy` do.
    policies_dir: str = "policies"
    #: Refuse to send when the conversation on screen cannot be identified.
    #: Leaving this on trades a missed reply for never sending blind; turning it
    #: off is only sensible while debugging an app whose thread title the parser
    #: cannot see.
    fail_closed: bool = True
    #: Rolling spend ceiling. Unlike `safety.budget_usd`, hitting this pauses the
    #: watch until the window clears rather than ending it. 0 switches it off.
    max_usd_per_hour: float = 0.0
    #: Backoff after an iteration that failed. Doubles per consecutive failure up
    #: to the cap, and resets on the first iteration that gets through -- so a
    #: flapping device costs a slow poll rather than a dead watch.
    backoff_initial_s: float = 30.0
    backoff_max_s: float = 900.0


@dataclass
class SkillsConfig:
    enabled: bool = True
    skills_dir: str = "skills"
    #: After each run, fold what it learned about the app back into that app's
    #: skill. One extra call per run, on `llm.model_skill`, and the reason the
    #: agent gets better at an app the more it is driven there. `--no-learn`
    #: turns it off for a run whose trace is not worth keeping.
    learn_after_run: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    run: RunConfig = field(default_factory=RunConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)

    # -- derived -----------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return Path(self.memory.db).expanduser()

    def api_key(self) -> str:
        return self.llm.api_key or os.environ.get(self.llm.api_key_env, "")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Never leak the key through the config dict (used by the web UI).
        d["llm"]["api_key"] = "***" if d["llm"].get("api_key") else ""
        return d


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: env var -> dotted config path. Only settings worth overriding from the shell.
_ENV_MAP = {
    "ADBAGENT_MODEL": "llm.model",
    "ADBAGENT_MODEL_SMALL": "llm.model_small",
    "ADBAGENT_MODEL_IMAGE": "llm.model_image",
    "ADBAGENT_MODEL_SKILL": "llm.model_skill",
    "ADBAGENT_MODEL_SKILL_IMAGE": "llm.model_skill_image",
    "ADBAGENT_PROVIDER": "llm.provider",
    "ADBAGENT_SERVICE_TIER": "llm.service_tier",
    "ADBAGENT_BASE_URL": "llm.base_url",
    "ADBAGENT_RPM": "llm.rpm",
    "ADBAGENT_MAX_TOKENS": "llm.max_tokens",
    "ADBAGENT_MAX_TOKENS_IMAGE": "llm.max_tokens_image",
    "ADBAGENT_DB": "memory.db",
    "ADBAGENT_BUDGET_USD": "safety.budget_usd",
    "ADBAGENT_MAX_STEPS": "run.max_steps",
    "ADBAGENT_PAGER_SWEEP": "run.pager_sweep",
    "ADBAGENT_VISION_IN_DECIDER": "llm.vision_in_decider",
    "ADBAGENT_SKILLS_DIR": "skills.skills_dir",
    "ADBAGENT_WATCH_INTERVAL": "watch.interval_s",
    "ADBAGENT_WATCH_POLICY": "watch.policy",
    "ADBAGENT_WATCH_POLICIES_DIR": "watch.policies_dir",
    "ADBAGENT_WATCH_LEDGER": "watch.ledger",
    "ADBAGENT_WATCH_DRAFT": "watch.draft",
    "ADBAGENT_DISABLE_AUTO_ROTATE": "device.disable_auto_rotate",
    "ANDROID_SERIAL": "device.serial",
}


def _coerce(current: Any, raw: Any) -> Any:
    """Coerce a raw value to the type of the existing default."""
    if isinstance(current, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if isinstance(raw, str):
            return [s for s in (p.strip() for p in raw.split(",")) if s]
        return list(raw)
    return raw


def _set_path(cfg: Config, dotted: str, raw: Any) -> None:
    section_name, _, key = dotted.partition(".")
    section = getattr(cfg, section_name, None)
    if section is None or not is_dataclass(section):
        raise KeyError(f"unknown config section: {section_name!r}")
    if not any(f.name == key for f in fields(section)):
        raise KeyError(f"unknown config key: {dotted!r}")
    setattr(section, key, _coerce(getattr(section, key), raw))


def _apply_mapping(cfg: Config, data: Dict[str, Any], origin: str) -> List[str]:
    """Apply a nested {section: {key: value}} mapping. Returns warnings."""
    warnings: List[str] = []
    for section_name, values in data.items():
        if not isinstance(values, dict):
            warnings.append(f"{origin}: ignoring non-object entry {section_name!r}")
            continue
        for key, value in values.items():
            try:
                section = getattr(cfg, section_name, None)
                if section and hasattr(section, key):
                    current_val = getattr(section, key)
                    if value == "" and current_val != "":
                        continue
                _set_path(cfg, f"{section_name}.{key}", value)
            except (KeyError, ValueError, TypeError) as exc:
                warnings.append(f"{origin}: {exc}")
    return warnings



def find_config_file(explicit: Optional[str] = None,
                     start: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    here = (start or Path.cwd()).resolve()
    for name in DEFAULT_CONFIG_NAMES:
        candidate = here / name
        if candidate.is_file():
            return candidate
    fallback = Path("~/.config/adbagent/config.json").expanduser()
    return fallback if fallback.is_file() else None


def load_config(config_path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> "LoadedConfig":
    """Build a Config from defaults < env < config.json < explicit overrides.

    `overrides` uses dotted paths ("llm.model") and skips None values, so a CLI
    parser can hand its whole namespace over without filtering first.
    """
    cfg = Config()
    warnings: List[str] = []

    for env_name, dotted in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw:
            try:
                _set_path(cfg, dotted, raw)
            except (KeyError, ValueError, TypeError) as exc:
                warnings.append(f"${env_name}: {exc}")

    path = find_config_file(config_path)
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{path}: could not read ({exc})")
        else:
            if isinstance(data, dict):
                warnings += _apply_mapping(cfg, data, str(path))
            else:
                warnings.append(f"{path}: top level must be an object")

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        try:
            _set_path(cfg, dotted, value)
        except (KeyError, ValueError, TypeError) as exc:
            warnings.append(f"--{dotted}: {exc}")

    return LoadedConfig(config=cfg, path=path, warnings=warnings)


@dataclass
class LoadedConfig:
    config: Config
    path: Optional[Path]
    warnings: List[str]


def save_device_serial(serial: str,
                       config_path: Optional[str] = None) -> Path:
    """Persist `device.serial` into the config file after a successful pairing.

    Reads the existing JSON (if any), sets ``device.serial``, and writes it
    back, preserving every other key the user has configured.  If no config
    file exists yet, one is created in the current directory.
    """
    path = find_config_file(config_path)
    if path is None:
        path = Path.cwd() / DEFAULT_CONFIG_NAMES[0]

    # Read existing content (or start fresh).
    data: Dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if not isinstance(data, dict):
        data = {}

    # Update only the serial key.
    data.setdefault("device", {})["serial"] = serial

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
