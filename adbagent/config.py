"""Configuration: dataclasses plus a loader.

Precedence, lowest to highest: built-in defaults < environment < config.json < CLI flags.

The API key is deliberately NOT part of this structure -- it is read from the
environment at call time only, so it can never be written into config.json or
into a run artifact.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_NAMES = ("config.json", "adbagent.json")

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
    temperature: float = 0.0
    max_tokens: int = 1500
    #: Client-side request throttle. Fireworks free accounts are capped at 10 RPM;
    #: paid accounts go to 6000. 120 is a safe default for a paid account.
    rpm: int = 120
    base_url: str = ""
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
    #: because a text-only model given an image part fails the whole call, and
    #: the catalogue flag that would settle it (`adbagent models --vision`) is not
    #: consulted at run time.
    vision_in_decider: bool = False

    def small(self) -> str:
        return self.model_small or self.model

    def image(self) -> str:
        return self.model_image or self.model

    def skill(self) -> str:
        return self.model_skill or self.model


@dataclass
class DeviceConfig:
    serial: str = ""
    #: u2's own default is 50, which silently truncates deep Compose/RN trees.
    max_depth: int = 40
    #: Drop nodes not marked important-for-accessibility. Much smaller XML.
    compressed: bool = True
    #: Adaptive settle: re-dump until two consecutive hashes match, or this budget.
    settle_budget_s: float = 2.0
    settle_interval_s: float = 0.18
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


@dataclass
class SkillsConfig:
    enabled: bool = True
    skills_dir: str = "skills"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    run: RunConfig = field(default_factory=RunConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)

    # -- derived -----------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return Path(self.memory.db).expanduser()

    def api_key(self) -> str:
        return os.environ.get(self.llm.api_key_env, "")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: env var -> dotted config path. Only settings worth overriding from the shell.
_ENV_MAP = {
    "ADBAGENT_MODEL": "llm.model",
    "ADBAGENT_MODEL_SMALL": "llm.model_small",
    "ADBAGENT_MODEL_IMAGE": "llm.model_image",
    "ADBAGENT_MODEL_SKILL": "llm.model_skill",
    "ADBAGENT_PROVIDER": "llm.provider",
    "ADBAGENT_SERVICE_TIER": "llm.service_tier",
    "ADBAGENT_BASE_URL": "llm.base_url",
    "ADBAGENT_RPM": "llm.rpm",
    "ADBAGENT_MAX_TOKENS": "llm.max_tokens",
    "ADBAGENT_DB": "memory.db",
    "ADBAGENT_BUDGET_USD": "safety.budget_usd",
    "ADBAGENT_MAX_STEPS": "run.max_steps",
    "ADBAGENT_PAGER_SWEEP": "run.pager_sweep",
    "ADBAGENT_VISION_IN_DECIDER": "llm.vision_in_decider",
    "ADBAGENT_SKILLS_DIR": "skills.skills_dir",
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
