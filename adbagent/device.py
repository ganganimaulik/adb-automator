"""The device layer: the only module that imports uiautomator2.

Everything that touches the phone goes through `Device`. That gives us one place
to enforce timeouts, one chokepoint for shell commands, one recovery ladder, and
one teardown path that puts the phone back the way we found it.

Several behaviours of uiautomator2 3.7.0 are worked around here; each is
commented at the site. The most important is the socket timeout: the `timeout=`
argument on every jsonrpc call is inert, because `AdbHTTPConnection.connect()`
builds the socket via adbutils without applying it, and adbutils defaults to
600 seconds. Left alone, a single wedged call hangs an "infinite" agent for ten
minutes.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import adbutils
import uiautomator2 as u2
from PIL import Image

from .config import Config
from .fingerprint import attach
from .screen import Screen, parse

log = logging.getLogger("adbagent.device")

T = TypeVar("T")


class DeviceTimeout(RuntimeError):
    """A device call exceeded our own wall-clock watchdog."""


class ShellDenied(PermissionError):
    """A shell command was refused by the blocklist."""


class DeviceLost(RuntimeError):
    """The device is gone and the recovery ladder could not bring it back."""


# ---------------------------------------------------------------------------
# Shell blocklist
# ---------------------------------------------------------------------------

#: Commands that can brick the phone, strand a Wi-Fi agent, or destroy data.
#: Enforced before execution, never "best effort".
DENY_PATTERNS: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    # wipe / device policy
    r"\bwipe\b", r"\brecovery\b", r"--wipe", r"\bMASTER_CLEAR\b", r"\bFACTORY_RESET\b",
    r"\bdpm\b", r"\bdevice_policy\b",
    # power / boot
    r"\breboot\b", r"\bsvc\s+power\s+(?:shutdown|reboot)\b",
    r"\bsetprop\s+sys\.powerctl\b", r"\bbootloader\b", r"\bfastboot\b", r"\bsideload\b",
    # destructive filesystem
    r"\brm\s+(?:-[a-z]*\s+)*/(?:system|data|vendor|sdcard|storage)\b",
    r"\brm\s+-[a-z]*r[a-z]*f\s+/\s*$", r"\bmkfs\b", r"\bdd\s+[^|]*\bof=/dev\b",
    r"\bmount\b[^|]*\bremount\b",
    # package destruction
    r"\bpm\s+uninstall\b", r"\bcmd\s+package\s+uninstall\b",
    r"\bpm\s+disable(?:-user)?\b", r"\bpm\s+clear\b", r"\bpm\s+(?:hide|suspend)\b",
    # connectivity kill switches -- these would strand a Wi-Fi-attached agent
    r"\bsettings\s+put\s+global\s+airplane_mode_on\b",
    r"\bsvc\s+(?:wifi|data|bluetooth)\s+disable\b",
    r"\bcmd\s+wifi\b[^|]*\b(?:disconnect|forget|disable)\b",
    # debugging self-destruct
    r"\bsettings\s+put\s+global\s+adb_enabled\s+0\b",
    r"\blocksettings\b",
    # messaging / telephony -- costs money, reaches other people
    r"\bservice\s+call\s+isms\b", r"\bam\s+start\b[^|]*\bACTION_CALL\b",
    r"\bam\s+broadcast\b[^|]*\bSENDTO\b",
))

#: Shell metacharacters that would let a crafted argument escape the command we
#: think we are running.
_METACHARS = re.compile(r"[;&|`$><\n]")


def check_shell(command: str, allow_meta: bool = False) -> None:
    """Raise ShellDenied if `command` is not safe to run."""
    for pattern in DENY_PATTERNS:
        if pattern.search(command):
            raise ShellDenied(
                f"refused: {command!r} matches the blocklist ({pattern.pattern})")
    if not allow_meta and _METACHARS.search(command):
        raise ShellDenied(f"refused: {command!r} contains shell metacharacters")


# ---------------------------------------------------------------------------
# uiautomator2 patches
# ---------------------------------------------------------------------------

def patch_socket_timeout() -> None:
    """Make the `timeout=` argument on jsonrpc calls actually do something.

    u2's `AdbHTTPConnection.connect()` assigns `self.sock` from adbutils without
    applying `self.timeout`, so every RPC inherits adbutils' 600s default. We
    re-apply it after the socket exists.
    """
    from uiautomator2 import core

    if getattr(core.AdbHTTPConnection, "_adbagent_patched", False):
        return

    original = core.AdbHTTPConnection.connect

    def connect(self):  # type: ignore[no-untyped-def]
        original(self)
        timeout = getattr(self, "timeout", None)
        if isinstance(timeout, (int, float)) and timeout > 0 and self.sock is not None:
            self.sock.settimeout(float(timeout))

    core.AdbHTTPConnection.connect = connect
    core.AdbHTTPConnection._adbagent_patched = True
    log.debug("patched AdbHTTPConnection.connect to honour timeouts")


def _guard(fn: Callable[[], T], timeout: float, what: str) -> T:
    """Run `fn` with a wall-clock ceiling.

    Belt and braces alongside the socket patch: a device that stops responding
    mid-transfer should surface as an error we can recover from, not as a hang.
    The worker is a daemon thread, so an orphan cannot keep the process alive.
    """
    box: List = []
    error: List[BaseException] = []

    def run() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True, name=f"adbagent-{what}")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise DeviceTimeout(f"{what} exceeded {timeout:.0f}s")
    if error:
        raise error[0]
    return box[0]


# ---------------------------------------------------------------------------
# Wireless pairing
# ---------------------------------------------------------------------------

def adb_path() -> str:
    return adbutils.adb_path()


def pair(host_port: str, code: str, timeout: float = 30.0) -> str:
    """`adb pair` -- adbutils has no binding, so shell out.

    Android 11+ shows two different ports: the pairing port (with the 6-digit
    code) and the connect port. They are not the same, and the connect port
    changes every time Wireless debugging is toggled.
    """
    proc = subprocess.run(
        [adb_path(), "pair", host_port],
        input=f"{code}\n", text=True, capture_output=True, timeout=timeout,
    )
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or "Successfully paired" not in out:
        raise RuntimeError(f"pairing failed: {out}")
    return out


def connect_wireless(addr: str, timeout: float = 5.0) -> str:
    """`adb connect`, with the error checking adbutils omits.

    `AdbClient.connect()` does not raise on failure -- it returns the server's
    message -- so a typo'd address silently looks like success.
    """
    msg = adbutils.adb.connect(addr, timeout=timeout)
    if "connected to" not in msg:  # also covers "already connected to"
        raise RuntimeError(msg)
    return msg


def mdns_candidates() -> List[str]:
    """Discover wireless-debugging endpoints, whose port changes on every toggle."""
    try:
        proc = subprocess.run([adb_path(), "mdns", "services"],
                              text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    found: List[str] = []
    for line in proc.stdout.splitlines():
        if "_adb-tls-connect._tcp" in line:
            m = re.search(r"(\d+\.\d+\.\d+\.\d+:\d+)", line)
            if m:
                found.append(m.group(1))
    return found


def list_devices() -> List[adbutils.AdbDevice]:
    return adbutils.adb.device_list()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@dataclass
class _Restore:
    """State we changed and must put back."""
    ime: Optional[str] = None
    window_scale: Optional[str] = None
    transition_scale: Optional[str] = None
    animator_scale: Optional[str] = None
    screen_off_timeout: Optional[str] = None


#: Names the on-device server accepts for `press`. Anything else silently
#: returns false rather than raising, so we validate before sending.
PRESS_KEYS = frozenset({
    "home", "back", "left", "right", "up", "down", "center", "menu", "search",
    "enter", "delete", "del", "recent", "volume_up", "volume_down",
    "volume_mute", "camera", "power",
})


class Device:
    """A guarded uiautomator2 session."""

    def __init__(self, cfg: Config, serial: str = ""):
        self.cfg = cfg
        self.serial = serial or cfg.device.serial
        self._restore = _Restore()
        self._d: Optional[u2.Device] = None
        self._size: Tuple[int, int] = (0, 0)
        self._recoveries = 0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "Device":
        patch_socket_timeout()
        log.info("connecting to %s", self.serial or "(only attached device)")
        self._d = u2.connect(self.serial) if self.serial else u2.connect()

        d = self._d
        d.settings["max_depth"] = self.cfg.device.max_depth
        d.settings["wait_timeout"] = 10.0
        # A short post-action pause lets the UI start reacting before we dump;
        # the adaptive settle loop then waits for it to finish.
        d.settings["operation_delay"] = (0, 0.2)
        d.settings["operation_delay_methods"] = ["click", "swipe", "drag", "press"]

        self._size = tuple(d.window_size())  # type: ignore[assignment]
        self._snapshot_state()
        if self.cfg.device.disable_animations:
            self._set_animations("0")
        return self

    def close(self) -> None:
        """Put back everything we changed. Safe to call twice."""
        if self._d is None:
            return
        d, self._d = self._d, None
        for label, fn in (
            ("animations", lambda: self._restore_animations()),
            ("ime", lambda: self._restore_ime()),
            ("screen timeout", lambda: self._restore_screen_timeout()),
            ("stay-awake", lambda: d.adb_device.shell("svc power stayon false", timeout=10)),
            ("uiautomator server", lambda: d.stop_uiautomator()),
        ):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - teardown must not mask errors
                log.warning("teardown: could not restore %s: %s", label, exc)

    def __enter__(self) -> "Device":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def u2(self) -> u2.Device:
        if self._d is None:
            raise DeviceLost("device is not open")
        return self._d

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    # -- saved state -------------------------------------------------------

    def _snapshot_state(self) -> None:
        get = self._setting_get
        self._restore.ime = self._safe(lambda: self.u2.current_ime())
        self._restore.window_scale = get("global", "window_animation_scale")
        self._restore.transition_scale = get("global", "transition_animation_scale")
        self._restore.animator_scale = get("global", "animator_duration_scale")
        self._restore.screen_off_timeout = get("system", "screen_off_timeout")

    def _setting_get(self, namespace: str, key: str) -> Optional[str]:
        out = self._safe(lambda: self.shell(f"settings get {namespace} {key}", timeout=10))
        if not out:
            return None
        out = out.strip()
        return None if out in ("null", "") else out

    def _set_animations(self, value: str) -> None:
        for key in ("window_animation_scale", "transition_animation_scale",
                    "animator_duration_scale"):
            self._safe(lambda k=key: self.shell(f"settings put global {k} {value}",
                                                timeout=10))

    def _restore_animations(self) -> None:
        if not self.cfg.device.disable_animations:
            return
        for key, saved in (("window_animation_scale", self._restore.window_scale),
                           ("transition_animation_scale", self._restore.transition_scale),
                           ("animator_duration_scale", self._restore.animator_scale)):
            if saved:
                self.shell(f"settings put global {key} {saved}", timeout=10)

    def _restore_ime(self) -> None:
        """u2's `set_input_ime` permanently rewrites the default keyboard.

        `set_input_ime(False)` only runs `ime disable` -- it does not put the
        user's own keyboard back -- so we do it ourselves.
        """
        saved = self._restore.ime
        if not saved or "com.github.uiautomator" in saved:
            return
        if self.u2.current_ime() != saved:
            self.shell(f"ime enable {saved}", timeout=10)
            self.shell(f"ime set {saved}", timeout=10)
            log.info("restored IME to %s", saved)

    def _restore_screen_timeout(self) -> None:
        if self._restore.screen_off_timeout:
            self.shell(
                f"settings put system screen_off_timeout "
                f"{self._restore.screen_off_timeout}", timeout=10)

    @staticmethod
    def _safe(fn: Callable[[], T]) -> Optional[T]:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            log.debug("ignoring: %s", exc)
            return None

    # -- shell -------------------------------------------------------------

    def shell(self, command: str, timeout: float = 20.0,
              allow_meta: bool = False) -> str:
        """The single chokepoint for raw device commands."""
        check_shell(command, allow_meta=allow_meta)
        result = _guard(lambda: self.u2.adb_device.shell(command, timeout=timeout),
                        timeout + 5, f"shell({command[:40]})")
        return result if isinstance(result, str) else str(result)

    # -- observation -------------------------------------------------------

    def dump(self) -> str:
        return _guard(
            lambda: self.u2.dump_hierarchy(compressed=self.cfg.device.compressed,
                                           max_depth=self.cfg.device.max_depth),
            self.cfg.device.watchdog_s, "dump_hierarchy")

    def current_app(self) -> Tuple[str, str]:
        """(package, activity). Uses the adb path, which survives a dead server."""
        try:
            info = _guard(lambda: self.u2.app_current(), 15, "app_current")
            return info.get("package", ""), info.get("activity", "")
        except Exception as exc:  # noqa: BLE001
            log.debug("app_current failed: %s", exc)
            return "", ""

    def observe(self, settle: bool = False) -> Screen:
        """One dump -> parsed, pruned, fingerprinted screen.

        With `settle=True`, re-dump until two consecutive dumps agree or the
        budget expires. That is both cheaper on average and more correct than a
        fixed sleep: fast screens return immediately, slow ones get the time
        they need.
        """
        budget = self.cfg.device.settle_budget_s
        interval = self.cfg.device.settle_interval_s
        deadline = time.monotonic() + budget

        screen = self._observe_once()
        if not settle:
            return screen

        while time.monotonic() < deadline:
            time.sleep(interval)
            nxt = self._observe_once()
            if nxt.exact_id == screen.exact_id:
                return nxt
            screen = nxt
        log.debug("screen never settled within %.1fs", budget)
        return screen

    def _observe_once(self) -> Screen:
        xml = self.dump()
        package, activity = self.current_app()
        screen = parse(xml, width=self._size[0], height=self._size[1],
                       activity=activity)
        if package and not screen.package:
            screen.package = package
        return attach(screen)

    def screenshot(self, max_long_edge: int = 1280, quality: int = 82) -> bytes:
        """A JPEG, downscaled on the device where possible.

        Never send the raw PNG: a 1080x2400 `screencap -p` is 1-3 MB, and
        Fireworks caps total base64 image payload at 10 MB.
        """
        long_edge = max(self._size) or max_long_edge
        scale = min(1.0, max_long_edge / long_edge) if long_edge else 1.0
        image: Optional[Image.Image] = None
        try:
            raw = _guard(lambda: self.u2.jsonrpc.takeScreenshot(scale, quality),
                         self.cfg.device.watchdog_s, "takeScreenshot")
            if raw:
                image = Image.open(io.BytesIO(base64.b64decode(raw)))
        except Exception as exc:  # noqa: BLE001 - fall through to the adb path
            log.debug("server screenshot failed (%s); using screencap", exc)
        if image is None:
            image = _guard(lambda: self.u2.adb_device.screenshot(),
                           self.cfg.device.watchdog_s, "screencap")

        w, h = image.size
        factor = min(1.0, max_long_edge / max(w, h))
        if factor < 1.0:
            # Aspect ratio must be preserved -- distortion is a documented cause
            # of models misjudging what they are looking at.
            image = image.resize((max(1, round(w * factor)), max(1, round(h * factor))),
                                 Image.LANCZOS)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    # -- actions -----------------------------------------------------------

    def tap(self, x: int, y: int) -> None:
        # Any coordinate below 1 is interpreted by u2 as a *fraction* of the
        # screen, so a legitimate pixel 0 must be nudged to 1.
        self._act(lambda: self.u2.click(max(1, int(x)), max(1, int(y))), "tap")

    def long_press(self, x: int, y: int, duration: float = 0.6) -> None:
        self._act(lambda: self.u2.long_click(max(1, int(x)), max(1, int(y)), duration),
                  "long_press")

    def swipe(self, fx: int, fy: int, tx: int, ty: int, duration: float = 0.25) -> None:
        self._act(lambda: self.u2.swipe(max(1, int(fx)), max(1, int(fy)),
                                        max(1, int(tx)), max(1, int(ty)),
                                        duration=duration), "swipe")

    def scroll(self, direction: str, scale: float = 0.6,
               box: Optional[Sequence[int]] = None) -> None:
        """Scroll the screen or a box. `direction` is the content's travel."""
        # swipe_ext takes the *gesture* direction, which is the opposite of the
        # direction the content moves: to scroll down, you swipe up.
        gesture = {"down": "up", "up": "down", "left": "right", "right": "left"}
        gesture_dir = gesture.get(direction, direction)
        self._act(
            lambda: self.u2.swipe_ext(gesture_dir, scale=scale,
                                      box=tuple(box) if box else None),
            "scroll")

    def press(self, key: str) -> None:
        key = key.lower()
        if key not in PRESS_KEYS:
            raise ValueError(f"unsupported key {key!r}; server accepts {sorted(PRESS_KEYS)}")
        self._act(lambda: self.u2.press(key), "press")

    def input_text(self, text: str, clear: bool = True) -> None:
        self._act(lambda: self.u2.send_keys(text, clear=clear), "input_text")

    def clear_text(self) -> None:
        self._act(lambda: self.u2.clear_text(), "clear_text")

    def hide_keyboard(self) -> None:
        self._safe(lambda: self.u2.hide_keyboard())

    def open_app(self, package: str) -> None:
        self._act(lambda: self.u2.app_start(package, stop=False), "open_app")

    def _act(self, fn: Callable[[], T], what: str) -> T:
        return _guard(fn, self.cfg.device.watchdog_s, what)

    # -- health ------------------------------------------------------------

    def is_awake(self) -> bool:
        out = self._safe(lambda: self.shell("dumpsys power", timeout=15)) or ""
        if "mWakefulness=" in out:
            return "mWakefulness=Awake" in out
        return bool(self._safe(lambda: self.u2.info.get("screenOn")))

    def is_locked(self) -> bool:
        out = self._safe(lambda: self.shell("dumpsys window", timeout=15)) or ""
        return "mDreamingLockscreen=true" in out

    def wake(self) -> None:
        """Wake and swipe up. A PIN or pattern still needs a human."""
        if not self.is_awake():
            self._safe(lambda: self.u2.screen_on())
        if self.is_locked():
            w, h = self._size
            self._safe(lambda: self.u2.swipe(w // 2, int(h * 0.85),
                                             w // 2, int(h * 0.25), duration=0.2))

    def recover(self, tier: int = 1) -> bool:
        """Escalating recovery. Returns True when the device answers again."""
        self._recoveries += 1
        log.warning("recovery tier %d (attempt %d)", tier, self._recoveries)
        try:
            if tier <= 1:
                self._safe(lambda: self.u2.reset_uiautomator())
            elif tier == 2:
                self._safe(lambda: self.shell("pkill -f com.wetest.uia2.Main",
                                              timeout=15))
                self._safe(lambda: self.shell("pkill -f uiautomator", timeout=15))
                self._reconnect()
            else:
                # The adb server itself is suspect. Killing it drops every
                # in-flight socket, which always kills the u2 server too, so a
                # full rebuild has to follow.
                self._safe(lambda: adbutils.adb.server_kill())
                time.sleep(1.0)
                if self.serial and ":" in self.serial:
                    self._safe(lambda: adbutils.adb.disconnect(self.serial))
                    self._safe(lambda: connect_wireless(self.serial))
                self._reconnect()
            return bool(self._safe(lambda: self.dump()))
        except Exception as exc:  # noqa: BLE001
            log.error("recovery tier %d failed: %s", tier, exc)
            return False

    def _reconnect(self) -> None:
        self._d = u2.connect(self.serial) if self.serial else u2.connect()
        self._d.settings["max_depth"] = self.cfg.device.max_depth
        self._size = tuple(self._d.window_size())  # type: ignore[assignment]
