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

import atexit
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


def _is_connection_error(exc: BaseException) -> bool:
    """True when *exc* signals the device / u2-server socket has gone away.

    We deliberately test by class-name rather than importing every possible
    module, because the actual exception can come from ``http.client``,
    ``urllib3``, ``requests``, ``adbutils``, or ``uiautomator2`` depending on
    what was in-flight when the link dropped.
    """
    cls_name = type(exc).__name__
    # http.client.RemoteDisconnected  ("Remote end closed connection …")
    # requests.ConnectionError / urllib3.ConnectionError / builtins
    # ConnectionResetError / ConnectionRefusedError / BrokenPipeError
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if cls_name in ("RemoteDisconnected", "ProtocolError",
                    "ConnectionClosedError", "AdbError"):
        return True
    # Some libraries chain the real cause as __cause__
    if exc.__cause__ is not None and _is_connection_error(exc.__cause__):
        return True
    return False


def _guard(fn: Callable[[], T], timeout: float, what: str) -> T:
    """Run `fn` with a wall-clock ceiling.

    Belt and braces alongside the socket patch: a device that stops responding
    mid-transfer should surface as an error we can recover from, not as a hang.
    The worker is a daemon thread, so an orphan cannot keep the process alive.

    Connection-level exceptions (``RemoteDisconnected``, ``ConnectionError``,
    etc.) are wrapped into :class:`DeviceLost` so the recovery ladder in
    :meth:`Agent._loop` can handle them instead of crashing.
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
        exc = error[0]
        if _is_connection_error(exc):
            raise DeviceLost(
                f"{what}: connection lost ({type(exc).__name__}: {exc})"
            ) from exc
        raise exc
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


# ---------------------------------------------------------------------------
# QR-code pairing
# ---------------------------------------------------------------------------

class _ConnectListener:
    """mDNS listener that captures the first ``_adb-tls-connect._tcp`` service."""

    def __init__(self) -> None:
        self.address: Optional[str] = None
        self._event = threading.Event()

    def add_service(self, zc: "zeroconf.Zeroconf", type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info and info.parsed_addresses() and info.port:
            self.address = f"{info.parsed_addresses()[0]}:{info.port}"
            self._event.set()

    def remove_service(self, zc: "zeroconf.Zeroconf", type_: str, name: str) -> None:
        pass  # not relevant

    def update_service(self, zc: "zeroconf.Zeroconf", type_: str, name: str) -> None:
        pass  # not relevant

    def wait(self, timeout: float) -> Optional[str]:
        self._event.wait(timeout)
        return self.address


def pair_qr(timeout: float = 120.0,
            on_qr: Optional[Callable[[str], None]] = None) -> str:
    """Pair with a phone by displaying a QR code in the terminal.

    The flow:
      1. Generate a random password and service name.
      2. Advertise an ``_adb-tls-pairing._tcp`` mDNS service.
      3. Render a QR code that encodes ``WIFI:T:ADB;S:<name>;P:<password>;;``.
      4. Wait for the phone to pair (it discovers the pairing port via mDNS).
      5. Discover the ``_adb-tls-connect._tcp`` service that appears after pairing.
      6. ``adb connect`` to the device.

    Returns the connected device serial (``ip:port``).
    """
    import secrets
    import socket

    import qrcode
    import qrcode.constants
    import zeroconf as zc_mod

    # 1. Generate credentials.
    password = secrets.token_hex(6)  # 12-char hex string
    hostname = socket.gethostname().split(".")[0][:16] or "adbagent"
    service_name = f"{hostname}_adb-tls-pairing"
    qr_payload = f"WIFI:T:ADB;S:{service_name};P:{password};;"

    # 2. Render QR code to stdout (text mode, no image viewer needed).
    if on_qr:
        on_qr(qr_payload)
    else:
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=1,
            border=2,
        )
        qr.add_data(qr_payload)
        qr.make(fit=True)
        qr.print_tty()

    # 3. Advertise the pairing service over mDNS so the phone can find us.
    #    We bind to port 5555 as a placeholder — ``adb pair`` actually runs its
    #    own TLS server on a random port. The phone resolves the *real* pairing
    #    endpoint via the adb server's own mDNS advertisement that ``adb pair``
    #    triggers.
    local_ip = _get_local_ip()
    zeroconf_inst = zc_mod.Zeroconf()

    pairing_info = zc_mod.ServiceInfo(
        "_adb-tls-pairing._tcp.local.",
        f"{service_name}._adb-tls-pairing._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=5555,
        properties={"password": password},
    )
    zeroconf_inst.register_service(pairing_info)

    # 4. Also start browsing for the connect service that appears after pairing.
    listener = _ConnectListener()
    browser = zc_mod.ServiceBrowser(
        zeroconf_inst, "_adb-tls-connect._tcp.local.", listener)

    try:
        # 5. Wait for the phone to discover us and connect.
        connect_addr = listener.wait(timeout)

        # If mDNS browsing didn't catch it, fall back to adb mdns services.
        if not connect_addr:
            candidates = mdns_candidates()
            if candidates:
                connect_addr = candidates[0]

        if not connect_addr:
            raise RuntimeError(
                "Timed out waiting for the phone to scan the QR code. "
                "Make sure Wireless Debugging is enabled and both devices "
                "are on the same network.")

        # 6. Connect.
        msg = connect_wireless(connect_addr)
        log.info("connected: %s", msg)
        return connect_addr

    finally:
        browser.cancel()
        zeroconf_inst.unregister_service(pairing_info)
        zeroconf_inst.close()


def _get_local_ip() -> str:
    """Best-effort detection of the machine's LAN IP address."""
    import socket as _socket
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


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
    accelerometer_rotation: Optional[str] = None
    user_rotation: Optional[str] = None


#: Wall-clock ceiling on the whole of `Device.close`, shared across its steps.
#: Teardown happens after the run has already printed its result, so the only
#: thing waiting on it is the user's prompt. Generous enough for a healthy phone
#: (a normal teardown is well under a second, and `stop_uiautomator` alone can
#: legitimately take ~13s), short enough that a dead link costs a pause rather
#: than a hang.
TEARDOWN_BUDGET_S = 45.0

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
        if self.cfg.device.disable_auto_rotate:
            self._disable_auto_rotate()
        # Before keeping the screen on, turn it on. `wake` existed and nothing
        # called it, so a phone that had dimmed off since the last run started the
        # next one against a black screen: an empty dump, a `degenerate` verdict,
        # and a screenshot of nothing for the model to puzzle over. `stayon` only
        # holds a screen that is already awake.
        self.wake()
        self._keep_screen_awake()
        return self

    def close(self) -> None:
        """Put back everything we changed. Safe to call twice.

        `self._d` is cleared at the *end*, not the start: every restore below
        reaches the phone through `self.shell`, which reads it, so clearing it
        first turned all four of them into `device is not open` and left the
        phone with animations off, rotation locked, the agent's IME selected and
        a 30-minute screen timeout.

        The whole thing runs under one deadline. `shell` has its own watchdog,
        but the last two steps talk to adbutils and u2 directly, and there the
        floor is adbutils' 600s socket timeout -- so a wireless link that goes
        quiet during teardown wedges the exit for ten minutes with the run
        already finished and printed. `_guard` runs each step on a daemon
        thread, which is what makes an abandoned one harmless.
        """
        d = self._d
        if d is None:
            return
        deadline = time.monotonic() + TEARDOWN_BUDGET_S
        try:
            for label, fn in (
                ("animations", self._restore_animations),
                ("auto-rotate", self._restore_auto_rotate),
                ("ime", self._restore_ime),
                ("screen timeout", self._restore_screen_timeout),
                ("stay-awake", lambda: d.adb_device.shell("svc power stayon false", timeout=10)),
                ("uiautomator server", lambda: d.stop_uiautomator()),
            ):
                left = deadline - time.monotonic()
                if left <= 0:
                    log.warning("teardown: out of time, skipped %s", label)
                    continue
                try:
                    _guard(fn, left, f"teardown({label})")
                except Exception as exc:  # noqa: BLE001 - teardown must not mask errors
                    log.warning("teardown: could not restore %s: %s", label, exc)
        finally:
            self._drop_u2_atexit_hook(d)
            self._d = None

    @staticmethod
    def _drop_u2_atexit_hook(d: Optional[u2.Device]) -> None:
        """Take u2's own `stop_uiautomator` off the atexit list for `d`.

        Every `u2.connect()` registers one, bound to that object, and they all
        take the same per-(serial, port) server lock. Two things make leaving
        them there dangerous. A teardown step abandoned on a stalled socket
        leaves its daemon thread holding that lock, so the handler blocks
        interpreter shutdown *forever* -- after the last line of output, with
        Ctrl-C the only way out. And `_reconnect` swaps in a fresh device
        without dropping the old one's handler, so a run that recovered twice
        arrives at exit with three of them queued, each on a link we already
        know was wedged.

        We stop the server ourselves in `close`, on whichever device is current,
        so the handlers are redundant as well as hazardous.
        """
        if d is None:
            return
        try:
            atexit.unregister(d.stop_uiautomator)
        except Exception as exc:  # noqa: BLE001 - never worth failing over
            log.debug("could not unregister u2 atexit hook: %s", exc)


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
        self._restore.accelerometer_rotation = get("system", "accelerometer_rotation")
        self._restore.user_rotation = get("system", "user_rotation")

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

    def _disable_auto_rotate(self) -> None:
        self._safe(lambda: self.shell("settings put system accelerometer_rotation 0", timeout=10))
        self._safe(lambda: self.shell("settings put system user_rotation 0", timeout=10))

    def _restore_auto_rotate(self) -> None:
        if not self.cfg.device.disable_auto_rotate:
            return
        if self._restore.accelerometer_rotation:
            self.shell(
                f"settings put system accelerometer_rotation "
                f"{self._restore.accelerometer_rotation}", timeout=10)
        if self._restore.user_rotation:
            self.shell(
                f"settings put system user_rotation "
                f"{self._restore.user_rotation}", timeout=10)


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

    def _keep_screen_awake(self) -> None:
        """Prevent the device screen from sleeping during a run.

        Two complementary mechanisms:
        * ``svc power stayon true`` tells Android to keep the screen on
          while *any* power source is connected (USB, AC, or wireless).
        * A long ``screen_off_timeout`` (30 min) acts as a safety net for
          wirelessly-connected devices where the charging state may not
          apply.

        Both settings are restored by :meth:`close`.
        """
        # stayon bypasses the blocklist via adb_device.shell(), matching
        # the restore path in close().
        self._safe(lambda: self.u2.adb_device.shell(
            "svc power stayon true", timeout=10))
        self._safe(lambda: self.shell(
            "settings put system screen_off_timeout 1800000", timeout=10))
        log.info("screen keep-awake enabled")

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

        Either way, a dump holding nothing but the status and navigation bars is
        never handed back while there is budget left to take another one. Such a
        dump is a frame of nothing -- see `Screen.chrome_only` -- and equality is
        no defence against it: two of them in a row agree, so the settle loop
        used to certify one as a stable screen and pass it to the model.
        """
        budget = self.cfg.device.settle_budget_s
        interval = self.cfg.device.settle_interval_s
        deadline = time.monotonic() + budget

        screen = self._observe_once()
        while screen.chrome_only and time.monotonic() < deadline:
            time.sleep(interval)
            screen = self._observe_once()
        if screen.chrome_only:
            # Out of budget. Nothing better is coming, so hand it over rather
            # than block; `activity` is the only clue left about what is in front.
            log.debug("dump held nothing but system chrome after %.1fs (activity %s)",
                      budget, screen.activity or "?")
            return screen
        if not settle:
            return screen

        while time.monotonic() < deadline:
            time.sleep(interval)
            nxt = self._observe_once()
            # A frame of nothing is not a stable state, and it must not reset the
            # comparison either: the two real frames either side of it are what
            # this loop is here to match.
            if nxt.chrome_only:
                continue
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
        px, py = max(1, int(x)), max(1, int(y))
        print(f"Tap at ({px}, {py})")
        self._act(lambda: self.u2.click(px, py), "tap")

    def long_press(self, x: int, y: int, duration: float = 0.6) -> None:
        px, py = max(1, int(x)), max(1, int(y))
        print(f"Long press at ({px}, {py}) (duration={duration:.2f}s)")
        self._act(lambda: self.u2.long_click(px, py, duration),
                  "long_press")

    def swipe(self, fx: int, fy: int, tx: int, ty: int, duration: float = 0.25) -> None:
        pfx, pfy = max(1, int(fx)), max(1, int(fy))
        ptx, pty = max(1, int(tx)), max(1, int(ty))
        print(f"Swipe ({pfx}, {pfy}) -> ({ptx}, {pty}) (duration={duration:.2f}s)")
        self._act(lambda: self.u2.swipe(pfx, pfy, ptx, pty, duration=duration), "swipe")

    def scroll(self, direction: str, scale: float = 0.6,
               box: Optional[Sequence[int]] = None,
               duration: Optional[float] = None) -> None:
        """Scroll or swipe the screen or a box.

        For vertical scrolling ("down"/"up"), "down" moves content down (swipes finger up)
        and "up" moves content up (swipes finger down).
        For horizontal swiping ("left"/"right"), "left" swipes finger left (showing next item/photo)
        and "right" swipes finger right (showing previous item/photo).
        """
        extra = []
        if box:
            extra.append(f"box={tuple(box)}")
        if duration is not None:
            extra.append(f"duration={duration:.2f}s")
        extra_str = f", {', '.join(extra)}" if extra else ""
        print(f"Scrolling {direction} (scale={scale}{extra_str})")
        gesture = {"down": "up", "up": "down", "left": "left", "right": "right"}
        gesture_dir = gesture.get(direction, direction)
        kwargs: dict = {"scale": scale}
        if box:
            kwargs["box"] = tuple(box)
        if duration is not None:
            kwargs["duration"] = duration
        self._act(
            lambda: self.u2.swipe_ext(gesture_dir, **kwargs),
            "scroll")

    def press(self, key: str) -> None:
        key = key.lower()
        if key not in PRESS_KEYS:
            raise ValueError(f"unsupported key {key!r}; server accepts {sorted(PRESS_KEYS)}")
        print(f"Press key {key!r}")
        self._act(lambda: self.u2.press(key), "press")

    def input_text(self, text: str, clear: bool = True, press_enter: bool = False) -> None:
        print(f"Input text {text!r} (clear={clear}, press_enter={press_enter})")
        self._act(lambda: self.u2.send_keys(text, clear=clear), "input_text")
        if press_enter:
            self.press("enter")

    def get_clipboard(self) -> str:
        try:
            res = _guard(lambda: self.u2.clipboard, self.cfg.device.watchdog_s, "get_clipboard")
            return res if isinstance(res, str) else str(res or "")
        except Exception as exc:  # noqa: BLE001
            log.debug("get_clipboard u2 failed (%s); using shell", exc)
            out = self._safe(lambda: self.shell("cmd clipboard get", timeout=10)) or ""
            return out.strip()

    def set_clipboard(self, text: str) -> None:
        print(f"Set clipboard {text!r}")
        try:
            _guard(lambda: self.u2.set_clipboard(text), self.cfg.device.watchdog_s, "set_clipboard")
        except Exception as exc:  # noqa: BLE001
            log.debug("set_clipboard u2 failed (%s); using shell", exc)
            self._safe(lambda: self.shell(f"cmd clipboard set {text!r}", timeout=10))

    def open_app(self, package: str, timeout_s: Optional[float] = None) -> bool:
        """Launch `package` and wait until it is the app in front.

        `app_start` is fire-and-forget twice over: handed a package that is not
        installed it returns happily having done nothing, and handed a real one it
        returns before the window is added. Returning there leaves the caller
        observing whatever the phone happened to be showing mid-launch -- usually
        the status and nav bars alone, which is how a launched app came to be
        described as the home screen in `runs/71295f360ea5`.

        Returns whether `package` reached the foreground within the budget. False
        is not necessarily fatal (a slow cold start still lands, just later), so
        it is reported rather than raised.
        """
        self._act(lambda: self.u2.app_start(package, stop=False), "open_app")
        budget = (self.cfg.device.launch_timeout_s if timeout_s is None
                  else timeout_s)
        return self.wait_foreground(package, budget)

    def wait_foreground(self, package: str, timeout_s: float) -> bool:
        """Poll until `package` is the foreground app, or the budget expires."""
        if not package or timeout_s <= 0:
            return False
        deadline = time.monotonic() + timeout_s
        while True:
            current, _ = self.current_app()
            if current == package:
                return True
            if time.monotonic() >= deadline:
                log.debug("%s was not in front within %.1fs (front: %s)",
                          package, timeout_s, current or "?")
                return False
            time.sleep(self.cfg.device.launch_poll_s)

    def list_apps(self, query: str = "", third_party_only: bool = False) -> List[str]:
        """List installed application packages on the device, optionally filtered by query string."""
        cmd = "pm list packages"
        if third_party_only:
            cmd += " -3"
        cmd += " -e"
        out = self.shell(cmd, timeout=15)
        packages: List[str] = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("package:"):
                pkg = line[8:].strip()
                if pkg:
                    packages.append(pkg)
        if query:
            q = query.strip().lower()
            packages = [p for p in packages if q in p.lower()]
        return sorted(packages)


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
        # The outgoing device is about to be unreachable from here, so its
        # atexit handler has to go now or nothing will ever take it off the list.
        self._drop_u2_atexit_hook(self._d)
        self._d = u2.connect(self.serial) if self.serial else u2.connect()
        self._d.settings["max_depth"] = self.cfg.device.max_depth
        self._size = tuple(self._d.window_size())  # type: ignore[assignment]
