"""Shell blocklist, watchdog and the timeout patch.

These are the parts of the device layer that can be tested without a phone.
"""

from __future__ import annotations

import atexit
import threading
import time

import pytest

from adbagent.device import (DENY_PATTERNS, DeviceTimeout, PRESS_KEYS, ShellDenied,
                             _guard, check_shell, patch_socket_timeout)


@pytest.mark.parametrize("command", [
    # wipe / policy
    "am broadcast -a android.intent.action.MASTER_CLEAR",
    "recovery --wipe_data",
    "dpm remove-active-admin com.x/.Admin",
    # power
    "reboot bootloader",
    "svc power shutdown",
    "setprop sys.powerctl reboot",
    # filesystem
    "rm -rf /data",
    "rm -rf /sdcard",
    "mkfs.ext4 /dev/block/x",
    "dd if=/dev/zero of=/dev/block/sda",
    # packages
    "pm uninstall com.whatsapp",
    "pm clear com.android.settings",
    "pm disable-user com.android.settings",
    # connectivity -- would strand a Wi-Fi agent
    "settings put global airplane_mode_on 1",
    "svc wifi disable",
    "svc data disable",
    "cmd wifi forget-network 1",
    # self-destruct
    "settings put global adb_enabled 0",
    "locksettings clear --old 1234",
    # telephony
    "service call isms 5 s16 +15551234567",
])
def test_dangerous_commands_are_refused(command):
    with pytest.raises(ShellDenied):
        check_shell(command)


@pytest.mark.parametrize("command", [
    "dumpsys power",
    "dumpsys window",
    "settings get global window_animation_scale",
    "settings put global window_animation_scale 0",
    "settings put system screen_off_timeout 1800000",
    "settings put system accelerometer_rotation 0",
    "settings put system user_rotation 0",
    "input tap 100 200",
    "am start -a android.intent.action.VIEW -d https://example.com",
    "pm list packages",
    "getprop ro.build.version.sdk",
    "ime set com.google.android.inputmethod.latin/.LatinIME",
    "svc power stayon false",
    "svc power stayon true",
    "pkill -f com.wetest.uia2.Main",
])
def test_ordinary_commands_are_allowed(command):
    check_shell(command)


def test_metacharacters_are_refused_by_default():
    with pytest.raises(ShellDenied):
        check_shell("dumpsys window | grep mCurrentFocus")
    with pytest.raises(ShellDenied):
        check_shell("echo hi; pm uninstall com.x")
    # ...but a caller that genuinely needs a pipe can opt in, and the blocklist
    # still applies to what is being piped.
    check_shell("dumpsys window | grep mCurrentFocus", allow_meta=True)
    with pytest.raises(ShellDenied):
        check_shell("dumpsys window | pm uninstall com.x", allow_meta=True)


def test_blocklist_is_case_insensitive():
    with pytest.raises(ShellDenied):
        check_shell("PM UNINSTALL com.x")


def test_deny_patterns_all_compile():
    assert DENY_PATTERNS
    for pattern in DENY_PATTERNS:
        assert pattern.pattern


def test_guard_returns_the_value():
    assert _guard(lambda: 7, 1.0, "fast") == 7


def test_guard_times_out_rather_than_hanging():
    start = time.monotonic()
    with pytest.raises(DeviceTimeout):
        _guard(lambda: time.sleep(5), 0.2, "slow")
    assert time.monotonic() - start < 2.0


def test_guard_propagates_exceptions():
    with pytest.raises(ValueError, match="boom"):
        _guard(lambda: (_ for _ in ()).throw(ValueError("boom")), 1.0, "raise")


def test_socket_timeout_patch_is_applied_and_idempotent():
    from uiautomator2 import core

    patch_socket_timeout()
    patch_socket_timeout()
    assert core.AdbHTTPConnection._adbagent_patched is True


def test_press_key_vocabulary_matches_the_server():
    """The server accepts these names and silently returns false for others."""
    assert {"back", "home", "enter", "recent", "delete"} <= PRESS_KEYS
    assert "escape" not in PRESS_KEYS
    assert "tab" not in PRESS_KEYS


def test_device_scroll_gesture_directions_and_duration(capsys):
    class DummyU2:
        def __init__(self):
            self.calls = []

        def swipe_ext(self, gesture, **kw):
            self.calls.append((gesture, kw))

    class DummyConfig:
        class DeviceConfig:
            watchdog_s = 5.0
        device = DeviceConfig()

    from adbagent.device import Device
    d = Device.__new__(Device)
    d._d = DummyU2()
    d._act = lambda fn, what: fn()
    d.cfg = DummyConfig()

    # Direction left -> gesture left (swipe right-to-left)
    d.scroll("left", scale=0.8, duration=0.15)
    assert d._d.calls[-1] == ("left", {"scale": 0.8, "duration": 0.15})

    # Direction right -> gesture right (swipe left-to-right)
    d.scroll("right", scale=0.8, duration=0.15)
    assert d._d.calls[-1] == ("right", {"scale": 0.8, "duration": 0.15})

    # Direction down -> gesture up (scroll down)
    d.scroll("down", scale=0.6)
    assert d._d.calls[-1] == ("up", {"scale": 0.6})

    # Direction up -> gesture down (scroll up)
    d.scroll("up", scale=0.6)
    assert d._d.calls[-1] == ("down", {"scale": 0.6})

    captured = capsys.readouterr()
    assert "Scrolling left (scale=0.8, duration=0.15s)" in captured.out
    assert "Scrolling down (scale=0.6)" in captured.out


# ---------------------------------------------------------------------------
# Teardown
#
# `close` runs after the run has already printed its result, so a step that
# blocks there costs the user their prompt and nothing else -- which is exactly
# how it went unnoticed that all four restores were failing, and how a wedged
# adb socket turned into "it never exits unless I press ctrl+c".
# ---------------------------------------------------------------------------

class _StubAdb:
    def __init__(self):
        self.commands = []

    def shell(self, command, timeout=None):
        self.commands.append(command)
        return ""


class _StubU2:
    """Just enough u2.Device for `Device.close` to run against."""

    def __init__(self, ime="com.sample/.Ime", stop_blocks_forever=False):
        self.adb_device = _StubAdb()
        self.stopped = False
        self._ime = ime
        self._stop_blocks_forever = stop_blocks_forever

    def current_ime(self):
        return "com.github.uiautomator/.FastInputIME"

    def stop_uiautomator(self):
        self.stopped = True
        if self._stop_blocks_forever:
            threading.Event().wait()  # a socket with no timeout, in effect


def _closable_device(**stub_kw):
    """A `Device` wired to a stub, with state worth putting back."""
    from adbagent.config import Config
    from adbagent.device import Device, _Restore

    dev = Device.__new__(Device)
    dev.cfg = Config()
    dev.cfg.device.disable_animations = True
    dev.cfg.device.disable_auto_rotate = True
    dev._restore = _Restore(
        ime="com.sample/.Ime",
        window_scale="1.0", transition_scale="1.0", animator_scale="1.0",
        screen_off_timeout="60000",
        accelerometer_rotation="1", user_rotation="0",
    )
    dev._d = _StubU2(**stub_kw)
    return dev


def test_close_actually_restores_what_open_changed():
    """The restores must run *before* `_d` is cleared.

    Clearing it first made every one of them raise `device is not open`, so a
    run left the phone with animations off, rotation locked, the agent's IME
    selected and a 30-minute screen timeout -- reported only as four warnings
    nobody reads.
    """
    dev = _closable_device()
    stub = dev._d
    dev.close()

    sent = " | ".join(stub.adb_device.commands)
    assert "settings put global window_animation_scale 1.0" in sent
    assert "settings put global transition_animation_scale 1.0" in sent
    assert "settings put global animator_duration_scale 1.0" in sent
    assert "settings put system accelerometer_rotation 1" in sent
    assert "settings put system screen_off_timeout 60000" in sent
    assert "ime set com.sample/.Ime" in sent
    assert "svc power stayon false" in sent
    assert stub.stopped
    assert dev._d is None


def test_close_is_idempotent():
    dev = _closable_device()
    stub = dev._d
    dev.close()
    before = len(stub.adb_device.commands)
    dev.close()
    assert len(stub.adb_device.commands) == before


def test_close_is_bounded_when_a_step_never_returns(monkeypatch, caplog):
    """A teardown step on a dead socket must cost a pause, not the process.

    adbutils floors its socket timeout at 600s and `stop_uiautomator` reaches it
    directly, so before this the exit could block for ten minutes with the run
    finished and printed.
    """
    from adbagent import device as devmod

    monkeypatch.setattr(devmod, "TEARDOWN_BUDGET_S", 1.0)
    dev = _closable_device(stop_blocks_forever=True)
    stub = dev._d

    started = time.monotonic()
    dev.close()
    elapsed = time.monotonic() - started

    assert stub.stopped, "the stalling step should have been entered"
    assert elapsed < 5.0, f"close() took {elapsed:.1f}s -- the budget did not hold"
    assert "could not restore uiautomator server" in caplog.text
    assert "exceeded" in caplog.text  # the watchdog, not some other failure
    assert dev._d is None


def test_close_drops_u2s_duplicate_atexit_hook(monkeypatch):
    """u2 registers `stop_uiautomator` with atexit, under the same server lock.

    Leaving it registered means that if the step above was abandoned on a
    stalled socket -- its daemon thread still holding that lock -- interpreter
    shutdown blocks on it forever, after the last line of output.
    """
    from adbagent import device as devmod

    dropped = []
    monkeypatch.setattr(devmod.atexit, "unregister", dropped.append)

    dev = _closable_device()
    stub = dev._d
    dev.close()

    assert dropped == [stub.stop_uiautomator]


def test_close_drops_the_atexit_hook_even_when_a_step_stalls(monkeypatch):
    """The stalling case is the one that needs it, so it must not be skipped."""
    from adbagent import device as devmod

    dropped = []
    monkeypatch.setattr(devmod.atexit, "unregister", dropped.append)
    monkeypatch.setattr(devmod, "TEARDOWN_BUDGET_S", 1.0)

    dev = _closable_device(stop_blocks_forever=True)
    stub = dev._d
    dev.close()

    assert dropped == [stub.stop_uiautomator]


def test_reconnect_drops_the_replaced_devices_atexit_hook(monkeypatch):
    """A recovery swaps in a fresh u2.Device, and each one registers its own hook.

    Without this, a run that recovered twice reaches interpreter shutdown with
    three `stop_uiautomator` handlers queued on the same server lock -- over a
    link the recovery already proved was wedged.
    """
    from adbagent import device as devmod

    dropped = []
    monkeypatch.setattr(devmod.atexit, "unregister", dropped.append)

    replacement = _StubU2()
    monkeypatch.setattr(devmod.u2, "connect", lambda *a, **kw: replacement)
    monkeypatch.setattr(_StubU2, "settings", {}, raising=False)
    monkeypatch.setattr(_StubU2, "window_size", lambda self: (1080, 2400),
                        raising=False)

    dev = _closable_device()
    original = dev._d
    dev.serial = "192.168.1.23:33463"

    dev._reconnect()
    assert dropped == [original.stop_uiautomator]
    assert dev._d is replacement

    dev.close()
    assert dropped == [original.stop_uiautomator, replacement.stop_uiautomator]
