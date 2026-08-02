"""Shell blocklist, watchdog and the timeout patch.

These are the parts of the device layer that can be tested without a phone.
"""

from __future__ import annotations

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


def test_device_scroll_gesture_directions_and_duration():
    class DummyU2:
        def __init__(self):
            self.calls = []
        def swipe_ext(self, gesture_dir, **kwargs):
            self.calls.append((gesture_dir, kwargs))

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

