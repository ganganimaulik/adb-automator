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

from . import xmlgen as X


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


# ---------------------------------------------------------------------------
# Observation across a launch
#
# `runs/71295f360ea5`: WhatsApp was in front, the screenshot showed its chat
# list, and the tree the model was given held a clock, a battery and three nav
# buttons -- so the step reported "the Android home screen is visible". Two
# defects, one per test group below: `app_start` was not waited on, and the
# settle loop certified two identical frames-of-nothing as a stable screen.
# ---------------------------------------------------------------------------

def _replaying_device(dumps, front=("", "")):
    """A `Device` that replays scripted dumps. No phone, no sleeping to speak of."""
    from adbagent.config import Config
    from adbagent.device import Device

    dev = Device.__new__(Device)
    dev.cfg = Config()
    dev.cfg.device.settle_interval_s = 0.005
    dev._size = (X.W, X.H)
    remaining = list(dumps)
    dev.taken = []

    def dump():
        # The last dump repeats: a phone does not run out of screens.
        xml = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        dev.taken.append(xml)
        return xml

    dev.dump = dump
    dev.current_app = lambda: front
    return dev


def test_observe_re_dumps_past_a_mid_launch_frame():
    app = X.media_viewer()
    dev = _replaying_device([X.mid_launch(), X.mid_launch(), app],
                            front=("com.whatsapp", ".Main"))
    screen = dev.observe(settle=True)
    assert not screen.chrome_only
    assert screen.package == "com.whatsapp"
    assert len(dev.taken) >= 3


def test_two_identical_mid_launch_frames_do_not_count_as_settled():
    """Equality is no defence: a frame of nothing agrees with the next one."""
    dev = _replaying_device([X.mid_launch()], front=("com.whatsapp", ".Main"))
    dev.cfg.device.settle_budget_s = 0.05
    screen = dev.observe(settle=True)
    # Nothing better ever arrived, so it is handed over rather than blocking --
    # but only after the budget was spent trying, not on the second matching dump.
    assert screen.chrome_only
    assert len(dev.taken) > 2


def test_a_blank_frame_between_two_identical_frames_still_settles():
    app = X.media_viewer()
    dev = _replaying_device([app, X.mid_launch(), app],
                            front=("com.whatsapp", ".Main"))
    screen = dev.observe(settle=True)
    assert screen.package == "com.whatsapp"
    # The blank was skipped, not treated as a change: the third dump matched the
    # first, so three dumps were enough.
    assert len(dev.taken) == 3


def test_a_mid_launch_dump_is_not_re_dumped_forever_when_settle_is_off():
    dev = _replaying_device([X.mid_launch()], front=("com.whatsapp", ".Main"))
    dev.cfg.device.settle_budget_s = 0.05
    start = time.monotonic()
    assert dev.observe().chrome_only
    assert time.monotonic() - start < 1.0


def _launching_device(fronts, **device_cfg):
    """A `Device` whose foreground package follows `fronts`, one per poll."""
    from adbagent.config import Config
    from adbagent.device import Device

    class DummyU2:
        def __init__(self):
            self.started = []

        def app_start(self, package, stop=False):
            self.started.append(package)

    dev = Device.__new__(Device)
    dev.cfg = Config()
    dev.cfg.device.launch_poll_s = 0.0
    for key, value in device_cfg.items():
        setattr(dev.cfg.device, key, value)
    dev._d = DummyU2()
    remaining = list(fronts)
    dev.polls = []

    def current_app():
        pkg = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        dev.polls.append(pkg)
        return pkg, ""

    dev.current_app = current_app
    return dev


def test_open_app_waits_until_the_package_is_really_in_front():
    """`app_start` returns before the window exists; the launch is not the app."""
    dev = _launching_device(["com.android.systemui", "com.android.systemui",
                             "com.whatsapp"])
    assert dev.open_app("com.whatsapp") is True
    assert dev._d.started == ["com.whatsapp"]
    assert dev.polls == ["com.android.systemui", "com.android.systemui",
                         "com.whatsapp"]


def test_open_app_reports_a_launch_that_never_lands():
    """Not installed, or too slow to matter. Either way the caller must know."""
    dev = _launching_device(["com.android.systemui"], launch_timeout_s=0.05)
    assert dev.open_app("com.whatsapp") is False
    assert len(dev.polls) > 1


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
# The phone's own clock
#
# Read from the device because the timestamps the model reads were rendered by
# apps in the *phone's* timezone. Whatever comes back is going into the prompt
# as a stated fact, so it is validated rather than trusted.
# ---------------------------------------------------------------------------

def _dev_answering(reply):
    """A `Device` whose shell answers `reply` (or raises it). No phone."""
    from adbagent.device import Device

    dev = Device.__new__(Device)

    def shell(command, timeout=20.0, allow_meta=False):
        dev.asked = command
        if isinstance(reply, BaseException):
            raise reply
        return reply

    dev.shell = shell
    dev.asked = ""
    return dev


def test_the_date_comes_from_the_phone():
    dev = _dev_answering("2026-08-06\n")
    assert dev.today() == "2026-08-06"
    # No metacharacters, so it survives `check_shell` on the real path.
    assert dev.asked == "date +%Y-%m-%d"
    check_shell(dev.asked)


@pytest.mark.parametrize("reply", [
    "",                              # nothing came back
    "date: bad date",                # toybox complaining
    "Thu Aug  6 17:12:34 IST 2026",  # a different format than we asked for
    "2026-13-45",                    # shaped right, not a date
    DeviceTimeout("shell exceeded 10s"),
    RuntimeError("device offline"),
])
def test_an_answer_that_is_not_a_date_is_no_date_at_all(reply):
    """"" and not the host's clock. A wrong date is worse than none: the prompt
    states it as fact and the model cannot check it against the screen."""
    assert _dev_answering(reply).today() == ""


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
