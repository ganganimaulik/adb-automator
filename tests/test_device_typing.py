"""`Device.input_text`: waiting for focus, reading the field back, retrying.

No phone. `_Phone` stands in for uiautomator2 with fields that take focus a
set number of reads after a tap, and a keys route each test can make drop
everything, drop only the emoji, half-land or raise -- the ways typing has
been seen to fail.
"""

from __future__ import annotations

import io
import re
import sys
import time
from types import SimpleNamespace
from typing import List, Optional

import pytest
from uiautomator2.exceptions import UiObjectNotFoundError

from adbagent import device as devmod
from adbagent.config import Config
from adbagent.device import Device, _compare


class _Field:
    def __init__(self, left: int = 40, top: int = 1960, right: int = 860,
                 bottom: int = 2080, text: str = "", hint: str = "Message...",
                 cls: str = "android.widget.EditText",
                 rid: str = "com.example.chat:id/composer"):
        self.bounds = (left, top, right, bottom)
        self.text, self.hint, self.cls, self.rid = text, hint, cls, rid

    def contains(self, x: int, y: int) -> bool:
        l, t, r, b = self.bounds
        return l <= x <= r and t <= y <= b

    @property
    def center(self):
        l, t, r, b = self.bounds
        return ((l + r) // 2, (t + b) // 2)

    def info(self) -> dict:
        l, t, r, b = self.bounds
        # An empty field reads as its hint, as accessibility reports it.
        return {"className": self.cls, "resourceName": self.rid,
                "text": self.text or self.hint, "focused": True,
                "bounds": {"left": l, "top": t, "right": r, "bottom": b}}


class _Node:
    def __init__(self, phone: "_Phone", selector: dict):
        self.phone, self.selector = phone, selector

    @property
    def exists(self) -> bool:
        self.phone.reads += 1
        self.phone.tick()
        f = self.phone.focused
        pattern = self.selector.get("classNameMatches", ".*")
        return f is not None and re.fullmatch(pattern, f.cls) is not None

    @property
    def info(self) -> dict:
        return self.phone.focused.info()

    def set_text(self, text: str) -> bool:
        self.phone.calls.append(("set_text", text))
        if not self.phone.set_text_works:
            return False
        self.phone.focused.text = text
        return True


class _Phone:
    """Just enough of `u2.Device` for `Device.input_text`.

    `keys` is what uiautomator2's own route does with the text: "types" it,
    "drops" it, "drops_emoji" (keeps only the Basic Multilingual Plane),
    "half" (lands the first half) or "raises".
    """

    def __init__(self, fields: List[_Field], focus_after: int = 0,
                 keys: str = "types", focused: Optional[_Field] = None,
                 set_text_works: bool = True):
        self.fields = fields
        self.focus_after = focus_after
        self.keys = keys
        self.focused = focused
        self.set_text_works = set_text_works
        self.pending: Optional[list] = None
        self.calls: list = []
        self.reads = 0
        self.clipboard = ""
        #: Called with the phone once the keys have gone out.
        self.after_keys = None
        self.jsonrpc = SimpleNamespace(pasteClipboard=self._paste)

    def __call__(self, **selector) -> _Node:
        return _Node(self, selector)

    def tick(self) -> None:
        """One read of the screen's worth of time passing."""
        if self.pending is None:
            return
        if self.pending[1] <= 0:
            self.focused, self.pending = self.pending[0], None
        else:
            self.pending[1] -= 1

    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", x, y))
        hit = next((f for f in self.fields if f.contains(x, y)), None)
        if hit is not None and hit is not self.focused:
            self.pending = [hit, self.focus_after]

    def send_keys(self, text: str, clear: bool = False) -> None:
        self.calls.append(("send_keys", text, clear, self.focused is not None))
        if self.keys == "raises":
            raise UiObjectNotFoundError({"code": -32002, "message": "focused=True"})
        f = self.focused
        if f is not None and self.keys != "drops":
            if self.keys == "drops_emoji":
                text = "".join(ch for ch in text if ord(ch) <= 0xFFFF)
            elif self.keys == "half":
                text = text[:len(text) // 2]
            f.text = text if clear else f.text + text
        if self.after_keys is not None:
            self.after_keys(self)

    def set_clipboard(self, text: str) -> None:
        self.calls.append(("set_clipboard", text))
        self.clipboard = text

    def _paste(self) -> None:
        self.calls.append(("paste",))
        if self.focused is not None:
            self.focused.text += self.clipboard

    def press(self, key: str) -> None:
        self.calls.append(("press", key))

    def routes(self) -> List[str]:
        return [c[0] for c in self.calls if c[0] != "click"]


@pytest.fixture
def quick(monkeypatch):
    """Budgets small enough for a test, polled often enough to use them."""
    monkeypatch.setattr(devmod, "FOCUS_WAIT_S", 0.4)
    monkeypatch.setattr(devmod, "TYPED_WAIT_S", 0.2)
    monkeypatch.setattr(devmod, "TYPING_POLL_S", 0.005)


def _device(phone: _Phone) -> Device:
    dev = Device.__new__(Device)
    dev.cfg = Config()
    dev._d = phone
    return dev


def test_typing_waits_for_the_field_to_take_focus(quick):
    """The timing half of the silent first-attempt failure.

    Focus follows the tap by a few hundred milliseconds -- a React Native
    field asks its JS thread first -- and both of uiautomator2's routes type
    into whatever holds focus when they are dispatched. Sent straight after the
    tap they reached nothing, and the composer was left showing its
    placeholder with the keyboard up.
    """
    field = _Field()
    phone = _Phone([field], focus_after=3)
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    sent = next(c for c in phone.calls if c[0] == "send_keys")
    assert sent[3] is True, "the keys went out before the field had focus"
    assert field.text == "Hey 🔥"
    assert (typed.landed, typed.via, typed.reads) == (True, "keys", "Hey 🔥")
    assert phone.routes() == ["send_keys"]


def test_an_emoji_the_keys_drop_is_set_directly(quick):
    """Set through accessibility the string travels as JSON, so the emoji
    arrives whole -- whatever the keyboard route did with it."""
    field = _Field()
    phone = _Phone([field], keys="drops_emoji")
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    assert field.text == "Hey 🔥"
    assert (typed.landed, typed.via) == (True, "set_text")
    assert ("set_text", "Hey 🔥") in phone.calls


def test_text_the_keys_drop_entirely_is_set_directly(quick):
    field = _Field()
    phone = _Phone([field], keys="drops")
    typed = _device(phone).input_text("up for coffee?", at=field.center)

    assert field.text == "up for coffee?"
    assert (typed.landed, typed.via) == (True, "set_text")


def test_a_field_that_already_held_the_text_needs_nothing_more(quick):
    """Typing it again changes nothing on screen, and that is not a failure.
    Read off the field, the text is there."""
    field = _Field(text="Hey 🔥")
    phone = _Phone([field], keys="drops", focused=field)
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    assert (typed.landed, typed.via) == (True, "keys")
    assert "set_text" not in phone.routes()


def test_a_field_holding_more_than_was_asked_is_put_right(quick):
    """A stray key or a leftover draft: replacing means replacing."""
    field = _Field(text="Hey 🔥g")
    phone = _Phone([field], keys="drops", focused=field)
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    assert field.text == "Hey 🔥"
    assert (typed.landed, typed.via) == (True, "set_text")


def test_an_append_the_keys_drop_is_pasted_once(quick):
    field = _Field(text="see you")
    phone = _Phone([field], keys="drops", focused=field)
    typed = _device(phone).input_text(" tomorrow 🙂", clear=False,
                                      at=field.center)

    assert field.text == "see you tomorrow 🙂"
    assert (typed.landed, typed.via) == (True, "paste")
    assert phone.routes().count("paste") == 1
    assert "set_text" not in phone.routes(), "an append must not replace"


def test_an_append_that_half_landed_is_not_pasted_on_top(quick):
    """What landed is not known, and a paste over it could type it twice."""
    field = _Field(text="see you")
    phone = _Phone([field], keys="half", focused=field)
    typed = _device(phone).input_text(" tomorrow", clear=False,
                                      at=field.center)

    assert typed.landed is False
    assert "paste" not in phone.routes()
    assert "set_text" not in phone.routes()


def test_focus_passed_to_the_next_box_is_not_typed_into_again(quick):
    """One-time-code boxes hand focus along as each digit lands. The box that
    took the digit can no longer be read, which is "unknown" -- not "missed",
    which would type the digit into the next box too."""
    first = _Field(left=100, right=200, top=800, bottom=900, hint="", rid="")
    second = _Field(left=220, right=320, top=800, bottom=900, hint="", rid="")
    phone = _Phone([first, second])
    phone.after_keys = lambda p: setattr(p, "focused", second)
    typed = _device(phone).input_text("4", at=first.center)

    assert first.text == "4" and second.text == ""
    assert typed.landed is None
    assert phone.routes() == ["send_keys"]


def test_a_tap_that_focuses_nothing_still_types_but_cannot_confirm(quick):
    """What it always did, and no worse: the tree then grades it."""
    phone = _Phone([_Field()])
    started = time.monotonic()
    typed = _device(phone).input_text("hello", at=(540, 300))

    assert typed.landed is None and typed.reads is None
    assert phone.routes() == ["send_keys"]
    # The focus wait was spent, and nothing more.
    assert time.monotonic() - started < devmod.FOCUS_WAIT_S + 1.0


def test_focus_that_arrives_after_the_wait_is_still_read_and_retried(quick):
    """Too late for the keys, not too late to read the field and fix it."""
    field = _Field()
    phone = _Phone([field], keys="drops")
    phone.after_keys = lambda p: setattr(p, "focused", field)
    phone.click = lambda x, y: phone.calls.append(("click", x, y))
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    assert field.text == "Hey 🔥"
    assert (typed.landed, typed.via) == (True, "set_text")


def test_enter_follows_text_that_landed(quick):
    field = _Field()
    phone = _Phone([field])
    typed = _device(phone).input_text("weather", press_enter=True,
                                      at=field.center)

    assert typed.entered is True
    assert phone.calls[-1] == ("press", "enter")


def test_enter_is_held_back_from_a_field_that_does_not_hold_the_text(quick):
    """Submitting a field that reads wrong sends something nobody asked for."""
    field = _Field()
    phone = _Phone([field], keys="drops", set_text_works=False)
    typed = _device(phone).input_text("Hey 🔥", press_enter=True,
                                      at=field.center)

    assert typed.landed is False
    assert typed.reads == "Message..."
    assert typed.entered is False
    assert ("press", "enter") not in phone.calls
    assert "enter was not pressed" in typed.note


def test_enter_still_follows_text_that_could_not_be_checked(quick):
    """Unknown is not a miss: holding enter back there would change what
    every unreadable field did before."""
    phone = _Phone([_Field()])
    typed = _device(phone).input_text("weather", press_enter=True,
                                      at=(540, 300))
    assert typed.landed is None
    assert typed.entered is True


def test_a_masked_field_is_not_read_back_or_retyped(quick):
    field = _Field(cls="android.widget.EditText")
    phone = _Phone([field], focused=field)
    # A password field reads back as its mask.
    field.info = lambda: {"className": "android.widget.EditText",
                          "resourceName": "", "text": "••••••",
                          "bounds": {"left": 40, "top": 1960,
                                     "right": 860, "bottom": 2080}}
    typed = _device(phone).input_text("hunter", at=field.center)

    assert typed.landed is None
    assert phone.routes() == ["send_keys"]


def test_a_keys_route_that_raises_is_a_miss_not_a_crash(quick):
    """uiautomator2's own errors -- its keyboard refusing the broadcast, then
    no focused node for its fallback -- are not ValueErrors, and used to end
    the run rather than the step."""
    field = _Field()
    phone = _Phone([field], keys="raises")
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    assert (typed.landed, typed.via) == (True, "set_text")
    assert "UiObjectNotFoundError" in typed.note


def test_a_tap_inside_the_field_that_has_focus_is_typed_into_at_once(quick):
    field = _Field(text="")
    phone = _Phone([field], focused=field)
    _device(phone).input_text("hello", at=field.center)
    # One read before the tap, one after it, then the read-back.
    assert phone.reads <= 4


def test_focus_still_on_the_previous_field_is_waited_through(quick):
    """Tapping the second field while the first still holds focus: the first
    is not evidence the tap landed, so the typing waits for focus to move."""
    name = _Field(top=400, bottom=500, rid="com.example.form:id/name",
                  text="Asha")
    email = _Field(top=600, bottom=700, rid="com.example.form:id/email",
                   hint="Email")
    phone = _Phone([name, email], focus_after=3, focused=name)
    typed = _device(phone).input_text("asha@example.com", at=email.center)

    assert name.text == "Asha"
    assert email.text == "asha@example.com"
    assert typed.landed is True


def test_an_unencodable_console_does_not_stop_the_typing(quick, monkeypatch):
    """`runs/b494220f9f6a`: the run's stdout was a cp1252 pipe, printing the
    action line raised UnicodeEncodeError on the emoji, and the text was never
    sent. The step failed, and the comment only got in two steps later, by a
    long-press on the field and a tap on Paste."""
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout",
                        io.TextIOWrapper(raw, encoding="cp1252"))
    field = _Field()
    phone = _Phone([field])
    typed = _device(phone).input_text("Hey 🔥", at=field.center)

    assert field.text == "Hey 🔥"
    assert typed.landed is True


# ---------------------------------------------------------------------------
# What counts as "the text is in the field"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reads,text", [
    ("Hey 🔥", "Hey 🔥"),
    ("hey 🔥", "Hey 🔥"),                 # a keyboard's capital, or its absence
    ("HEY 🔥", "Hey 🔥"),                 # textAllCaps, applied before it is read
    ("Hey ❤", "Hey ❤️"),                  # the presentation selector dropped
    ("(555) 012-3456", "5550123456"),    # a field that groups digits
    ("Hey  🔥 ", "Hey 🔥"),
    ("?", "?"),
])
def test_text_the_field_holds_compares_exact(reads, text):
    assert _compare(reads, text) == (True, True)


@pytest.mark.parametrize("reads,text,holds", [
    ("Hey ", "Hey 🔥", False),            # the emoji dropped
    ("Hey 🔥g", "Hey 🔥", True),          # held, with a stray key after it
    ("Message...", "Hey 🔥", False),      # still the placeholder
    ("up for", "up for movie?", False),
    ("", "?", False),
])
def test_text_the_field_does_not_hold_exactly(reads, text, holds):
    assert _compare(reads, text) == (holds, False)
