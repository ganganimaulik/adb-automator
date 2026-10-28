"""The MCP server, through a real MCP client, against a scripted phone.

Both protocol eras are exercised where they differ: before 2026-07-28 the
server asks the person mid-call (`elicitation/create`); from 2026-07-28 the
call comes back asking for input and the client retries it with the answer.
`approve` and `hand_over` have to work either way.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import anyio
import mcp_types as types
import pytest
from mcp import Client, StdioServerParameters

from adbagent import phonelock
from adbagent.config import Config
from adbagent.mcp_server import INSTRUCTIONS, build_server, lean_schema

from . import fake

ERAS = ["legacy", "2026-07-28"]


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    return c


def text(result) -> str:
    return "\n".join(getattr(block, "text", "") for block in result.content)


def index_of(screen: str, label: str) -> int:
    for line in screen.splitlines():
        if f'"{label}"' in line:
            return int(re.search(r"#(\d+)", line).group(1))
    raise AssertionError(f"{label!r} not on screen:\n{screen}")


def session_of(started: str) -> str:
    """The SESSION id a start_session answer is headed by."""
    return re.search(r"^SESSION (\w+)", started, re.M).group(1)


def tap(session_id: str, index: int) -> dict:
    return {"session_id": session_id,
            "actions": [{"action": "tap", "target": {"index": index}}]}


class Person:
    """Answers elicitations the way the person in front of the client would."""

    def __init__(self, answer: str = "accept"):
        self.answer = answer
        self.asked: list = []

    async def __call__(self, context, params):
        self.asked.append(params.message)
        if self.answer == "accept":
            return types.ElicitResult(action="accept", content={})
        return types.ElicitResult(action=self.answer)


def drive(cfg, dev, script, *, mode="legacy", person=None):
    """Run `script(client)` against a fresh server; returns what it returns."""
    server, phone = build_server(cfg, open_device=lambda _cfg: dev,
                                 idle_timeout_s=30)
    out = {}

    async def main():
        async with Client(server, mode=mode,
                          elicitation_callback=person) as client:
            out["value"] = await script(client)

    try:
        anyio.run(main)
    finally:
        phone.close()
    return out["value"]


# ---------------------------------------------------------------------------
# What a client sees
# ---------------------------------------------------------------------------

def test_the_tools_a_client_is_offered(cfg):
    async def script(client):
        return (await client.list_tools()).tools

    tools = {t.name: t for t in drive(cfg, fake.FakeDevice(cfg), script)}
    assert set(tools) == {"start_session", "act", "look", "approve",
                          "hand_over", "end_session", "list_policies"}
    assert tools["list_policies"].annotations.read_only_hint is True
    schema = json.dumps(tools["act"].input_schema)
    # The action vocabulary reaches the client as an enum...
    assert '"tap_at"' in schema and '"scroll_to_edge"' in schema
    # ...without the codebase's own docstrings riding along.
    assert ":mod:" not in schema and "stall ladder" not in schema
    # Filled by the server from the person, never by the model.
    assert "decision" not in json.dumps(tools["approve"].input_schema)
    assert tools["look"].annotations.read_only_hint is True
    # Everything after start_session names the session it is for.
    for name in ("act", "look", "approve", "hand_over", "end_session"):
        assert "session_id" in tools[name].input_schema["required"], name


def test_the_instructions_name_every_tool():
    for name in ("start_session", "act", "end_session", "look", "approve",
                 "hand_over", "list_policies"):
        assert name in INSTRUCTIONS


def test_lean_schema_keeps_field_descriptions_and_fields_named_like_keywords():
    schema = {"title": "Step", "description": "A docstring.", "type": "object",
              "properties": {
                  "text": {"title": "Text", "type": "string",
                           "description": "What to type."},
                  "title": {"type": "string", "description": "A field."}}}
    assert lean_schema(schema) == {
        "type": "object", "properties": {
            "text": {"type": "string", "description": "What to type."},
            "title": {"type": "string", "description": "A field."}}}


# ---------------------------------------------------------------------------
# A session
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ERAS)
def test_a_session_from_start_to_end(cfg, mode):
    dev = fake.FakeDevice(cfg)

    async def script(client):
        started = text(await client.call_tool(
            "start_session", {"goal": "open the Wi-Fi settings"}))
        sid = session_of(started)
        acted = text(await client.call_tool(
            "act", tap(sid, index_of(started, "Wi-Fi"))))
        ended = text(await client.call_tool(
            "end_session", {"session_id": sid,
                            "summary": "Wi-Fi settings are open"}))
        late = await client.call_tool("act", {"session_id": sid, "actions": [
            {"action": "press_key", "key": "back"}]})
        return started, acted, ended, late

    started, acted, ended, late = drive(cfg, dev, script, mode=mode)
    assert "CURRENT SCREEN (step 1)" in started
    assert "WHAT HAPPENED" in acted and "-> success" in acted
    assert dev.state == "wifi"
    assert ended.startswith("SESSION ENDED: success")
    assert "Wi-Fi settings are open" in ended
    assert late.is_error and "has ended" in text(late)
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


def test_a_driver_notifies_in_the_same_batch_as_the_tap_after_it(cfg, monkeypatch):
    """`notify` never touches the phone, so the tap batched after it still
    lands on the screen it was planned on -- and each line reaches the owner
    once, however many times the driver says it."""
    sent = []
    monkeypatch.setattr("adbagent.notify.Telegram.send",
                        lambda self, message: sent.append(message))
    cfg.notify.telegram_token = "1:test-token"
    cfg.notify.telegram_chat_id = "42"
    dev = fake.FakeDevice(cfg)

    async def script(client):
        started = text(await client.call_tool(
            "start_session", {"goal": "open the Wi-Fi settings"}))
        sid = session_of(started)
        acted = text(await client.call_tool("act", {"session_id": sid, "actions": [
            {"action": "notify", "text": "Settings: opening Wi-Fi"},
            {"action": "tap", "target": {"index": index_of(started, "Wi-Fi")}}]}))
        again = text(await client.call_tool("act", {"session_id": sid, "actions": [
            {"action": "notify", "text": "settings: opening wi-fi!"}]}))
        await client.call_tool("end_session", {"session_id": sid,
                                               "summary": "Wi-Fi settings are open"})
        return acted, again

    acted, again = drive(cfg, dev, script)
    assert sent == ["Settings: opening Wi-Fi"]
    assert dev.state == "wifi"
    assert "-> sent" in acted and "stopped before" not in acted
    assert "-> already sent" in again


def test_the_instructions_say_how_notify_behaves():
    assert "notify" in INSTRUCTIONS and "once, ever" in INSTRUCTIONS


def test_a_driver_reads_back_what_was_notified_before_it_opens_anything(cfg):
    """What a pass on a chat list does with it: look the rows up first. The
    answer arrives with the screen it was asked on, so the tap batched after
    it still lands."""
    runs = Path(cfg.run.artifacts_dir)
    runs.mkdir(parents=True)
    (runs / "notified.jsonl").write_text(
        json.dumps({"at": 1_793_000_000, "sent": True,
                    "text": "Settings: new message from S: Too early ."}) + "\n",
        encoding="utf-8")
    dev = fake.FakeDevice(cfg)

    async def script(client):
        started = text(await client.call_tool(
            "start_session", {"goal": "open the Wi-Fi settings"}))
        sid = session_of(started)
        acted = text(await client.call_tool("act", {"session_id": sid, "actions": [
            {"action": "read_notified",
             "text": "Settings: new message from S\nSettings: new message from G"},
            {"action": "tap", "target": {"index": index_of(started, "Wi-Fi")}}]}))
        await client.call_tool("end_session", {"session_id": sid,
                                               "summary": "Wi-Fi settings are open"})
        return acted

    acted = drive(cfg, dev, script)
    assert "-> found (1 notified before, oldest first: " in acted
    assert "Settings: new message from S: Too early ." in acted
    assert dev.state == "wifi" and "stopped before" not in acted


def test_the_instructions_say_what_reading_back_is_for():
    assert "read_notified" in INSTRUCTIONS
    assert "already read by an earlier pass" in INSTRUCTIONS


def test_a_bad_action_is_an_error_the_client_can_read(cfg):
    async def script(client):
        sid = session_of(text(await client.call_tool("start_session",
                                                     {"goal": "x"})))
        return await client.call_tool("act", {"session_id": sid,
                                              "actions": [{"action": "tap"}]})

    result = drive(cfg, fake.FakeDevice(cfg), script)
    assert result.is_error and "requires a target" in text(result)


def test_look_comes_with_a_screenshot(cfg):
    async def script(client):
        sid = session_of(text(await client.call_tool("start_session",
                                                     {"goal": "x"})))
        return await client.call_tool("look", {"session_id": sid, "grid": True})

    result = drive(cfg, fake.FakeDevice(cfg), script)
    kinds = [block.type for block in result.content]
    assert kinds == ["text", "image"]
    assert result.content[1].mime_type == "image/jpeg"


def test_a_second_driver_is_turned_away(cfg):
    holder = phonelock.PhoneLock(cfg.run.artifacts_dir,
                                 "adbagent watch: reply to new chats")
    holder.acquire()
    try:
        async def script(client):
            return await client.call_tool("start_session", {"goal": "x"})

        result = drive(cfg, fake.FakeDevice(cfg), script)
    finally:
        holder.release()
    assert result.is_error
    assert "adbagent watch: reply to new chats" in text(result)


def test_a_second_conversation_can_neither_drive_nor_end_the_session(cfg):
    """Seen with Antigravity, which runs one server for all of its chats: a
    scheduled run found the phone taken, was told "end_session first", and
    ended another conversation's session halfway through its pass. Both come
    down one connection here too, as they did there."""
    dev = fake.FakeDevice(cfg)

    async def script(client):
        started = text(await client.call_tool(
            "start_session", {"goal": "open the Wi-Fi settings"}))
        mine = session_of(started)
        wifi = index_of(started, "Wi-Fi")
        refused = await client.call_tool("start_session",
                                         {"goal": "something else"})
        theirs = []
        for guess in ("", "0" * 8):
            theirs += [
                await client.call_tool("act", tap(guess, wifi)),
                await client.call_tool("look", {"session_id": guess}),
                await client.call_tool("end_session", {
                    "session_id": guess, "skip_remaining": True,
                    "summary": "Ended previous leftover session to start a "
                               "clean pass."}),
            ]
        acted = text(await client.call_tool("act", tap(mine, wifi)))
        ended = text(await client.call_tool(
            "end_session", {"session_id": mine,
                            "summary": "Wi-Fi settings are open"}))
        after = text(await client.call_tool("start_session",
                                            {"goal": "something else"}))
        stale = await client.call_tool("act", {"session_id": mine, "actions": [
            {"action": "press_key", "key": "back"}]})
        return mine, refused, theirs, acted, ended, after, stale

    mine, refused, theirs, acted, ended, after, stale = drive(cfg, dev, script)
    said = text(refused)
    assert refused.is_error and "in use by another session" in said
    assert "open the Wi-Fi settings" in said
    assert "end_session first" not in said
    for result in [refused] + theirs:
        assert result.is_error and "in use by another session" in text(result)
        assert mine not in text(result)       # the id is what would let them in
    assert "-> success" in acted and dev.state == "wifi"
    assert len(dev.taps) == 1                 # the owner's, and nobody else's
    assert ended.startswith("SESSION ENDED: success")
    assert "Wi-Fi settings are open" in ended
    # Once it is over the phone is free, and the old id opens nothing new.
    assert session_of(after) != mine
    assert stale.is_error and "in use by another session" in text(stale)
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


@pytest.mark.parametrize("mode", ERAS)
def test_only_the_owner_can_put_an_approval_to_the_person(cfg, mode):
    dev = fake.FakeDevice(cfg)
    person = Person("accept")

    async def script(client):
        started = text(await client.call_tool("start_session",
                                              {"goal": "forget the network"}))
        mine = session_of(started)
        wifi = text(await client.call_tool(
            "act", tap(mine, index_of(started, "Wi-Fi"))))
        asked = text(await client.call_tool(
            "act", tap(mine, index_of(wifi, "Forget network"))))
        request = int(re.search(r"request_id (\d+)", asked).group(1))
        return await client.call_tool("approve", {"session_id": "0" * 8,
                                                  "request_id": request})

    theirs = drive(cfg, dev, script, mode=mode, person=person)
    assert theirs.is_error and "in use by another session" in text(theirs)
    assert person.asked == []                 # nobody asked on their say-so
    assert len(dev.taps) == 1                 # and nothing pressed


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

def _forget(person, mode, cfg):
    dev = fake.FakeDevice(cfg)

    async def script(client):
        started = text(await client.call_tool("start_session",
                                              {"goal": "forget the network"}))
        sid = session_of(started)
        wifi = text(await client.call_tool(
            "act", tap(sid, index_of(started, "Wi-Fi"))))
        asked = text(await client.call_tool(
            "act", tap(sid, index_of(wifi, "Forget network"))))
        if "APPROVAL NEEDED" not in asked:
            return asked, ""
        request = int(re.search(r"request_id (\d+)", asked).group(1))
        approved = await client.call_tool("approve", {"session_id": sid,
                                                      "request_id": request})
        return asked, text(approved)

    asked, approved = drive(cfg, dev, script, mode=mode, person=person)
    return dev, asked, approved


@pytest.mark.parametrize("mode", ERAS)
def test_the_person_approves_an_irreversible_press(cfg, mode):
    person = Person("accept")
    dev, asked, approved = _forget(person, mode, cfg)
    assert "APPROVAL NEEDED (request_id 1)" in asked
    assert len(person.asked) == 1 and "Forget network" in person.asked[0]
    assert len(dev.taps) == 2
    assert '"Forget network"' in approved


@pytest.mark.parametrize("mode", ERAS)
def test_the_person_declines_an_irreversible_press(cfg, mode):
    dev, _, approved = _forget(Person("decline"), mode, cfg)
    assert len(dev.taps) == 1
    assert "refused to press 'Forget network'" in approved


def test_a_client_that_cannot_ask_gets_the_press_declined(cfg):
    """No elicitation, so nobody can be asked: the press does not happen."""
    dev, asked, _ = _forget(None, "legacy", cfg)
    assert "APPROVAL NEEDED" not in asked
    assert "cannot ask the person" in asked and "--allow-destructive" in asked
    assert len(dev.taps) == 1


# ---------------------------------------------------------------------------
# Handing the phone over
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ERAS)
def test_hand_over_gives_the_phone_back_when_the_person_is_done(cfg, mode):
    dev = fake.FakeDevice(cfg)
    person = Person("accept")

    async def script(client):
        sid = session_of(text(await client.call_tool("start_session",
                                                     {"goal": "x"})))
        return text(await client.call_tool(
            "hand_over", {"session_id": sid,
                          "reason": "enter the one-time code"}))

    back = drive(cfg, dev, script, mode=mode, person=person)
    assert len(person.asked) == 1
    assert "enter the one-time code" in person.asked[0]
    assert dev.closes >= 1 and dev.opens == 1
    assert "the person took the phone" in back
    assert "CURRENT SCREEN" in back


def test_hand_over_declined_ends_the_session(cfg):
    dev = fake.FakeDevice(cfg)

    async def script(client):
        sid = session_of(text(await client.call_tool("start_session",
                                                     {"goal": "x"})))
        return text(await client.call_tool(
            "hand_over", {"session_id": sid, "reason": "pick an account"}))

    ended = drive(cfg, dev, script, person=Person("decline"))
    assert ended.startswith("SESSION ENDED: aborted")
    assert "the person kept the phone" in ended
    assert dev.opens == 0


def test_hand_over_needs_a_client_that_can_ask(cfg):
    dev = fake.FakeDevice(cfg)

    async def script(client):
        sid = session_of(text(await client.call_tool("start_session",
                                                     {"goal": "x"})))
        return await client.call_tool("hand_over", {"session_id": sid,
                                                    "reason": "log in"})

    result = drive(cfg, dev, script, person=None)
    assert result.is_error and "cannot ask the person" in text(result)
    assert dev.closes == 1          # only the session's own end, never a release


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

LOUD_SERVER = """
import sys
sys.path.insert(0, {repo!r})
from adbagent.config import Config
from adbagent.mcp_server import serve
from tests import fake

class Loud(fake.FakeDevice):
    def tap(self, x, y):
        print(f"Tap at ({{x}}, {{y}})")     # what Device._log_action does
        super().tap(x, y)

cfg = Config()
cfg.run.artifacts_dir = sys.argv[1]
dev = Loud(cfg)
serve(cfg, open_device=lambda _cfg: dev, idle_timeout_s=60)
"""


def test_a_print_from_the_phone_never_reaches_the_protocol(tmp_path, caplog):
    """Seen against a real phone: `Device` prints every action, and a buffered
    stdout flushed those lines into the protocol stream at exit. They belong on
    stderr, at once."""
    from pathlib import Path

    from mcp.client.stdio import stdio_client

    repo = str(Path(__file__).resolve().parent.parent)
    script = tmp_path / "loud_server.py"
    script.write_text(LOUD_SERVER.format(repo=repo), encoding="utf-8")
    errlog = tmp_path / "server.err"
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(script), str(tmp_path / "runs")],
                                   cwd=repo)

    async def main():
        with open(errlog, "w", encoding="utf-8") as err:
            async with Client(stdio_client(params, errlog=err)) as client:
                started = text(await client.call_tool("start_session",
                                                      {"goal": "x"}))
                sid = session_of(started)
                await client.call_tool("act",
                                       tap(sid, index_of(started, "Wi-Fi")))
                during = errlog.read_text(encoding="utf-8")
                await client.call_tool("end_session", {"session_id": sid,
                                                       "summary": "done"})
        return during

    with caplog.at_level("ERROR"):
        during = anyio.run(main)
    assert "Tap at (" in during
    assert not [r for r in caplog.records if "parse" in r.getMessage().lower()]

def test_adbagent_mcp_speaks_the_protocol_on_stdio(tmp_path):
    """The real command in a real subprocess: nothing but protocol on stdout."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "adbagent", "mcp", "--artifacts-dir", str(tmp_path / "runs")],
        cwd=str(tmp_path))

    async def main():
        async with Client(params) as client:
            return [t.name for t in (await client.list_tools()).tools]

    names = anyio.run(main)
    assert "start_session" in names and "act" in names


def _with_policy(cfg, tmp_path, front: str):
    cfg.watch.policies_dir = str(tmp_path / "policies")
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "demo.md").write_text(
        f"---\n{front}\n---\n\n# Demo\n- Never reply twice.\n", encoding="utf-8")


def test_a_policy_makes_the_session_one_pass_of_its_watch(cfg, tmp_path):
    _with_policy(cfg, tmp_path, "goal: open the Wi-Fi settings\nsend_limits: like=5")
    dev = fake.FakeDevice(cfg)
    phone = {}

    async def script(client):
        return text(await client.call_tool(
            "start_session", {"policy": "demo", "send_limits": "like=3, rose=0"}))

    server, phone_ = build_server(cfg, open_device=lambda _cfg: dev,
                                  idle_timeout_s=30)

    async def main():
        async with Client(server) as client:
            phone["first"] = await script(client)
            phone["limits"] = dict(phone_.session.send_limits)

    try:
        anyio.run(main)
    finally:
        phone_.close()
    first = phone["first"]
    assert "GOAL:\nopen the Wi-Fi settings" in first
    assert "ONE PASS" in first                    # the watch's pass contract
    assert "REPLY POLICY" in first and "Never reply twice." in first
    assert phone["limits"] == {"like": 3, "rose": 0}


def test_a_client_cannot_raise_a_policys_send_cap(cfg, tmp_path):
    _with_policy(cfg, tmp_path, "goal: x\nsend_limits: like=2")
    server, phone = build_server(cfg, open_device=lambda _cfg: fake.FakeDevice(cfg),
                                 idle_timeout_s=30)

    async def main():
        async with Client(server) as client:
            await client.call_tool("start_session",
                                   {"policy": "demo", "send_limits": "like=9"})
            return dict(phone.session.send_limits)

    try:
        assert anyio.run(main) == {"like": 2}
    finally:
        phone.close()


def test_a_session_needs_a_goal_from_somewhere(cfg):
    async def script(client):
        return await client.call_tool("start_session", {})

    result = drive(cfg, fake.FakeDevice(cfg), script)
    assert result.is_error and "give a goal" in text(result)


# ---------------------------------------------------------------------------
# Several policies, one pass of each in turn
# ---------------------------------------------------------------------------

def _two_policies(cfg, tmp_path, second_goal="open the Bluetooth settings"):
    cfg.watch.policies_dir = str(tmp_path / "policies")
    folder = tmp_path / "policies"
    folder.mkdir()
    (folder / "wifi.md").write_text(
        "---\ngoal: open the Wi-Fi settings\ntitle: Wi-Fi\n"
        "send_limits: like=5\n---\n\n# Wi-Fi\n- Never reply twice.\n",
        encoding="utf-8")
    goal_line = f"goal: {second_goal}\n" if second_goal else ""
    (folder / "bluetooth.md").write_text(
        f"---\n{goal_line}title: Bluetooth\n---\n\n# Bluetooth\n"
        f"- Only ever say hi.\n", encoding="utf-8")


def test_several_policies_are_a_pass_each_in_one_session(cfg, tmp_path):
    _two_policies(cfg, tmp_path)
    dev = fake.FakeDevice(cfg)

    async def script(client):
        started = text(await client.call_tool(
            "start_session", {"policies": ["wifi", "bluetooth"]}))
        sid = session_of(started)
        handed_on = text(await client.call_tool(
            "end_session", {"session_id": sid, "summary": "Wi-Fi is open"}))
        ended = text(await client.call_tool(
            "end_session", {"session_id": sid, "summary": "Bluetooth is open"}))
        return started, handed_on, ended

    started, handed_on, ended = drive(cfg, dev, script)
    assert "POLICY 1 of 2: Wi-Fi" in started
    assert "GOAL:\nopen the Wi-Fi settings" in started
    assert "ONE PASS" in started                 # each is a pass of its watch
    assert "Never reply twice." in started
    assert "POLICY 2 of 2: Bluetooth" in handed_on
    assert "GOAL:\nopen the Bluetooth settings" in handed_on
    assert "Only ever say hi." in handed_on
    assert "policy 1 of 2 (Wi-Fi) ended: success -- Wi-Fi is open" in handed_on
    assert ended.startswith("SESSION ENDED: success")
    assert "policy 2 of 2 (Bluetooth) ended: success" in ended
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


def test_skip_remaining_gives_the_phone_back_early(cfg, tmp_path):
    _two_policies(cfg, tmp_path)

    async def script(client):
        sid = session_of(text(await client.call_tool(
            "start_session", {"policies": ["wifi", "bluetooth"]})))
        return text(await client.call_tool(
            "end_session", {"session_id": sid, "summary": "enough",
                            "skip_remaining": True}))

    ended = drive(cfg, fake.FakeDevice(cfg), script)
    assert ended.startswith("SESSION ENDED: success")
    assert "not run: Bluetooth -- end_session skipped them" in ended
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


def test_list_policies_names_each_with_the_goal_it_runs_under(cfg, tmp_path):
    _two_policies(cfg, tmp_path, second_goal="")

    async def script(client):
        return text(await client.call_tool("list_policies", {}))

    listed = drive(cfg, fake.FakeDevice(cfg), script)
    assert "- wifi (Wi-Fi): open the Wi-Fi settings [send_limits: like=5]" \
        in listed
    assert "- bluetooth (Bluetooth): no goal saved" in listed


@pytest.mark.parametrize("args, says", [
    ({"policies": ["wifi", "bluetooth"], "goal": "do it all"},
     "leave `goal` empty"),
    ({"policies": ["wifi", "bluetooth"], "policy": "wifi"},
     "not both"),
    ({"policies": ["wifi", "wifi"]}, "named twice"),
])
def test_what_several_policies_will_not_take(cfg, tmp_path, args, says):
    _two_policies(cfg, tmp_path)

    async def script(client):
        return await client.call_tool("start_session", args)

    result = drive(cfg, fake.FakeDevice(cfg), script)
    assert result.is_error and says in text(result)
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


def test_one_of_several_with_no_goal_saved_is_refused(cfg, tmp_path):
    _two_policies(cfg, tmp_path, second_goal="")

    async def script(client):
        return await client.call_tool("start_session",
                                      {"policies": ["wifi", "bluetooth"]})

    result = drive(cfg, fake.FakeDevice(cfg), script)
    assert result.is_error and "carries no goal" in text(result)


def test_the_instructions_say_when_a_quiet_session_lets_go(cfg):
    from adbagent.mcp_server import instructions

    said = instructions(cfg)
    assert "left without a call for 5 minutes lets go of the phone" in said
    assert (said.index("ONE PHONE") < said.index("lets go of the phone")
            < said.index("POLICIES"))
    cfg.device.let_go_after_s = 0
    assert "lets go" not in instructions(cfg)


def test_a_call_to_a_phone_taken_meanwhile_says_who_has_it(cfg):
    """The session let go of the phone while its driver was quiet, and a run
    took it. The driver's next call does nothing and says so; once the phone
    is free, the call after takes it back."""
    cfg.device.let_go_after_s = 0.2
    other = phonelock.PhoneLock(cfg.run.artifacts_dir,
                                "adbagent run: turn on wifi")

    async def script(client):
        started = text(await client.call_tool(
            "start_session", {"goal": "open the Wi-Fi settings"}))
        sid = session_of(started)
        await anyio.sleep(0.8)
        other.acquire()
        refused = await client.call_tool(
            "act", tap(sid, index_of(started, "Wi-Fi")))
        other.release()
        acted = await client.call_tool(
            "act", tap(sid, index_of(started, "Wi-Fi")))
        return refused, acted

    dev = fake.FakeDevice(cfg)
    try:
        refused, acted = drive(cfg, dev, script)
    finally:
        other.release()
    assert refused.is_error
    assert "adbagent run: turn on wifi" in text(refused)
    assert "Nothing was done" in text(refused)
    assert not acted.is_error and "-> success" in text(acted)
    assert "took it back for this call" in text(acted)
    assert dev.state == "wifi" and len(dev.taps) == 1
