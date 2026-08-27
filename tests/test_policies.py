"""Policy files: the front matter, and the directory of them.

The property that matters most here is negative: front matter must never reach
the prompt, and a policy that has none must survive a round trip through the
editor byte for byte. Everything else is a picker convenience.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adbagent import policies
from adbagent.policies import Policy, PolicyStore
from adbagent.watch import load_policy

WITH_GOAL = """\
---
goal: work through the feed and reply to anyone new
title: Hinge
---

# Hinge policy

- Only ever like the first photo.
"""


# ---------------------------------------------------------------------------
# front matter
# ---------------------------------------------------------------------------

def test_front_matter_is_split_off():
    meta, body = policies.split_front_matter(WITH_GOAL)
    assert meta == {"goal": "work through the feed and reply to anyone new",
                    "title": "Hinge"}
    assert body.startswith("# Hinge policy")


def test_a_file_without_front_matter_is_returned_untouched():
    text = "reply only to people I follow\n"
    assert policies.split_front_matter(text) == ({}, text)
    assert policies.goal_in(text) == ""


def test_a_leading_horizontal_rule_is_not_front_matter():
    """The failure this guards is silently eating the first paragraph of
    somebody's instructions, and the instructions are what talks to people."""
    text = "---\n\nReply only to people I follow.\n\n---\n"
    meta, body = policies.split_front_matter(text)
    assert meta == {} and body == text


def test_an_unclosed_fence_is_not_front_matter():
    text = "---\ngoal: something\n\nreply to everyone\n"
    assert policies.split_front_matter(text) == ({}, text)


def test_prose_inside_the_fence_is_not_front_matter():
    text = "---\ngoal: fine\nbut this line is not a key\n---\n\nbody\n"
    assert policies.split_front_matter(text) == ({}, text)


def test_a_multi_line_goal_round_trips():
    goal = "watch my instagram dms\nand reply to anyone I follow"
    text = policies.with_front_matter({"goal": goal}, "be brief")
    assert policies.split_front_matter(text) == ({"goal": goal}, "be brief")


def test_no_metadata_means_no_fence_at_all():
    """A policy with no goal saved from the editor is the file that was typed."""
    assert policies.with_front_matter({}, "be brief") == "be brief"
    assert policies.with_front_matter({"goal": "  "}, "be brief") == "be brief"


def test_a_generated_file_ends_in_a_newline():
    assert policies.with_front_matter({"goal": "g"}, "be brief").endswith(
        "---\n\nbe brief\n")


def test_instructions_are_what_reaches_the_prompt():
    assert "goal:" not in policies.instructions(WITH_GOAL)
    assert policies.instructions(WITH_GOAL).startswith("# Hinge policy")


def test_load_policy_strips_the_front_matter(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text(WITH_GOAL, encoding="utf-8")
    text = load_policy(str(p))
    assert "work through the feed" not in text
    assert text.startswith("# Hinge policy")


def test_a_policy_that_is_only_front_matter_is_empty(tmp_path):
    """No instructions is the same failure as an empty file: a loop deciding for
    itself what to say to people."""
    p = tmp_path / "policy.md"
    p.write_text("---\ngoal: reply to things\n---\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(str(p))


def test_unknown_keys_survive_a_save(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("---\ngoal: g\nauthor: someone\n---\n\nbody\n", encoding="utf-8")
    store = PolicyStore(str(tmp_path))
    store.write(str(p), "new body", goal="g2")
    meta, body = policies.split_front_matter(p.read_text(encoding="utf-8"))
    assert meta == {"goal": "g2", "author": "someone"}
    assert body == "new body"


def test_a_save_that_sends_no_title_keeps_the_one_on_disk(tmp_path):
    """The editor has no title field. A form that does not send one is not a
    request to drop what somebody wrote into the file by hand."""
    p = tmp_path / "policy.md"
    p.write_text("---\ngoal: g\ntitle: Hinge\n---\n\nbody\n", encoding="utf-8")
    PolicyStore(str(tmp_path)).write(str(p), "new body", goal="g2")
    assert policies.read(p).title == "Hinge"


def test_an_emptied_goal_box_clears_the_goal(tmp_path):
    """The other direction: the goal *is* edited from the UI, so an empty one
    means what it says."""
    p = tmp_path / "policy.md"
    p.write_text("---\ngoal: g\n---\n\nbody\n", encoding="utf-8")
    PolicyStore(str(tmp_path)).write(str(p), "body", goal="")
    assert policies.read(p).goal == ""
    assert p.read_text(encoding="utf-8") == "body"


def test_slug_refuses_to_leave_the_directory():
    assert policies.slug("../../etc/passwd") == "etc-passwd"
    assert policies.slug("hinge.md") == "hinge"
    assert policies.slug("  ") == ""


def test_title_falls_back_to_the_first_heading():
    p = Policy(path=Path("x.md"), name="x", body="# Instagram DMs\n\n- be brief")
    assert p.label == "Instagram DMs"


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

def make(dirpath: Path, name: str, goal: str = "", body: str = "be brief") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / f"{name}.md"
    path.write_text(policies.with_front_matter({"goal": goal}, body),
                    encoding="utf-8")
    return path


def test_list_reads_every_policy_with_its_goal(tmp_path):
    make(tmp_path / "policies", "hinge", goal="work the feed")
    make(tmp_path / "policies", "insta", goal="watch my dms")
    found = PolicyStore(str(tmp_path / "policies")).list()
    assert [(p.name, p.goal) for p in found] == [
        ("hinge", "work the feed"), ("insta", "watch my dms")]


def test_list_includes_the_configured_policy_from_outside_the_directory(tmp_path):
    make(tmp_path / "policies", "hinge")
    outside = make(tmp_path, "legacy")
    found = PolicyStore(str(tmp_path / "policies"), str(outside)).list()
    assert sorted(p.name for p in found) == ["hinge", "legacy"]


def test_list_includes_a_configured_policy_that_is_not_written_yet(tmp_path):
    """It is what a bare `adbagent watch` runs. A picker that hid it would have
    no way back to the policy that is actually configured."""
    store = PolicyStore(str(tmp_path / "policies"), str(tmp_path / "new.md"))
    assert [(p.name, p.exists) for p in store.list()] == [("new", False)]


def test_a_bare_name_resolves_into_the_policies_directory(tmp_path):
    path = make(tmp_path / "policies", "hinge")
    store = PolicyStore(str(tmp_path / "policies"))
    assert store.resolve("hinge") == str(path)
    assert store.resolve("hinge.md") == str(path)


def test_a_filename_from_the_wrong_directory_still_finds_it(tmp_path, monkeypatch):
    """`--policy hinge_policy.md` from the repo root, with the file in
    `policies/`: it named a file, just not from here."""
    path = make(tmp_path / "policies", "hinge_policy")
    monkeypatch.chdir(tmp_path)
    store = PolicyStore("policies")
    assert Path(store.resolve("hinge_policy.md")).name == path.name


def test_resolve_with_nothing_set_is_empty_not_a_directory(tmp_path):
    """`Path("")` is `Path(".")`, whose truthiness makes every "no policy" guard
    downstream pass and lands a write on a directory."""
    assert PolicyStore(str(tmp_path)).resolve() == ""


def test_resolve_falls_back_to_the_configured_policy(tmp_path):
    path = make(tmp_path, "one")
    assert PolicyStore("", str(path)).resolve() == str(path)


def test_owns_only_the_directory_and_the_configured_file(tmp_path):
    store = PolicyStore(str(tmp_path / "policies"), str(tmp_path / "legacy.md"))
    assert store.owns(tmp_path / "policies" / "hinge.md")
    assert store.owns(tmp_path / "legacy.md")
    assert not store.owns(tmp_path / "elsewhere.md")
    assert not store.owns(tmp_path / "policies" / "hinge.py")
    assert not store.owns(tmp_path / "policies" / "nested" / "hinge.md")


def test_write_saves_the_goal_beside_the_instructions(tmp_path):
    store = PolicyStore(str(tmp_path / "policies"))
    saved = store.write("hinge", "be brief", goal="watch my dms")
    assert saved.path == tmp_path / "policies" / "hinge.md"
    reread = policies.read(saved.path)
    assert reread.goal == "watch my dms" and reread.body == "be brief"


def test_reading_a_policy_that_is_not_there_is_not_an_error(tmp_path):
    policy = policies.read(tmp_path / "nope.md")
    assert policy.exists is False and policy.body == "" and policy.goal == ""


def test_store_for_reads_the_config(tmp_path):
    from adbagent.config import Config
    cfg = Config()
    cfg.watch.policies_dir = str(tmp_path / "policies")
    cfg.watch.policy = str(tmp_path / "one.md")
    store = policies.store_for(cfg)
    assert store.dir == tmp_path / "policies"
    assert store.current == tmp_path / "one.md"
