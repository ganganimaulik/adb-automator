import json

import pytest

from adbagent.actions import AgentAction
from adbagent.config import Config
from adbagent.memory import Memory
from adbagent.skills import (DEFAULT_EXPLORE_STEPS, MIN_LEARNABLE_STEPS,
                             AppTrace, ExplorationBlocked, Skill,
                             SkillGenerator, SkillRegistry, TraceCollector,
                             Workflow, exploration_goal, explore_app,
                             goal_app_candidates, learn_from_run,
                             package_from_text, resolve_package)
from adbagent.cli import main

from . import fake


def test_skill_serialization(tmp_path):
    skill = Skill(
        name="TestApp",
        packages=["com.example.testapp"],
        aliases=["testapp", "test"],
        description="A test app for verification.",
        workflows=[Workflow(name="search", steps="1. Tap search. 2. Type text.")],
        nuances=["Search bar requires enter key."],
        recommendations=["Clear input before typing."]
    )

    d = skill.to_dict()
    assert d["name"] == "TestApp"
    assert d["packages"] == ["com.example.testapp"]
    assert d["workflows"][0]["name"] == "search"

    restored = Skill.from_dict(d)
    assert restored.name == skill.name
    assert restored.packages == skill.packages
    assert restored.workflows[0].steps == skill.workflows[0].steps


def test_skill_markdown_conversion():
    md = """# SampleApp

**Packages**: com.sample.app, com.sample.beta
**Aliases**: sample, sampleapp

## Description
This is a sample application for testing markdown parsing.

## App Nuances & UI Quirks
- Tapping back floating icon exits to home screen.
- Long list items require fast scrolling.

## Recommendations & Best Practices
- Prefer search bar autocomplete suggestions.
"""
    skill = Skill.from_markdown(md)
    assert skill.name == "SampleApp"
    assert "com.sample.app" in skill.packages
    assert "sample" in skill.aliases
    assert "sample application" in skill.description
    assert len(skill.nuances) == 2
    assert len(skill.recommendations) == 1
    assert "search bar autocomplete" in skill.recommendations[0]


def test_skill_matching():
    skill = Skill(
        name="WhatsApp",
        packages=["com.whatsapp", "com.whatsapp.w4b"],
        aliases=["whatsapp", "wa"],
        description="Messaging app",
        nuances=["Search top bar first."]
    )

    assert skill.matches_package("com.whatsapp")
    assert skill.matches_package("com.whatsapp.w4b")
    assert not skill.matches_package("com.other.app")

    assert skill.matches_query("whatsapp")
    assert skill.matches_query("wa")
    assert skill.matches_query("WhatsApp")

    assert skill.matches_goal("Open WhatsApp and send message")
    assert skill.matches_goal("Send wa message to John")
    assert not skill.matches_goal("Play jazz on Spotify")


def test_skill_registry(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Save a JSON skill
    sk1 = Skill(name="AppOne", packages=["com.app.one"], aliases=["one"])
    p1 = skills_dir / "appone.json"
    p1.write_text(json.dumps(sk1.to_dict()), encoding="utf-8")

    registry = SkillRegistry(skills_dir)
    assert len(registry.list_skills()) == 1

    matched = registry.find_by_package("com.app.one")
    assert matched is not None
    assert matched.name == "AppOne"

    goal_matched = registry.find_for_run("", "Open AppOne and search")
    assert goal_matched is not None
    assert goal_matched.name == "AppOne"


def _registry_with(*skills: Skill) -> SkillRegistry:
    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = {s.name.lower(): s for s in skills}
    return registry


def test_find_for_run_prefers_the_app_the_goal_names():
    """What the last session left on screen is not what the run is for: a price
    comparison starting on top of Bumble must not load Bumble's skill."""
    bumble = Skill(name="Bumble", packages=["com.bumble.app"], aliases=["bumble"])
    zepto = Skill(name="Zepto", packages=["com.zeptoconsumerapp"], aliases=["zepto"])
    blinkit = Skill(name="Blinkit", packages=["com.grofers.customerapp"],
                    aliases=["blinkit"])
    registry = _registry_with(bumble, zepto, blinkit)
    goal = "compare price of coconut water on zepto and blinkit"

    # Bumble in front, both goal apps elsewhere: a goal-named skill wins.
    picked = registry.find_for_run("com.bumble.app", goal, goal_names_app=True)
    assert picked is not None and picked.name in ("Zepto", "Blinkit")

    # And a goal spanning two apps follows the one on screen.
    assert registry.find_for_run(
        "com.zeptoconsumerapp", goal, goal_names_app=True).name == "Zepto"
    assert registry.find_for_run(
        "com.grofers.customerapp", goal, goal_names_app=True).name == "Blinkit"


def test_find_for_run_falls_back_to_the_foreground_app():
    """A vague goal makes the open app the likely subject, so its skill loads."""
    bumble = Skill(name="Bumble", packages=["com.bumble.app"], aliases=["bumble"])
    registry = _registry_with(bumble)
    assert registry.find_for_run("com.bumble.app", "read my matches").name == "Bumble"


def test_find_for_run_no_fallback_when_the_goal_named_other_apps():
    """The goal pointed at apps that have no skill yet: the leftover app's
    skill is noise, and nothing loading is the right answer."""
    bumble = Skill(name="Bumble", packages=["com.bumble.app"], aliases=["bumble"])
    registry = _registry_with(bumble)
    picked = registry.find_for_run(
        "com.bumble.app", "compare price of coconut water on zepto and blinkit",
        goal_names_app=True)
    assert picked is None


def test_goal_app_candidates():
    class Dev:
        def list_apps(self):
            return ["com.bumble.app", "com.grofers.customerapp",
                    "com.whatsapp", "com.zeptoconsumerapp"]

    found = goal_app_candidates(
        Dev(), "compare price of yu coconut water 1 liter on zepto and blinkit")
    # "zepto" is a substring of its package; "blinkit" appears nowhere in
    # `com.grofers.customerapp`, which is the documented limit of a
    # package-name-only lookup.
    assert found == ["com.zeptoconsumerapp"]
    # A goal naming no app names none -- including the instruction words.
    assert goal_app_candidates(Dev(), "read my unread messages") == []
    # CamelCase and punctuation still reach the package.
    assert goal_app_candidates(Dev(), "open BumbleApp, then tour it") == \
        ["com.bumble.app"]


def test_skill_prompt_text():
    skill = Skill(
        name="Spotify",
        packages=["com.spotify.music"],
        description="Music player",
        workflows=[Workflow(name="search_song", steps="1. Tap search.")],
        nuances=["Audio ads block player."],
        recommendations=["Check mini player bar."]
    )

    prompt_text = skill.to_prompt_text()
    assert "APP SKILL & GUIDANCE (Spotify):" in prompt_text
    assert "search_song" in prompt_text
    assert "Audio ads block player." in prompt_text


class MockLLM:
    def __init__(self, response_text):
        self.response_text = response_text
        self.model = "mock-model"

    def _post_chat(self, messages, model=None):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.response_text
                    }
                }
            ]
        }


GENERATED = {
    "name": "GeneratedApp",
    "packages": ["com.generated.app"],
    "aliases": ["genapp"],
    "description": "Auto generated skill description",
    "workflows": [{"name": "task1", "steps": "Step 1 details"}],
    "nuances": ["Nuance 1"],
    "recommendations": ["Recommendation 1"],
}


def test_skill_generator_saves_what_the_model_returned(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    skill = SkillGenerator(registry).generate_from_exploration(
        "com.generated.app", tasks="open the app and look around",
        screen_summaries=["home screen"], actions_taken=["tap #1"],
        llm_client=MockLLM(json.dumps(GENERATED)))

    assert skill.name == "GeneratedApp"
    assert skill.packages == ["com.generated.app"]
    assert [w.name for w in skill.workflows] == ["task1"]
    # Saved to disk and available to the next run, which is the point of the
    # command -- an in-memory skill helps nobody.
    assert (tmp_path / "skills" / "generatedapp.json").is_file()
    assert SkillRegistry(tmp_path / "skills").find_by_package(
        "com.generated.app") is not None


def test_a_fenced_reply_is_still_parsed(tmp_path):
    """Models wrap JSON in ```json fences whatever the prompt says."""
    registry = SkillRegistry(tmp_path / "skills")
    skill = SkillGenerator(registry).generate_from_exploration(
        "com.generated.app", tasks="t", screen_summaries=[], actions_taken=[],
        llm_client=MockLLM("```json\n" + json.dumps(GENERATED) + "\n```"))
    assert skill.name == "GeneratedApp"


def test_an_unparseable_reply_falls_back_instead_of_raising(tmp_path):
    """An exploration run that ends in a bad reply should still leave a skill
    holding the tasks it was given, not lose the whole session."""
    registry = SkillRegistry(tmp_path / "skills")
    skill = SkillGenerator(registry).generate_from_exploration(
        "com.generated.app", tasks="open the app",
        screen_summaries=[], actions_taken=[],
        llm_client=MockLLM("I'm afraid I can't do that"))
    assert skill.packages == ["com.generated.app"]
    assert "open the app" in skill.workflows[0].steps


def test_generating_again_merges_into_the_existing_skill(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    generator = SkillGenerator(registry)
    generator.generate_from_exploration(
        "com.generated.app", tasks="t", screen_summaries=[], actions_taken=[],
        llm_client=MockLLM(json.dumps(GENERATED)))

    second = dict(GENERATED, nuances=["Nuance 2"],
                  workflows=[{"name": "task2", "steps": "Step 2"}])
    merged = generator.generate_from_exploration(
        "com.generated.app", tasks="t", screen_summaries=[], actions_taken=[],
        llm_client=MockLLM(json.dumps(second)))

    assert set(merged.nuances) == {"Nuance 1", "Nuance 2"}
    assert {w.name for w in merged.workflows} == {"task1", "task2"}


def test_merging_drops_a_reworded_restatement(tmp_path):
    """Two runs word the same quirk differently. Without this, a skill
    regenerated twenty times carries twenty phrasings of one nuance."""
    first = Skill(name="AppX", nuances=[
        "The People tab shows a swipeable card deck; swiping acts on real profiles."])
    second = Skill(name="AppX", nuances=[
        "The People tab shows a swipeable card deck with 'New here' badges; "
        "swiping acts on real profiles."])
    merged = first.merge(second)

    assert len(merged.nuances) == 1
    assert "New here" in merged.nuances[0]      # the richer wording survives


def test_merging_keeps_two_nuances_that_each_add_something(tmp_path):
    """Overlap is not restatement. Collapsing these would lose the clipboard
    popup or the keyboard, and each is a real finding."""
    first = Skill(name="AppX", nuances=[
        "The search screen shows 'No results' when the field is empty; a system "
        "clipboard popup may briefly overlay the bottom of the screen."])
    second = Skill(name="AppX", nuances=[
        "The search screen shows 'No results' when the field is empty; the soft "
        "keyboard opens automatically when the screen appears."])
    merged = first.merge(second)

    assert len(merged.nuances) == 2


def test_two_wordings_of_one_finding_collapse_even_without_containment():
    """Real pair from a regenerated skill, at 0.75 overlap. Neither contains the
    other -- "icon"/"tapped" against "opened" -- so containment kept both."""
    from adbagent.skills import collapse_restatements

    kept = collapse_restatements([
        "Search top bar icon must be tapped before typing contact names; typing "
        "without opening search will not find chats.",
        "Search top bar must be opened before typing contact names; typing "
        "without opening search will not find chats.",
    ])
    assert len(kept) == 1


def test_a_short_nuance_is_not_swallowed_by_a_long_unrelated_one():
    from adbagent.skills import collapse_restatements

    entries = [
        "Clear the search input before searching again.",
        "The media viewer toolbar auto-hides after a few seconds, and once it "
        "does the tree collapses to the image scroller alone, so there is no way "
        "to tell which photo is shown; tap the photo once to bring the toolbar "
        "and its title back, then read the title to identify the current item.",
    ]
    assert collapse_restatements(entries) == entries


def test_merging_keeps_the_workflow_that_says_more_not_the_newest():
    """This used to be "the newest wins", so one thin run could replace a
    detailed procedure with "Tap send." in the file the next run obeys."""
    detailed = "1. Open chat. 2. Tap the input box. 3. Type. 4. Tap the green Send button."
    existing = Skill(name="AppX", workflows=[Workflow("send_message", detailed)])
    thin = Skill(name="AppX", workflows=[Workflow("send_message", "Tap send.")])

    assert existing.merge(thin).workflows[0].steps == detailed
    # And a genuinely fuller version does replace the shorter one.
    fuller = Skill(name="AppX", workflows=[
        Workflow("send_message", detailed + " 5. Check the tick appears.")])
    assert "tick" in existing.merge(fuller).workflows[0].steps


def test_a_renamed_workflow_does_not_become_a_second_copy():
    steps = "Open the chat, tap the input box, type, tap the green Send button."
    existing = Skill(name="AppX", workflows=[Workflow("send_message", steps)])
    renamed = Skill(name="AppX", workflows=[Workflow("send_a_message", steps + " ")])

    merged = existing.merge(renamed)
    assert len(merged.workflows) == 1
    assert merged.workflows[0].name == "send_message"   # no name churn


def test_a_genuinely_new_workflow_is_still_added():
    existing = Skill(name="AppX", workflows=[Workflow("send_message", "Tap send.")])
    other = Skill(name="AppX", workflows=[
        Workflow("attach_media", "Tap the paperclip, pick Gallery, choose a photo.")])
    assert {w.name for w in existing.merge(other).workflows} == {"send_message",
                                                                "attach_media"}


def test_a_hand_written_markdown_skill_is_not_shadowed_by_a_generated_one(tmp_path):
    """`generate` always writes JSON. Before this, the first run that learned
    anything silently replaced guidance somebody had typed on purpose -- and
    which file won depended on the order the filesystem listed them."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "whatsapp.md").write_text(
        "# WhatsApp\n\n**Packages**: com.whatsapp\n\n"
        "## App Nuances & UI Quirks\n- Something I worked out by hand.\n",
        encoding="utf-8")

    SkillRegistry(skills_dir).save_skill(
        Skill(name="WhatsApp", packages=["com.whatsapp"],
              nuances=["Something a run worked out."]))

    reloaded = SkillRegistry(skills_dir).find_by_name_or_alias("whatsapp")
    assert set(reloaded.nuances) == {"Something I worked out by hand.",
                                     "Something a run worked out."}
    # Both files are still there; neither was overwritten on disk.
    assert (skills_dir / "whatsapp.md").is_file()
    assert (skills_dir / "whatsapp.json").is_file()


def test_skill_merge():
    sk1 = Skill(
        name="AppX",
        packages=["com.app.x"],
        workflows=[Workflow(name="wf1", steps="Step 1")],
        nuances=["Nuance 1"],
        recommendations=["Rec 1"]
    )
    sk2 = Skill(
        name="AppX",
        packages=["com.app.x", "com.app.x.beta"],
        workflows=[Workflow(name="wf2", steps="Step 2")],
        nuances=["Nuance 1", "Nuance 2"],
        recommendations=["Rec 2"]
    )
    merged = sk1.merge(sk2)
    assert len(merged.packages) == 2
    assert len(merged.workflows) == 2
    assert len(merged.nuances) == 2
    assert len(merged.recommendations) == 2


def test_skill_generator_with_screenshots(tmp_path):
    skills_dir = tmp_path / "skills"
    registry = SkillRegistry(skills_dir)
    generator = SkillGenerator(registry)

    llm_resp = json.dumps({
        "name": "VisionApp",
        "packages": ["com.vision.app"],
        "aliases": ["vapp"],
        "description": "Skill created with screenshot vision support",
        "workflows": [{"name": "vision_wf", "steps": "Step 1"}],
        "nuances": ["Visual nuance"],
        "recommendations": ["Visual rec"]
    })

    received_messages = []
    class MockVisionLLM:
        def __init__(self, response_text):
            self.response_text = response_text
            self.model = "mock-vision-model"

        def _post_chat(self, messages, model=None):
            received_messages.append(messages)
            return {
                "choices": [{"message": {"content": self.response_text}}]
            }

    mock_llm = MockVisionLLM(llm_resp)
    fake_screenshot = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    skill = generator.generate_from_exploration(
        app_name_or_pkg="VisionApp",
        tasks="do vision task",
        screen_summaries=["Screen 1"],
        actions_taken=["Action 1"],
        llm_client=mock_llm,
        screenshots=[fake_screenshot]
    )

    assert skill.name == "VisionApp"
    assert len(received_messages) == 1
    user_content = received_messages[0][1]["content"]
    assert isinstance(user_content, list)
    assert any(item.get("type") == "image_url" for item in user_content)




def test_cli_skills_list(capsys, tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    sk = Skill(name="TestCLIApp", packages=["com.cli.app"], description="CLI testing app")
    (skills_dir / "testcliapp.json").write_text(json.dumps(sk.to_dict()), encoding="utf-8")

    code = main(["skills", "list", "--skills-dir", str(skills_dir)])
    assert code == 0
    captured = capsys.readouterr()
    assert "TestCLIApp" in captured.out


def test_cli_skills_view(capsys, tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    sk = Skill(name="ViewApp", packages=["com.view.app"], description="App to view")
    (skills_dir / "viewapp.json").write_text(json.dumps(sk.to_dict()), encoding="utf-8")

    code = main(["skills", "view", "viewapp", "--skills-dir", str(skills_dir)])
    assert code == 0
    captured = capsys.readouterr()
    assert "# ViewApp" in captured.out


def test_cli_skills_create(tmp_path):
    skills_dir = tmp_path / "skills"
    code = main(["skills", "create", "NewApp", "--skills-dir", str(skills_dir)])
    assert code == 0
    assert (skills_dir / "newapp.json").exists()


# ---------------------------------------------------------------------------
# Live exploration
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = DEFAULT_EXPLORE_STEPS
    c.skills.skills_dir = str(tmp_path / "skills")
    c.safety.unattended = True
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


def note_then_done(label: str):
    """Tap `label`, writing a finding, then declare the tour finished."""

    def policy(screen, llm):
        for el in screen.elements:
            if el.best_text == label and el.interactive:
                return AgentAction(observation=f"the list shows {label}",
                                   reasoning="visit it", action="tap",
                                   target={"index": el.index},
                                   notes=[{"key": f"flow:open {label}",
                                           "value": f"tap {label} on the home list"}])
        return AgentAction(observation="seen enough", reasoning="tour finished",
                           action="done", text=f"visited {label}")

    return policy


def test_resolve_package_turns_a_name_into_the_installed_package():
    dev = fake.FakeDevice()
    assert resolve_package(dev, "whatsapp") == "com.whatsapp"
    assert resolve_package(dev, "com.whatsapp") == "com.whatsapp"
    assert resolve_package(dev, "settings") == "com.android.settings"


def test_resolve_package_reaches_a_name_the_dots_hide():
    """`BumbleApp` is not a substring of `com.bumble.app` -- the dot between the
    words defeats the search the `open_app` action uses."""
    dev = fake.FakeDevice()
    dev.installed = ["com.bumble.app", "com.spotify.music"]
    assert resolve_package(dev, "BumbleApp") == "com.bumble.app"
    assert resolve_package(dev, "bumble app") == "com.bumble.app"
    assert resolve_package(dev, "Bumble") == "com.bumble.app"


def test_resolve_package_says_nothing_rather_than_guessing():
    dev = fake.FakeDevice()
    assert resolve_package(dev, "com.not.installed") == ""
    assert resolve_package(dev, "") == ""
    # A two-letter fragment matches half the phone, so it is not a match at all.
    assert resolve_package(dev, "xy") == ""


def test_a_locked_phone_stops_the_run_instead_of_exploring_the_lock_screen(cfg, mem):
    """The old command opened the app behind the keyguard, saw
    com.android.systemui, and filed that as the app's skill."""
    dev = fake.FakeDevice(cfg, locked=True)
    with pytest.raises(ExplorationBlocked, match="lock screen"):
        explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg,
                    query="settings")


def test_an_app_that_is_not_installed_stops_the_run(cfg, mem):
    dev = fake.FakeDevice(cfg)
    with pytest.raises(ExplorationBlocked, match="no installed app matches"):
        explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg,
                    query="com.nope.missing")


def test_an_app_that_will_not_come_forward_stops_the_run(cfg, mem):
    """`app_start` reports nothing when it fails, so the foreground is checked.
    Here Spotify is installed but the scripted phone only ever draws Settings."""
    dev = fake.FakeDevice(cfg)
    with pytest.raises(ExplorationBlocked, match="would not come to the foreground"):
        explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg,
                    query="spotify")


def test_exploring_by_name_opens_the_resolved_package_and_drives_it(cfg, mem):
    dev = fake.FakeDevice(cfg, start="launcher")
    llm = fake.FakeLLM(dev, note_then_done("Wi-Fi"))
    exp = explore_app(dev, mem, llm, cfg, query="settings")

    assert exp.package == "com.android.settings"
    assert "open_app(com.android.settings)" in dev.actions
    assert exp.steps > 0 and llm.calls > 0
    assert exp.looked_around                      # more than one screen reached
    assert "wifi" in " ".join(exp.screens).lower()
    # The explorer's own findings, which are the primary source for synthesis.
    assert "flow:open wi-fi" in exp.notes.lower()


def test_exploring_with_no_app_named_uses_the_app_in_front(cfg, mem):
    dev = fake.FakeDevice(cfg)
    exp = explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg)

    assert exp.package == "com.android.settings"
    assert exp.chosen_by == "foreground"
    assert not any(a.startswith("open_app") for a in dev.actions)


# ---------------------------------------------------------------------------
# Working out which app the tasks are about
# ---------------------------------------------------------------------------

def test_the_tasks_name_the_app_so_the_argument_need_not():
    dev = fake.FakeDevice()
    assert package_from_text(dev, "open WhatsApp and read the last message") \
        == ("com.whatsapp", ["com.whatsapp"])
    assert package_from_text(dev, "tour spotify's playlists")[0] == "com.spotify.music"


def test_instruction_words_are_not_app_names():
    """"tap the chats tab" must not go looking for an app called Tab."""
    dev = fake.FakeDevice()
    assert package_from_text(dev, "tap each tab, open one item, scroll down") \
        == ("", [])
    assert package_from_text(dev, "") == ("", [])


def test_an_app_you_installed_beats_one_that_shipped_with_the_phone():
    """"tour the settings screen in Bumble" is about Bumble; Settings is what
    the sentence describes, not what it means."""
    dev = fake.FakeDevice()
    dev.installed = ["com.android.settings", "com.bumble.app"]
    dev.third_party = ["com.bumble.app"]
    pkg, candidates = package_from_text(dev, "tour the settings screen in Bumble")
    assert pkg == "com.bumble.app"
    assert candidates == ["com.android.settings", "com.bumble.app"]


def test_tasks_naming_two_apps_are_referred_back_rather_than_guessed(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.third_party = ["com.whatsapp", "com.spotify.music"]
    with pytest.raises(ExplorationBlocked, match="more than one installed app"):
        explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg,
                    tasks="share a whatsapp chat to spotify")


def test_the_app_the_tasks_name_is_opened_without_an_argument(cfg, mem):
    dev = fake.FakeDevice(cfg, start="launcher")
    exp = explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg,
                      tasks="open Settings and check the Wi-Fi screen")

    assert exp.package == "com.android.settings"
    assert exp.chosen_by == "tasks"
    assert "open_app(com.android.settings)" in dev.actions


def test_an_explicit_argument_still_wins_over_the_tasks(cfg, mem):
    dev = fake.FakeDevice(cfg, start="launcher")
    exp = explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg,
                      query="settings", tasks="read the whatsapp chats")
    assert exp.package == "com.android.settings"
    assert exp.chosen_by == "named"


def test_the_default_tasks_never_name_an_app(cfg, mem):
    """Inference reads what you wrote, not the boilerplate the harness adds when
    you write nothing."""
    dev = fake.FakeDevice(cfg)
    from adbagent.skills import DEFAULT_EXPLORE_TASKS
    assert package_from_text(dev, DEFAULT_EXPLORE_TASKS) == ("", [])


def test_exploring_with_nothing_but_the_launcher_in_front_stops(cfg, mem):
    dev = fake.FakeDevice(cfg, start="launcher")
    with pytest.raises(ExplorationBlocked, match="rather than an app"):
        explore_app(dev, mem, fake.FakeLLM(dev, note_then_done("Wi-Fi")), cfg)


def test_screenshots_are_spent_on_distinct_screens_not_the_first_n_steps(cfg, mem):
    """A tour crosses the same list repeatedly. Twelve pictures of the home
    screen teach the synthesis nothing that one does."""
    dev = fake.FakeDevice(cfg)
    seen = {"n": 0}

    def wander(screen, llm):
        seen["n"] += 1
        if seen["n"] > 8:
            return AgentAction(observation="done wandering", reasoning="enough",
                               action="done", text="toured")
        # Bounce between home and Wi-Fi: two screens, many steps.
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home list", reasoning="in",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="detail screen", reasoning="out",
                           action="press_key", key="back")

    exp = explore_app(dev, mem, fake.FakeLLM(dev, wander), cfg, query="settings")
    assert exp.steps >= 6
    assert len(exp.screens) <= 4          # far fewer records than steps
    assert len(exp.screenshots) == len(exp.screens)


def test_the_exploration_brief_says_what_to_record_and_what_not_to_touch():
    goal = exploration_goal("com.whatsapp", "read the last message in a chat")
    assert "com.whatsapp" in goal
    assert "read the last message in a chat" in goal
    for expected in ("notes", "quirk:", "press_key back", "done"):
        assert expected in goal
    assert "Do NOT send" in goal
    # On a dating or feed app the swipe *is* the irreversible action, and it
    # reaches a real person.
    assert "Do NOT swipe, like, pass, match" in goal
    # A skill is re-read every run; the account holder's data does not belong in it.
    assert "never what it currently holds" in goal


def test_the_synthesis_is_told_to_describe_controls_not_their_contents():
    from adbagent.skills import SKILL_SYNTHESIS_SYSTEM

    assert "never about the account using it" in SKILL_SYNTHESIS_SYSTEM
    assert "failed once and then worked" in SKILL_SYNTHESIS_SYSTEM


def test_a_skill_generated_by_name_is_filed_under_the_resolved_package(tmp_path):
    """`skills generate whatsapp` used to save a skill with no packages at all,
    so `find_by_package` never matched it and the agent never loaded it."""
    registry = SkillRegistry(tmp_path / "skills")
    named_only = dict(GENERATED, name="WhatsApp", packages=[])
    skill = SkillGenerator(registry).generate_from_exploration(
        "whatsapp", tasks="tour it", screen_summaries=["home"],
        actions_taken=["tap #1"], llm_client=MockLLM(json.dumps(named_only)),
        package="com.whatsapp", notes="flow:search - tap the magnifier")

    assert skill.matches_package("com.whatsapp")
    assert SkillRegistry(tmp_path / "skills").find_by_package("com.whatsapp") is not None


def test_the_explorers_own_findings_reach_the_synthesis_prompt(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    llm = MockLLM(json.dumps(GENERATED))
    sent = []
    llm._post_chat = lambda messages, model=None: (          # noqa: E731
        sent.append(messages) or {"choices": [{"message": {"content": llm.response_text}}]})

    SkillGenerator(registry).generate_from_exploration(
        "com.generated.app", tasks="tour it", screen_summaries=["home"],
        actions_taken=["tap #1"], llm_client=llm, package="com.generated.app",
        notes="quirk:back - back leaves the app from the search screen",
        outcome="success")

    prompt = sent[0][1]["content"]
    assert "back leaves the app from the search screen" in prompt
    assert "RESOLVED PACKAGE: com.generated.app" in prompt


def test_a_one_screen_exploration_reports_itself_as_one():
    assert not AppTrace(screens=["step 0: com.x"]).looked_around
    assert AppTrace(screens=["a", "b"]).looked_around


def test_synthesis_retries_without_the_screenshots_when_they_break_the_call(tmp_path):
    """A text-only `model_skill` fails the *whole* call on an image part. Losing
    the run's trace to that is worse than losing the pictures."""
    registry = SkillRegistry(tmp_path / "skills")

    class PicturesRefused(MockLLM):
        def __init__(self, response_text):
            super().__init__(response_text)
            self.attempts = []

        def _post_chat(self, messages, model=None):
            content = messages[1]["content"]
            self.attempts.append("images" if isinstance(content, list) else "text")
            if isinstance(content, list):
                raise RuntimeError("model does not support image input")
            return {"choices": [{"message": {"content": self.response_text}}]}

    llm = PicturesRefused(json.dumps(GENERATED))
    skill = SkillGenerator(registry).generate_from_exploration(
        "com.generated.app", tasks="t", screen_summaries=["home"],
        actions_taken=["tap #1"], llm_client=llm,
        screenshots=[b"\x89PNG\r\n\x1a\n"], package="com.generated.app")

    assert llm.attempts == ["images", "text"]
    assert skill.name == "GeneratedApp"      # the trace survived, not a template


# ---------------------------------------------------------------------------
# Learning from an ordinary run
# ---------------------------------------------------------------------------

def a_learnable_trace(**kw):
    fields = dict(package="com.generated.app", tasks="do the thing",
                  screens=["home", "detail"], actions=["step 1: tap #1"],
                  notes="quirk:back - back exits from search",
                  outcome="success", steps=MIN_LEARNABLE_STEPS)
    return AppTrace(**{**fields, **kw})


def test_a_finished_run_updates_the_apps_skill(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    skill = learn_from_run(a_learnable_trace(), MockLLM(json.dumps(GENERATED)),
                           registry, goal="open a chat and read it")

    assert skill is not None
    assert skill.matches_package("com.generated.app")
    assert (tmp_path / "skills" / "generatedapp.json").is_file()


def test_learning_merges_rather_than_replacing_what_was_known(tmp_path):
    """The point of learning after every run: run 20 in an app knows what runs
    1 to 19 found, not only what run 20 happened to touch."""
    registry = SkillRegistry(tmp_path / "skills")
    learn_from_run(a_learnable_trace(), MockLLM(json.dumps(GENERATED)), registry)

    later = dict(GENERATED, nuances=["Nuance from the second run"],
                 workflows=[{"name": "task2", "steps": "Step 2"}])
    merged = learn_from_run(a_learnable_trace(), MockLLM(json.dumps(later)),
                            SkillRegistry(tmp_path / "skills"))

    assert set(merged.nuances) == {"Nuance 1", "Nuance from the second run"}
    assert {w.name for w in merged.workflows} == {"task1", "task2"}


def test_a_run_that_went_nowhere_teaches_nothing(tmp_path):
    """Paying a synthesis call to hear back what the skill already said only
    dilutes it."""
    registry = SkillRegistry(tmp_path / "skills")
    assert learn_from_run(AppTrace(package="com.generated.app", screens=["home"],
                                   steps=1, outcome="failed"),
                          MockLLM(json.dumps(GENERATED)), registry) is None
    assert not list((tmp_path / "skills").glob("*.json"))


def test_a_run_that_never_left_the_launcher_teaches_nothing(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    assert learn_from_run(a_learnable_trace(package="com.android.launcher3"),
                          MockLLM(json.dumps(GENERATED)), registry) is None


def test_a_failed_run_still_teaches(tmp_path):
    """"Tapping this row does nothing" is only ever learned by a run that went
    wrong, and it is exactly what a skill is for."""
    registry = SkillRegistry(tmp_path / "skills")
    llm = MockLLM(json.dumps(GENERATED))
    sent = []
    llm._post_chat = lambda messages, model=None: (          # noqa: E731
        sent.append(messages) or {"choices": [{"message": {"content": llm.response_text}}]})

    skill = learn_from_run(a_learnable_trace(outcome="failed"), llm, registry)
    assert skill is not None
    assert "HOW THE EXPLORATION ENDED: failed" in sent[0][1]["content"]


def test_the_trace_attributes_a_multi_app_run_to_the_app_it_worked_in(cfg, mem):
    """A goal that crosses apps should update the skill for the one the steps
    were spent in, not whichever was in front when the run ended."""
    dev = fake.FakeDevice(cfg)
    collector = TraceCollector(dev)
    for step, pkg in enumerate(["com.whatsapp", "com.whatsapp", "com.whatsapp",
                                "com.google.android.apps.docs"], start=1):
        screen = dev.observe()
        screen.package = pkg
        screen.skeleton_id = f"{pkg}-{step}"
        collector.record(screen, step)

    assert collector.main_package == "com.whatsapp"


def test_the_trace_ignores_the_launcher_when_picking_the_app(cfg):
    dev = fake.FakeDevice(cfg)
    collector = TraceCollector(dev)
    for step, pkg in enumerate(["com.android.launcher3", "com.android.launcher3",
                                "com.whatsapp"], start=1):
        screen = dev.observe()
        screen.package = pkg
        screen.skeleton_id = f"{pkg}-{step}"
        collector.record(screen, step)

    assert collector.main_package == "com.whatsapp"


def a_two_app_tour(cfg, blinkit_steps=3):
    """Drive a collector through a zepto-then-blinkit price check."""
    from types import SimpleNamespace
    dev = fake.FakeDevice(cfg)
    collector = TraceCollector(dev)
    stops = ["com.zepto"] * 4 + ["com.blinkit"] * blinkit_steps
    for step, pkg in enumerate(stops, start=1):
        screen = dev.observe()
        screen.package = pkg
        screen.skeleton_id = f"{pkg}-{step}"
        collector("step", step=step, screen=screen, action=f"tap #{step}")
    collector.finish("success", SimpleNamespace(step=len(stops), llm_calls=len(stops),
                                                run_id="r1", scratchpad=None))
    return collector


def test_a_run_across_two_apps_leaves_a_trace_for_each(cfg):
    """The zepto-and-blinkit case: a price check across two apps is a tour of
    both, and learning from only the busier one leaves the other with no skill
    from a run that walked its screens just the same."""
    zepto, blinkit = a_two_app_tour(cfg).app_traces()

    assert (zepto.package, blinkit.package) == ("com.zepto", "com.blinkit")
    assert zepto.steps == 4 and blinkit.steps == 3
    assert all("com.zepto" in s for s in zepto.screens)
    assert all("com.blinkit" in s for s in blinkit.screens)
    assert len(zepto.actions) == 4 and len(blinkit.actions) == 3


class PerAppLLM(MockLLM):
    """Answers each synthesis with a skill named for the app being asked about."""

    def _post_chat(self, messages, model=None):
        content = messages[1]["content"]
        if isinstance(content, list):  # the with-screenshots attempt: parts, not text
            content = " ".join(part.get("text", "") for part in content
                               if isinstance(part, dict))
        pkg = "com.zepto" if "com.zepto" in content else "com.blinkit"
        name = "Zepto" if pkg == "com.zepto" else "Blinkit"
        skill = dict(GENERATED, name=name, packages=[pkg])
        return {"choices": [{"message": {"content": json.dumps(skill)}}]}


def test_a_run_across_two_apps_updates_both_skills(tmp_path, cfg):
    registry = SkillRegistry(tmp_path / "skills")
    for trace in a_two_app_tour(cfg).app_traces():
        learn_from_run(trace, PerAppLLM(""), registry)

    saved = SkillRegistry(tmp_path / "skills")
    assert len(saved.list_skills()) == 2
    assert saved.find_by_package("com.zepto") is not None
    assert saved.find_by_package("com.blinkit") is not None


def test_a_run_that_only_peeked_at_the_second_app_leaves_it_unlearned(tmp_path, cfg):
    """Opening an app for one step on the way past is not a tour of it, and a
    skill written from that is invention."""
    registry = SkillRegistry(tmp_path / "skills")
    learned = [learn_from_run(trace, PerAppLLM(""), registry)
               for trace in a_two_app_tour(cfg, blinkit_steps=1).app_traces()]

    assert learned[0] is not None and learned[1] is None
    saved = SkillRegistry(tmp_path / "skills")
    assert saved.find_by_package("com.zepto") is not None
    assert saved.find_by_package("com.blinkit") is None


def test_the_collector_passes_events_through_to_the_reporter(cfg, mem):
    """It wraps whatever reporter the caller already had; a run that started
    printing progress must not stop."""
    dev = fake.FakeDevice(cfg)
    seen = []
    collector = TraceCollector(dev, on_event=lambda kind, **kw: seen.append(kind))
    llm = fake.FakeLLM(dev, note_then_done("Wi-Fi"))

    from adbagent.agent import Agent
    outcome, state = Agent(dev, mem, llm, cfg, on_event=collector).run("open Wi-Fi")

    assert "step" in seen and "perceive" in seen
    trace = collector.finish(outcome, state)
    assert trace.package == "com.android.settings"
    assert trace.outcome == outcome
    assert trace.steps == state.step
    assert "flow:open wi-fi" in trace.notes.lower()
