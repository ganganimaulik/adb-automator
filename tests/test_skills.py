import json
from pathlib import Path
import pytest

from adbagent.config import Config
from adbagent.skills import Skill, Workflow, SkillRegistry, SkillGenerator
from adbagent.cli import main


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


def test_skill_generator(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    generator = SkillGenerator(registry)

    llm_resp = json.dumps({
        "name": "GeneratedApp",
        "packages": ["com.generated.app"],
        "aliases": ["genapp"],
        "description": "Auto generated skill description",
        "workflows": [{"name": "task1", "steps": "Step 1 details"}],
        "nuances": ["Nuance 1"],
        "recommendations": ["Recommendation 1"]
    })
    mock_llm = MockLLM(llm_resp)

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
