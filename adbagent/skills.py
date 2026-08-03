"""Per-app skills: guidance, workflows, UI nuances, and automated skill generation.

App Skills guide the agent on how to use specific Android apps, detailing step-by-step
common workflows, UI quirks/nuances, and recommended action strategies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("adbagent.skills")

DEFAULT_SKILLS_DIR = "skills"


@dataclass
class Workflow:
    name: str
    steps: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        if isinstance(data, str):
            return cls(name="general", steps=data)
        return cls(
            name=str(data.get("name", "general")),
            steps=str(data.get("steps", ""))
        )

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "steps": self.steps}


@dataclass
class Skill:
    name: str
    packages: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    workflows: List[Workflow] = field(default_factory=list)
    nuances: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    custom_prompt: str = ""

    def matches_package(self, package: str) -> bool:
        if not package:
            return False
        pkg_lower = package.lower().strip()
        for p in self.packages:
            if p.lower().strip() == pkg_lower:
                return True
        return False

    def matches_query(self, query: str) -> bool:
        if not query:
            return False
        q = query.lower().strip()
        if q == self.name.lower():
            return True
        if any(q == alias.lower() for alias in self.aliases):
            return True
        if any(pkg.lower() in q for pkg in self.packages):
            return True
        return False

    def matches_goal(self, goal: str) -> bool:
        if not goal:
            return False
        g_words = re.findall(r"\w+", goal.lower())
        if self.name.lower() in g_words:
            return True
        for alias in self.aliases:
            if alias.lower() in g_words:
                return True
        for pkg in self.packages:
            if pkg.lower() in goal.lower():
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "packages": self.packages,
            "aliases": self.aliases,
            "description": self.description,
            "workflows": [w.to_dict() for w in self.workflows],
            "nuances": self.nuances,
            "recommendations": self.recommendations,
            "custom_prompt": self.custom_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        raw_workflows = data.get("workflows") or []
        wf_objs = [Workflow.from_dict(w) for w in raw_workflows]
        return cls(
            name=str(data.get("name", "Untitled App")),
            packages=list(data.get("packages") or []),
            aliases=list(data.get("aliases") or []),
            description=str(data.get("description", "")),
            workflows=wf_objs,
            nuances=list(data.get("nuances") or []),
            recommendations=list(data.get("recommendations") or []),
            custom_prompt=str(data.get("custom_prompt", "")),
        )

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            "",
            f"**Packages**: {', '.join(self.packages) if self.packages else 'None'}",
            f"**Aliases**: {', '.join(self.aliases) if self.aliases else 'None'}",
            "",
            f"## Description",
            self.description or "No description provided.",
            "",
        ]

        if self.workflows:
            lines.append("## Common Workflows")
            for wf in self.workflows:
                lines.append(f"### {wf.name}")
                lines.append(wf.steps)
                lines.append("")

        if self.nuances:
            lines.append("## App Nuances & UI Quirks")
            for n in self.nuances:
                lines.append(f"- {n}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations & Best Practices")
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")

        if self.custom_prompt:
            lines.append("## Custom Guidance")
            lines.append(self.custom_prompt)
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, md_text: str, name_hint: str = "") -> "Skill":
        lines = md_text.splitlines()
        name = name_hint or "Custom App"
        packages: List[str] = []
        aliases: List[str] = []
        description = ""
        nuances: List[str] = []
        recommendations: List[str] = []

        current_section = ""
        desc_lines: List[str] = []

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("# "):
                name = line_str[2:].strip()
                continue
            if line_str.lower().startswith("**packages**:") or line_str.lower().startswith("packages:"):
                raw_pkgs = line_str.split(":", 1)[1].strip()
                packages = [p.strip() for p in raw_pkgs.split(",") if p.strip()]
                continue
            if line_str.lower().startswith("**aliases**:") or line_str.lower().startswith("aliases:"):
                raw_als = line_str.split(":", 1)[1].strip()
                aliases = [a.strip() for a in raw_als.split(",") if a.strip()]
                continue
            if line_str.startswith("## "):
                current_section = line_str[3:].lower().strip()
                continue
            if current_section.startswith("description"):
                if line_str:
                    desc_lines.append(line_str)
            elif "nuance" in current_section or "quirk" in current_section:
                if line_str.startswith("- ") or line_str.startswith("* "):
                    nuances.append(line_str[2:].strip())
            elif "recommend" in current_section or "best practice" in current_section:
                if line_str.startswith("- ") or line_str.startswith("* "):
                    recommendations.append(line_str[2:].strip())

        description = "\n".join(desc_lines).strip()
        return cls(
            name=name,
            packages=packages,
            aliases=aliases,
            description=description,
            nuances=nuances,
            recommendations=recommendations,
            custom_prompt=md_text if not (packages or nuances) else "",
        )

    def merge(self, other: "Skill") -> "Skill":
        """Merge another skill into this one, adding new workflows, nuances, and recommendations without duplicates."""
        merged_packages = list(dict.fromkeys([p for p in (self.packages + other.packages) if p]))
        merged_aliases = list(dict.fromkeys([a for a in (self.aliases + other.aliases) if a]))

        wf_map = {wf.name: wf for wf in self.workflows}
        for wf in other.workflows:
            if wf.name in wf_map:
                if wf.steps and wf.steps != wf_map[wf.name].steps:
                    wf_map[wf.name] = Workflow(name=wf.name, steps=wf.steps)
            else:
                wf_map[wf.name] = wf

        merged_workflows = list(wf_map.values())
        merged_nuances = list(dict.fromkeys([n.strip() for n in (self.nuances + other.nuances) if n.strip()]))
        merged_recommendations = list(dict.fromkeys([r.strip() for r in (self.recommendations + other.recommendations) if r.strip()]))

        merged_description = self.description
        if not merged_description or (other.description and len(other.description) > len(merged_description)):
            merged_description = other.description

        merged_custom = self.custom_prompt
        if other.custom_prompt and other.custom_prompt not in merged_custom:
            merged_custom = (merged_custom + "\n\n" + other.custom_prompt).strip()

        return Skill(
            name=self.name or other.name,
            packages=merged_packages,
            aliases=merged_aliases,
            description=merged_description,
            workflows=merged_workflows,
            nuances=merged_nuances,
            recommendations=merged_recommendations,
            custom_prompt=merged_custom,
        )

    def to_prompt_text(self) -> str:
        """Format skill guidance for inclusion in LLM prompts."""
        parts = [f"APP SKILL & GUIDANCE ({self.name}):"]
        if self.description:
            parts.append(f"Overview: {self.description}")

        if self.workflows:
            parts.append("Workflows:")
            for wf in self.workflows:
                parts.append(f"  - {wf.name}: {wf.steps}")

        if self.nuances:
            parts.append("Nuances & UI Quirks:")
            for n in self.nuances:
                parts.append(f"  - {n}")

        if self.recommendations:
            parts.append("Recommendations:")
            for r in self.recommendations:
                parts.append(f"  - {r}")

        if self.custom_prompt:
            parts.append(f"Additional Guidance:\n{self.custom_prompt}")

        return "\n".join(parts)


class SkillRegistry:
    """Manages loading, matching, and storing App Skills from ./skills or custom directory."""

    def __init__(self, skills_dir: Optional[Path | str] = None):
        self.skills_dir = Path(skills_dir or DEFAULT_SKILLS_DIR).expanduser()
        self.skills: Dict[str, Skill] = {}
        self.load_skills()

    def load_skills(self) -> None:
        self.skills.clear()
        if not self.skills_dir.exists():
            try:
                self.skills_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not create skills directory %s: %s", self.skills_dir, exc)
                return

        for path in self.skills_dir.glob("*"):
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    skill = Skill.from_dict(data)
                    self.skills[skill.name.lower()] = skill
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse skill JSON %s: %s", path, exc)
            elif path.suffix in (".md", ".markdown"):
                try:
                    text = path.read_text(encoding="utf-8")
                    skill = Skill.from_markdown(text, name_hint=path.stem.capitalize())
                    self.skills[skill.name.lower()] = skill
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse skill MD %s: %s", path, exc)

    def list_skills(self) -> List[Skill]:
        return sorted(self.skills.values(), key=lambda s: s.name)

    def find_by_package(self, package: str) -> Optional[Skill]:
        if not package:
            return None
        for skill in self.skills.values():
            if skill.matches_package(package):
                return skill
        return None

    def find_by_name_or_alias(self, query: str) -> Optional[Skill]:
        if not query:
            return None
        for skill in self.skills.values():
            if skill.matches_query(query):
                return skill
        return None

    def find_for_run(self, package: str, goal: str) -> Optional[Skill]:
        # 1. Package match (highest priority when inside app)
        if package:
            sk = self.find_by_package(package)
            if sk:
                return sk
        # 2. Goal text match (e.g. goal says "Open Spotify and search...")
        if goal:
            for skill in self.skills.values():
                if skill.matches_goal(goal):
                    return skill
        return None

    def save_skill(self, skill: Skill) -> Path:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{re.sub(r'[^a-z0-9_]', '_', skill.name.lower())}.json"
        target_path = self.skills_dir / filename
        target_path.write_text(json.dumps(skill.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self.skills[skill.name.lower()] = skill
        return target_path


SKILL_SYNTHESIS_SYSTEM = """\
You are an expert mobile automation strategist and app skills architect.
Your job is to analyze the exploration history of an Android application and generate or update a comprehensive App Skill specification.

The skill specification must be a single valid JSON object with the following schema:
{
  "name": "App Display Name",
  "packages": ["package.name.1", "package.name.2"],
  "aliases": ["alias1", "alias2"],
  "description": "Clear overview of what this app is for and primary workflows.",
  "workflows": [
    {
      "name": "workflow_name",
      "steps": "Step 1, Step 2, Step 3 instructions."
    }
  ],
  "nuances": [
    "Key UI quirk or pitfall 1",
    "Key UI quirk or pitfall 2"
  ],
  "recommendations": [
    "Recommended action strategy 1",
    "Recommended parameter or scrolling recommendation 2"
  ]
}

If an existing skill definition is provided, incorporate any new workflows, nuances, or recommendations discovered during exploration into the output while preserving existing valid items.
Be precise, actionable, and focus on practical automation hints that help an AI driver avoid getting stuck.
Reply with ONLY the JSON object.
"""


class SkillGenerator:
    """Generates or updates an App Skill by executing user task instructions on a device and synthesizing findings using LLM."""

    def __init__(self, registry: Optional[SkillRegistry] = None):
        self.registry = registry or SkillRegistry()

    def generate_from_exploration(self, app_name_or_pkg: str, tasks: str,
                                 screen_summaries: List[str],
                                 actions_taken: List[str],
                                 llm_client: Any,
                                 screenshots: Optional[List[bytes]] = None) -> Skill:
        """Synthesize exploration observations into a Skill object, updating existing skill if present."""
        existing_skill = (self.registry.find_by_name_or_alias(app_name_or_pkg) or
                          self.registry.find_by_package(app_name_or_pkg))

        prompt_parts = [f"APP NAME OR PACKAGE: {app_name_or_pkg}"]
        if existing_skill:
            prompt_parts.append(f"EXISTING SKILL DEFINITION:\n{json.dumps(existing_skill.to_dict(), indent=2)}")
        prompt_parts.append(f"USER TASKS PERFORMED: {tasks}")
        prompt_parts.append(f"SCREEN STATES OBSERVED:\n" + "\n".join(screen_summaries or ["(none)"]))
        prompt_parts.append(f"ACTIONS TAKEN DURING EXPLORATION:\n" + "\n".join(actions_taken or ["(none)"]))
        prompt_parts.append(f"Generate an updated, complete App Skill JSON object for {app_name_or_pkg} combining existing skill information and new exploration findings.")

        prompt = "\n\n".join(prompt_parts)

        if hasattr(llm_client, "cfg"):
            model_to_use = llm_client.cfg.llm.skill() or llm_client.cfg.llm.image() or llm_client.model
        else:
            model_to_use = getattr(llm_client, "model", "")

        user_content: Any = prompt
        if screenshots:
            from .llm import image_part, text_part
            content_list: List[Dict[str, Any]] = [text_part(prompt)]
            for shot in screenshots:
                if shot:
                    content_list.append(image_part(shot))
            user_content = content_list

        messages = [
            {"role": "system", "content": SKILL_SYNTHESIS_SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            if hasattr(llm_client, "_post"):
                raw_text, _ = llm_client._post(
                    messages, model=model_to_use, schema=None,
                    max_tokens=getattr(getattr(llm_client, "cfg", None), "llm", None).max_tokens if hasattr(llm_client, "cfg") else 1500,
                    purpose="skill_generation"
                )
                content = raw_text.strip()
            elif hasattr(llm_client, "_post_chat"):
                raw_resp = llm_client._post_chat(messages, model=model_to_use)
                content = raw_resp["choices"][0]["message"]["content"].strip()
            else:
                raise RuntimeError("Unsupported LLM client interface")

            # Clean JSON codeblock wrappers if present
            if content.startswith("```"):
                content = re.sub(r"^```[a-z]*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            data = json.loads(content.strip())
            generated_skill = Skill.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skill synthesis model call failed or returned invalid JSON: %s. Using fallback template.", exc)
            generated_skill = Skill(
                name=existing_skill.name if existing_skill else app_name_or_pkg.replace(".", "_").capitalize(),
                packages=[app_name_or_pkg] if ("." in app_name_or_pkg and not existing_skill) else [],
                aliases=[app_name_or_pkg.lower()],
                description=f"App skill for {app_name_or_pkg} based on user tasks: {tasks}",
                workflows=[Workflow(name="user_tasks", steps=tasks)],
                nuances=["Observe screen element indices carefully before tapping."],
                recommendations=["Use input_text with clear=true when editing fields."]
            )

        if existing_skill:
            final_skill = existing_skill.merge(generated_skill)
        else:
            final_skill = generated_skill

        self.registry.save_skill(final_skill)
        return final_skill
