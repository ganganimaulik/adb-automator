"""Per-app skills: guidance, workflows, UI nuances, and automated skill generation.

App Skills guide the agent on how to use specific Android apps, detailing step-by-step
common workflows, UI quirks/nuances, and recommended action strategies.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - `agent` imports this module at run time
    from .screen import Screen

log = logging.getLogger("adbagent.skills")

DEFAULT_SKILLS_DIR = "skills"


#: How much smaller a restatement may be than the entry that swallows it. At
#: 0.34 the richer entry can say roughly three times as much before containment
#: stops being restatement -- below that, a short generic line is being eaten by
#: a long unrelated one that happens to share two words.
_RESTATEMENT_FLOOR = 0.34

#: Distinctive words an entry needs before containment means anything. "Something
#: a run worked out" reduces to {something, worked}, which sits inside "Something I
#: worked out by hand" -- two unrelated lines, and the ratio floor cannot tell,
#: because 2-of-3 is a high ratio. Below this an entry is only ever dropped for
#: being an exact duplicate.
_MIN_TOKENS_TO_JUDGE = 4

#: Overlap at which two entries are the same finding worded twice, even though
#: neither contains the other. "Search top bar icon must be tapped before typing"
#: and "Search top bar must be opened before typing" measured 0.75: each holds a
#: word the other lacks, so containment alone kept both. Measured against real
#: skills, genuine pairs that each carry a distinct detail sit around 0.4.
_NEAR_IDENTICAL = 0.7


def collapse_restatements(entries: List[str]) -> List[str]:
    """Drop entries that say strictly less than another entry in the list.

    Two runs describing the same quirk word it differently, so exact-match
    dedup keeps both, and a skill regenerated twenty times carries twenty
    phrasings of one nuance into every prompt from then on. A skill that grows
    like that gets worse with use, which is the opposite of the point.

    An entry whose distinctive tokens all appear in a longer entry adds nothing
    that entry does not already say, so it goes. Containment first, because it
    loses no information by construction -- two entries that merely overlap, each
    carrying a detail the other lacks, are both kept.

    Containment alone is not quite enough, though. Two wordings of one finding can
    each hold a word the other lacks ("tapped" against "opened") while saying the
    same thing, so entries that overlap almost entirely are collapsed too, keeping
    the longer. That threshold is the one judgement call here; genuine pairs
    measured well below it.
    """
    from .scratchpad import distinctive

    tokens = [distinctive(e) for e in entries]

    def swallows(rich: int, weak: int) -> bool:
        """Does `rich` already say everything `weak` does?"""
        a, b = tokens[weak], tokens[rich]
        if len(a) < _MIN_TOKENS_TO_JUDGE or len(b) < _MIN_TOKENS_TO_JUDGE:
            return False
        overlap = len(a & b) / len(a | b)
        contained = a <= b and len(a) >= _RESTATEMENT_FLOOR * len(b)
        if not contained and overlap < _NEAR_IDENTICAL:
            return False
        # Only one of a pair may go, or both would. The longer text wins; on a
        # true tie the earlier one does.
        return (len(b), -rich) > (len(a), -weak)

    kept: List[str] = []
    for i, entry in enumerate(entries):
        if any(swallows(j, i) for j in range(len(entries)) if j != i):
            log.info("skill: dropped a restatement -- %.70s", entry)
            continue
        if entry not in kept:
            kept.append(entry)
    return kept


def _richer(kept: str, fresh: str) -> str:
    """Of two versions of the same thing, the one that says more.

    A superset of the other's distinctive words wins outright; failing that the
    longer text does, which is the rule `description` has always used.
    """
    from .scratchpad import distinctive

    a, b = distinctive(kept), distinctive(fresh)
    if a and b and a != b:
        if a < b:
            return fresh
        if b < a:
            return kept
    return fresh if len(fresh) > len(kept) else kept


def _dedupe_workflows(workflows: List["Workflow"]) -> List["Workflow"]:
    """Drop workflows whose steps merely restate another's.

    A run that renames ``send_message`` to ``send_a_message`` produces a second
    entry saying the same thing, and a map keyed by name cannot see it. Keyed on
    the steps here, keeping the first (older) name so a skill does not churn its
    workflow names every time it is regenerated.
    """
    keep = set(collapse_restatements([wf.steps for wf in workflows]))
    out: List["Workflow"] = []
    seen: set = set()
    for wf in workflows:
        if wf.steps and wf.steps not in keep:
            log.info("skill: dropped workflow %r, a restatement of another", wf.name)
            continue
        if wf.steps and wf.steps in seen:
            continue
        seen.add(wf.steps)
        out.append(wf)
    return out


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
            "## Description",
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
            kept = wf_map.get(wf.name)
            if kept is None:
                wf_map[wf.name] = wf
            elif wf.steps and wf.steps != kept.steps:
                # Whichever says more wins. This used to be "the newest wins",
                # which let one thin run replace a detailed procedure with "Tap
                # send." -- a silent regression in the file the next run obeys.
                # Additive like the nuances, for the same reason: a correction
                # that shortens a workflow has to be made by hand.
                wf_map[wf.name] = Workflow(
                    name=wf.name, steps=_richer(kept.steps, wf.steps))

        # Renames arrive as a second workflow saying the same thing under a new
        # name, which no name-keyed map can catch.
        merged_workflows = _dedupe_workflows(list(wf_map.values()))
        merged_nuances = collapse_restatements(
            [n.strip() for n in (self.nuances + other.nuances) if n.strip()])
        merged_recommendations = collapse_restatements(
            [r.strip() for r in (self.recommendations + other.recommendations) if r.strip()])

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

        # Sorted, because two files can name the same skill and whose content
        # survived used to depend on the order the filesystem happened to hand
        # them back.
        for path in sorted(self.skills_dir.glob("*")):
            skill: Optional[Skill] = None
            if path.suffix == ".json":
                try:
                    skill = Skill.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse skill JSON %s: %s", path, exc)
            elif path.suffix in (".md", ".markdown"):
                try:
                    skill = Skill.from_markdown(path.read_text(encoding="utf-8"),
                                                name_hint=path.stem.capitalize())
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse skill MD %s: %s", path, exc)
            if skill is None:
                continue

            key = skill.name.lower()
            if key in self.skills:
                # `generate` always writes JSON, so a hand-written `whatsapp.md`
                # would otherwise be silently shadowed the first time a run
                # learned anything -- losing guidance somebody typed on purpose.
                # Merging keeps both; the warning is so the two files can be
                # consolidated rather than quietly diverging forever.
                log.warning("two files define the skill %r; merging %s into what "
                            "was already loaded", skill.name, path.name)
                skill = self.skills[key].merge(skill)
            self.skills[key] = skill

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

    def path_for(self, skill: Skill) -> Path:
        """Where this skill lives on disk, saved or not."""
        return self.skills_dir / f"{re.sub(r'[^a-z0-9_]', '_', skill.name.lower())}.json"

    def save_skill(self, skill: Skill) -> Path:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.path_for(skill)
        target_path.write_text(json.dumps(skill.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self.skills[skill.name.lower()] = skill
        return target_path


# ---------------------------------------------------------------------------
# Live exploration
# ---------------------------------------------------------------------------

#: Step budget for an exploration when the caller names none. A tour of an app's
#: main screens is tens of steps; `run.max_steps` is sized for a collection run
#: -- 550 in the config this was written against -- and letting an open-ended
#: "look around" inherit that is how a whole budget goes on one app's launcher.
DEFAULT_EXPLORE_STEPS = 40

#: What to do in the app when the caller names no tasks. Breadth, because the
#: skill a later run needs is "where does this app keep things", and a run given
#: no instructions at all answers `done` on the first frame.
DEFAULT_EXPLORE_TASKS = ("tour the main screens: every tab or nav destination, "
                         "the search entry point, and one item opened from a list")

#: How many screenshots the synthesis prompt carries. Each one is paid for, so
#: they are spent on distinct screens rather than on the first N steps.
MAX_EXPLORE_SHOTS = 12

#: The phone itself, not an app worth a skill. Used only to reject an empty
#: target -- an explicitly named launcher is the caller's business.
_NOT_AN_APP = frozenset({
    "com.android.systemui", "android", "com.android.settings.intelligence",
})


class ExplorationBlocked(RuntimeError):
    """The phone cannot be explored, and retrying will not change that."""


def is_app_package(package: str) -> bool:
    return bool(package) and package not in _NOT_AN_APP and "launcher" not in package.lower()


#: Splits a typed app name into words, camelCase included, so "BumbleApp" can
#: reach `com.bumble.app`. Two-letter fragments match half the phone, so they
#: are dropped.
_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

#: Words that sit in half of all package names. Matching on one identifies
#: nothing, and picking the "best" of everything installed is a wrong answer
#: delivered confidently.
_TOO_COMMON = frozenset({
    "com", "org", "net", "app", "apps", "android", "google", "mobile", "free",
    "client", "the",
})

#: Words that carry no app identity when they turn up in an instruction. They
#: are what a task is *made of*, so leaving them in means "tap the chats tab"
#: goes looking for an app called Tab.
_INSTRUCTION_WORDS = frozenset({
    "open", "tour", "explore", "check", "read", "look", "list", "lists", "view",
    "visit", "browse", "find", "search", "scroll", "swipe", "tap", "press",
    "screen", "screens", "tab", "tabs", "page", "pages", "menu", "main", "each",
    "every", "then", "from", "into", "only", "these", "this", "that", "your",
    "own", "and", "for", "with", "back", "down", "one", "item", "items", "note",
    "notes", "record", "detail", "details", "bottom", "top", "nav", "navigation",
    "bar", "button", "buttons", "not", "dont", "avoid", "skip", "entirely",
    "reached", "opened", "destination", "destinations", "anything", "without",
})


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def resolve_package(dev: Any, query: str) -> str:
    """The installed package `query` names, or "" when nothing matches.

    The first two rules are the ones the ``open_app`` action uses -- an exact
    package, else a substring of one, picked between by `_best_app_match` -- so
    that ``skills generate whatsapp`` and a model deciding to
    ``open_app("whatsapp")`` land on the same package. They have to: a skill
    filed under a name the agent never reports as its foreground package is a
    skill that never loads.

    Two more rules follow, for the names people actually type. Both only ever
    run when the shared rules found nothing, and neither runs for a query with a
    dot in it -- that is someone naming a package, and a package they got wrong
    should be reported wrong rather than quietly swapped for a near neighbour:

    * punctuation dropped from both sides, so ``BumbleApp`` finds
      ``com.bumble.app`` -- the dot between the words is what defeats a plain
      substring search;
    * the query's longest distinctive word, so ``Bumble App`` finds it too.
    """
    from .actions import _best_app_match

    q = (query or "").strip()
    if not q:
        return ""
    installed = dev.list_apps()

    for pkg in installed:
        if pkg.lower() == q.lower():
            return pkg

    matches = [p for p in installed if q.lower() in p.lower()]

    if not matches and "." not in q:
        if _squash(q):
            matches = [p for p in installed if _squash(q) in _squash(p)]
        words = sorted((w.lower() for w in _NAME_WORD.findall(q)
                        if len(w) > 2 and w.lower() not in _TOO_COMMON),
                       key=len, reverse=True)
        for word in words:
            if matches:
                break
            # A dotted segment, or the start of one -- not any substring. "store"
            # sits inside `com.google.android.apps.restore`, and answering that
            # for "PlayStore" is worse than answering nothing, because nothing
            # says so and a wrong package quietly explores the wrong app.
            matches = [p for p in installed
                       if any(seg.startswith(word) for seg in p.lower().split("."))]

    return _best_app_match(matches, q) if matches else ""


def package_from_text(dev: Any, text: str) -> Tuple[str, List[str]]:
    """The installed app a free-text instruction names.

    Saves naming the app twice: ``--tasks "open Bumble and read the matches"``
    already says which app, and repeating it as an argument is a second chance
    to get it wrong.

    Returns ``(package, candidates)``. An empty package alongside several
    candidates means the text named more than one installed app and only the
    caller can settle it; empty for both means it named none.

    Matching is on whole package segments, not substrings -- free text has far
    too many words to let ``store`` reach ``...restore``.
    """
    # Whole words *and* their camelCase parts. "WhatsApp" splits to "whats" and
    # "app", neither of which is a segment of `com.whatsapp`, so splitting alone
    # would find nothing; squashing alone would miss "spotify's".
    raw = {_squash(token) for token in re.split(r"\s+", text or "")}
    raw |= {w.lower() for w in _NAME_WORD.findall(text or "")}
    words = {w for w in raw
             if len(w) > 2 and w not in _TOO_COMMON and w not in _INSTRUCTION_WORDS}
    if not words:
        return "", []

    hits = {p for p in dev.list_apps()
            if any(seg in words for seg in p.lower().split("."))}
    if len(hits) <= 1:
        return (next(iter(hits)) if hits else ""), sorted(hits)

    # Several apps named. One installed by the user beats one that shipped with
    # the phone: "tour the settings screen in Bumble" is about Bumble, and the
    # phone's own Settings is what the sentence describes, not what it means.
    third_party = hits & set(dev.list_apps(third_party_only=True))
    if len(third_party) == 1:
        return next(iter(third_party)), sorted(hits)
    return "", sorted(hits)


def ready_for_exploration(dev: Any) -> str:
    """"" when the phone can be driven, otherwise why it cannot."""
    try:
        dev.wake()
    except Exception as exc:  # noqa: BLE001 - a wake that fails is not fatal yet
        log.warning("could not wake the phone: %s", exc)
    if dev.is_locked():
        return ("the phone is on the lock screen. A PIN, pattern or fingerprint "
                "is yours to enter -- unlock it and run this again")
    return ""


def open_app_verified(dev: Any, package: str, *, timeout_s: float = 10.0) -> "Screen":
    """Launch `package` and wait until it is really the app in front.

    ``app_start`` is fire-and-forget: handed a package that is not installed it
    returns happily having done nothing, and handed a real one it returns before
    the first frame is drawn. Both used to read as success, so an exploration
    could spend its entire budget describing the lock screen. The returned
    screen is the caller's evidence -- check its ``package``.
    """
    dev.open_app(package)
    deadline = time.monotonic() + timeout_s
    screen = dev.observe()
    while screen.package != package and time.monotonic() < deadline:
        time.sleep(0.4)
        screen = dev.observe()
    return screen


def exploration_goal(package: str, tasks: str) -> str:
    """The brief handed to the agent for a skill-generating run.

    Exploration is not a normal goal: nothing on screen can satisfy it. So the
    brief has to say what counts as covered and what to write down, or the model
    answers `done` on the app's first frame -- which is exactly what the old
    one-line goal ("Explore app X and perform the following tasks: ...") got,
    leaving the synthesis to invent a skill from a single screen.
    """
    return f"""\
Explore the Android app {package} and write down how to drive it, so a later run
can work in this app without getting lost. This is read-only reconnaissance: the
notes you leave behind are the deliverable, not a task finished on screen.

What to do in the app: {tasks}

Record what you learn as you go, using `notes` -- one or two records per new
screen, keyed so they do not overwrite each other:
  - "flow:<short name>"   the tap-by-tap steps you just performed, naming each
                          control exactly as it is labelled on screen.
  - "screen:<short name>" what a screen is for and how you got to it.
  - "quirk:<short name>"  anything that would mislead a later run: a control that
                          opens something other than its label suggests, a screen
                          `back` does not leave, a list whose items never appear
                          in the element list, an index that moves between turns.
  - "tip:<short name>"    what to do instead of that quirk.

How to cover the app:
  - Visit each top-level destination -- bottom tabs, top tabs, the nav drawer --
    and return with `press_key back` before trying the next one.
  - Open the search entry point, and open one item from the main list so you see
    a detail screen as well as the lists.
  - Do NOT send, post, buy, pay, delete, log out, or change a setting. When a
    control would do any of those, record what it is and leave it alone.
  - Do NOT swipe, like, pass, match, react to or vote on a card, profile or post.
    In some apps that gesture is not navigation -- it is a message delivered to a
    real person, and it cannot be taken back. Read such a screen, record how it
    works, and leave with `press_key back`.
  - Decline permission prompts and dismiss rate-this-app popups.
  - Record what a control is and where, never what it currently holds. "The row shows
    the account's city" is a note; the city is somebody's address. The same goes for
    names, ages, photos, message text, balances and counts -- all of them are wrong by
    tomorrow, and the first few are not yours to write down.

Answer `done` once you have covered the main screens, or finished the tasks
above, and summarise what you covered in `text`. Stop when the screens start
repeating -- there is nothing left to learn from a fourth pass over the same list.
"""


@dataclass
class AppTrace:
    """What one run saw inside one app. The input to skill synthesis.

    Filled the same way whether the run was a skill exploration or an ordinary
    goal: both are a phone being driven through an app, and what a skill wants
    to know from them is identical.
    """

    query: str = ""
    #: The package actually driven, resolved from `query` and verified in front.
    package: str = ""
    tasks: str = ""
    #: One line per *distinct* screen reached, not per step.
    screens: List[str] = field(default_factory=list)
    #: Every step, in order, with what the model said it saw.
    actions: List[str] = field(default_factory=list)
    screenshots: List[bytes] = field(default_factory=list)
    #: The run's collected-data ledger -- the explorer's own findings.
    notes: str = ""
    #: How the app was picked: "named", "tasks" or "foreground". Reported,
    #: because two of the three are the harness guessing.
    chosen_by: str = ""
    outcome: str = ""
    steps: int = 0
    llm_calls: int = 0
    #: The run whose artifacts this trace came from, so a caller can find its
    #: `runs/<id>/` -- and write the synthesis that follows into the same log.
    run_id: str = ""

    @property
    def looked_around(self) -> bool:
        """Did the run actually move through the app?

        Synthesis fed one screen invents the other nine, and the invention gets
        filed under the right app name, which is worse than having no skill at
        all. Callers report this rather than presenting the result as learned.
        """
        return len(self.screens) > 1


def _summarise(screen: "Screen", step: int) -> str:
    labels = [e.best_text for e in screen.elements if e.best_text][:18]
    head = f"step {step}: {screen.package}{screen.activity}"
    return f"{head} | {', '.join(labels)}" if labels else f"{head} | (nothing labelled)"


def _describe(action: Any, step: int) -> str:
    parts = [f"step {step}: {action.describe() if hasattr(action, 'describe') else action}"]
    for label, value in (("saw", getattr(action, "observation", "")),
                         ("result", getattr(action, "_result_summary", ""))):
        if value:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


class TraceCollector:
    """An `on_event` hook that turns the agent's event stream into an `AppTrace`.

    Wrap whatever reporter the caller already has -- events pass straight
    through -- then read `.trace` once the run is over. Attach it to any run,
    not just an exploration: a goal pursued in an app is a tour of that app too,
    and `learn_from_run` is what makes the second one count.

    Screens are kept once per distinct structure rather than once per step: a
    40-step tour crosses the same list a dozen times, and a prompt that pays per
    screenshot should carry twelve *different* screens -- which is not what a
    "first N steps" cap collects.
    """

    def __init__(self, dev: Any, trace: Optional[AppTrace] = None,
                 on_event: Optional[Callable[..., None]] = None):
        self.dev = dev
        self.trace = trace or AppTrace()
        self.on_event = on_event or (lambda *a, **k: None)
        self.seen: set = set()
        #: Steps spent in each package, so a run that crosses apps can say which
        #: one it was mostly working in.
        self.steps_in: Dict[str, int] = {}

    def __call__(self, kind: str, **kw: Any) -> None:
        self.on_event(kind, **kw)
        if kind != "step":
            return
        state = kw.get("state")
        step = kw.get("step") or (state.step if state is not None else 0)
        screen = kw.get("screen")
        action = kw.get("action")
        if screen is not None:
            self.record(screen, step)
        if action is not None:
            self.trace.actions.append(_describe(action, step))

    def record(self, screen: "Screen", step: int) -> None:
        if screen.package:
            self.steps_in[screen.package] = self.steps_in.get(screen.package, 0) + 1
        key = screen.skeleton_id or f"{screen.package}{screen.activity}"
        if key in self.seen:
            return
        self.seen.add(key)
        self.trace.screens.append(_summarise(screen, step))
        if len(self.trace.screenshots) >= MAX_EXPLORE_SHOTS:
            return
        # Verification already grabbed a frame of this screen most turns; reuse
        # it rather than paying a second round trip for the same pixels.
        shot = screen.screenshot
        if shot is None:
            try:
                shot = self.dev.screenshot()
            except Exception as exc:  # noqa: BLE001
                log.warning("no screenshot for step %d: %s", step, exc)
                return
        if shot:
            self.trace.screenshots.append(shot)

    @property
    def main_package(self) -> str:
        """The app this run mostly worked in.

        A goal like "open WhatsApp and share it to Drive" touches three
        packages; the skill worth updating is the one the steps were spent in,
        not whichever happened to be in front when the run ended.
        """
        app_steps = {p: n for p, n in self.steps_in.items() if is_app_package(p)}
        if not app_steps:
            return ""
        return max(app_steps.items(), key=lambda kv: kv[1])[0]

    def finish(self, outcome: str, state: Any) -> AppTrace:
        """Close the trace off with what the finished run knows."""
        self.trace.outcome = outcome
        self.trace.steps = getattr(state, "step", 0)
        self.trace.llm_calls = getattr(state, "llm_calls", 0)
        self.trace.run_id = getattr(state, "run_id", "")
        scratchpad = getattr(state, "scratchpad", None)
        if scratchpad is not None:
            self.trace.notes = scratchpad.plain()
        if not self.trace.package:
            self.trace.package = self.main_package
        return self.trace


def explore_app(dev: Any, mem: Any, llm: Any, cfg: Any, *, query: str = "",
                tasks: str = "",
                on_event: Optional[Callable[..., None]] = None) -> AppTrace:
    """Drive an app on the phone and record what a skill needs to know.

    `query` is a package, an app name, or empty for whatever is in front.

    Raises `ExplorationBlocked` when the phone cannot be driven: a locked
    screen, an app that is not installed, an app that will not come forward.
    The command this replaces folded all three into one warning and then
    synthesised a skill from whatever happened to be on screen.
    """
    from .agent import Agent

    if llm is None:
        raise ExplorationBlocked("exploring needs a model; none was configured")

    exp = AppTrace(query=query, tasks=(tasks or "").strip() or DEFAULT_EXPLORE_TASKS)

    blocked = ready_for_exploration(dev)
    if blocked:
        raise ExplorationBlocked(blocked)

    target, exp.chosen_by = query, "named"
    if not target:
        # No app argument. The tasks usually name one already -- "open Bumble and
        # read the matches" -- and saying it a second time is only a second
        # chance to get it wrong. An app the tasks name beats whatever happens to
        # be on screen; the foreground is the fallback when they name none.
        named, candidates = package_from_text(dev, tasks)
        if not named and len(candidates) > 1:
            raise ExplorationBlocked(
                "the tasks name more than one installed app ("
                + ", ".join(candidates) + "), so which to explore is your call: "
                "`adbagent skills generate <app> --tasks ...`")
        target, exp.chosen_by = named, ("tasks" if named else "foreground")

    if target:
        package = resolve_package(dev, target)
        if not package:
            hint = target.split(".")[0]
            raise ExplorationBlocked(
                f"no installed app matches {target!r}. "
                f"`adbagent apps -s {hint}` lists what the phone actually has")
        screen = open_app_verified(dev, package)
        if screen.package != package:
            raise ExplorationBlocked(
                f"{package} would not come to the foreground "
                f"({screen.package or 'nothing'} is in front). Open it by hand once, "
                "then run this again")
    else:
        # Nothing named anywhere: explore whatever the phone is showing. Saves
        # naming an app you are already looking at, and is the only way to build
        # a skill for a screen you cannot reach from a cold launch.
        screen = dev.observe()
        package = screen.package
        if not is_app_package(package):
            raise ExplorationBlocked(
                f"the app in front is {package or 'nothing'}, which is the phone "
                "rather than an app. Name one, or say which app the tasks are "
                "about: `adbagent skills generate whatsapp`")

    exp.package = package
    exp.actions.append(f"opened {package}"
                       + (f" (resolved from {query!r})" if query and package != query else ""))

    collector = TraceCollector(dev, exp, on_event)
    collector.record(screen, 0)

    agent = Agent(dev, mem, llm, cfg, on_event=collector)
    outcome, state = agent.run(exploration_goal(package, exp.tasks))
    return collector.finish(outcome, state)


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

Ground every workflow, nuance and recommendation in the exploration below. Name the
controls with the labels the exploration actually saw. If the exploration only reached
a few screens, write a short skill about those screens -- do NOT fill the gaps with
what an app of this kind usually looks like. An invented step filed under the right app
name is worse than a missing one: the next run will follow it.

"packages" must list the RESOLVED PACKAGE exactly as given, since that is what the
agent matches a skill against at run time.

When the run failed or got stuck, the trace is evidence about the app wherever it shows
why -- a control that does nothing, a screen `back` will not leave, a list the element
tree never shows. Record those as nuances with the way around them. Do not record a
one-off as a rule: a slow load, a single mistap or a popup that appeared once is not a
property of the app.

Two traps in particular, both of which have produced confident and false nuances:

* An action that failed once and then worked is a transition or timing artefact, not a
  property of the app. Launching an app is the usual case: the screen mid-launch is
  neither the old app nor the new one, and a check made on that frame fails. Never
  write it up as the action doing the wrong thing.
* What was on screen BEFORE an action is not what the action did. If the app you were
  leaving is named in an observation, that is where the run started, not where the
  action took it.

Say only what the trace supports. "This failed once here" is not "this does not work".

Write about the app, never about the account using it. A skill is re-read on every run
and is not the place for either of these:

* The account holder's own data -- their name, age, city, photos, who they talk to,
  what a message said. Describe the control, not its contents: "the row shows the
  current city", not the city.
* State that will have changed by tomorrow -- a completeness percentage, a credit
  balance, an unread count, how many items a list held. Say that the figure is shown
  and where; do not record its value.

Both make a skill wrong within a day, and the first sends personal data to a model on
every later run for no benefit.
Reply with ONLY the JSON object.
"""


class SkillGenerator:
    """Generates or updates an App Skill by executing user task instructions on a device and synthesizing findings using LLM."""

    def __init__(self, registry: Optional[SkillRegistry] = None):
        self.registry = registry or SkillRegistry()

    @staticmethod
    def _ask(llm_client: Any, user_content: Any, model: str) -> str:
        """One synthesis call, returned as bare JSON text.

        Fences come back whatever the prompt asks for, so they are stripped here
        rather than at each call site.
        """
        messages = [{"role": "system", "content": SKILL_SYNTHESIS_SYSTEM},
                    {"role": "user", "content": user_content}]
        if hasattr(llm_client, "_post"):
            cfg = getattr(llm_client, "cfg", None)
            raw_text, _ = llm_client._post(
                messages, model=model, schema=None,
                max_tokens=cfg.llm.max_tokens if cfg is not None else 1500,
                purpose="skill_generation")
            content = raw_text.strip()
        elif hasattr(llm_client, "_post_chat"):
            raw_resp = llm_client._post_chat(messages, model=model)
            content = raw_resp["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError("Unsupported LLM client interface")

        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        return content.strip()

    def generate_from_exploration(self, app_name_or_pkg: str, tasks: str,
                                 screen_summaries: List[str],
                                 actions_taken: List[str],
                                 llm_client: Any,
                                 screenshots: Optional[List[bytes]] = None,
                                 *, package: str = "", notes: str = "",
                                 outcome: str = "", history: str = "") -> Skill:
        """Synthesize exploration observations into a Skill object, updating existing skill if present.

        `package` is the package the exploration verifiably drove, which is not
        always what the caller asked for: ``generate whatsapp`` explores
        ``com.whatsapp``, and a skill that does not say so never loads, because
        `find_by_package` is what the agent matches on.
        """
        existing_skill = ((self.registry.find_by_package(package) if package else None) or
                          self.registry.find_by_name_or_alias(app_name_or_pkg) or
                          self.registry.find_by_package(app_name_or_pkg))

        prompt_parts = [f"APP NAME OR PACKAGE: {app_name_or_pkg}"]
        if package:
            prompt_parts.append(f"RESOLVED PACKAGE: {package}")
        if existing_skill:
            prompt_parts.append(f"EXISTING SKILL DEFINITION:\n{json.dumps(existing_skill.to_dict(), indent=2)}")
        prompt_parts.append(f"USER TASKS PERFORMED: {tasks}")
        if notes:
            prompt_parts.append("FINDINGS THE EXPLORER RECORDED (its own words, the primary source):\n" + notes)
        prompt_parts.append("DISTINCT SCREENS REACHED:\n" + "\n".join(screen_summaries or ["(none)"]))
        prompt_parts.append("ACTIONS TAKEN DURING EXPLORATION:\n" + "\n".join(actions_taken or ["(none)"]))
        if outcome:
            prompt_parts.append(f"HOW THE EXPLORATION ENDED: {outcome}")
        if history:
            prompt_parts.append(history)
        prompt_parts.append(f"Generate an updated, complete App Skill JSON object for {package or app_name_or_pkg} combining existing skill information and new exploration findings.")

        prompt = "\n\n".join(prompt_parts)

        if hasattr(llm_client, "cfg"):
            model_to_use = llm_client.cfg.llm.skill() or llm_client.cfg.llm.image() or llm_client.model
        else:
            model_to_use = getattr(llm_client, "model", "")

        # Attempts, best first. A text-only `model_skill` fails the *whole* call
        # when it is handed an image part -- the same trap `llm.vision_in_decider`
        # documents -- and losing a run's whole trace to that is far worse than
        # losing the pictures, so the words get a second chance on their own.
        attempts: List[Any] = [prompt]
        if screenshots:
            from .llm import image_part, text_part
            with_images: List[Dict[str, Any]] = [text_part(prompt)]
            for shot in screenshots:
                if shot:
                    with_images.append(image_part(shot))
            attempts.insert(0, with_images)

        generated_skill: Optional[Skill] = None
        for content in attempts:
            try:
                data = json.loads(self._ask(llm_client, content, model_to_use))
                generated_skill = Skill.from_dict(data)
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("skill synthesis %s failed: %s",
                            "with screenshots" if isinstance(content, list) else "on text alone",
                            exc)

        if generated_skill is None:
            log.warning("no usable skill JSON; falling back to a template")
            anchor = package or app_name_or_pkg
            generated_skill = Skill(
                name=existing_skill.name if existing_skill else anchor.replace(".", "_").capitalize(),
                packages=[anchor] if ("." in anchor and not existing_skill) else [],
                aliases=[app_name_or_pkg.lower()],
                description=f"App skill for {anchor} based on user tasks: {tasks}",
                workflows=[Workflow(name="user_tasks", steps=tasks)],
                nuances=["Observe screen element indices carefully before tapping."],
                recommendations=["Use input_text with clear=true when editing fields."]
            )

        if existing_skill:
            final_skill = existing_skill.merge(generated_skill)
        else:
            final_skill = generated_skill

        # The binding the agent matches on, whatever the model chose to write.
        if package and not final_skill.matches_package(package):
            final_skill.packages.append(package)
        # And the name the caller typed, so `skills view <that>` finds it again.
        alias = app_name_or_pkg.strip().lower()
        if (alias and "." not in alias and alias != final_skill.name.lower()
                and alias not in (a.lower() for a in final_skill.aliases)):
            final_skill.aliases.append(alias)

        self.registry.save_skill(final_skill)
        return final_skill


# ---------------------------------------------------------------------------
# Learning from an ordinary run
# ---------------------------------------------------------------------------

#: A run has to have got somewhere before it can teach anything. One step into an
#: app and out again leaves nothing a skill did not already say, and paying a
#: synthesis call to hear it back only dilutes what is there.
MIN_LEARNABLE_STEPS = 3


def learn_from_run(trace: AppTrace, llm: Any, registry: SkillRegistry, *,
                   goal: str = "", cfg: Any = None) -> Optional[Skill]:
    """Fold what a finished run learned into that app's skill, and save it.

    This is what makes the agent better at an app the more it is used there: any
    run is a tour of the app it ran in, the skill it read at the start comes back
    as the baseline, and `Skill.merge` keeps what was already known rather than
    replacing it. Returns the updated skill, or None when the run has nothing to
    teach.

    Given `cfg`, the recorded runs for this app are read too, and what repeats
    across them goes into the prompt alongside this run's trace. That is the
    only way the "dead end, not a one-off" judgement the synthesis is asked for
    can actually be made: one trace cannot distinguish a control that always
    does nothing from one that was slow once.

    A failed run is not skipped. "Tapping this row does nothing" and "back leaves
    the app from here" are exactly what a skill is for, and they are only ever
    learned by a run that went wrong.
    """
    if not is_app_package(trace.package):
        log.info("nothing to learn: %r is not an app", trace.package)
        return None
    if trace.steps < MIN_LEARNABLE_STEPS or not trace.looked_around:
        log.info("nothing to learn from %s: %d step(s), %d screen(s)",
                 trace.package, trace.steps, len(trace.screens))
        return None
    return SkillGenerator(registry).generate_from_exploration(
        trace.package, goal or trace.tasks, trace.screens, trace.actions, llm,
        screenshots=trace.screenshots, package=trace.package,
        notes=trace.notes, outcome=trace.outcome,
        history=run_history(cfg, trace.package))


def run_history(cfg: Any, package: str) -> str:
    """The digest of earlier runs in this app, or "" when there is none.

    Never fatal: a skill written from this run alone is worth having, and losing
    it to an unreadable run directory would not be.
    """
    if cfg is None or not package:
        return ""
    try:
        from .history import for_package
        return for_package(cfg, package).to_prompt_text()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read run history for %s: %s", package, exc)
        return ""
