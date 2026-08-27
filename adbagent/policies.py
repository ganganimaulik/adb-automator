"""Policy files: several of them, each carrying the goal it was written for.

A policy is the operator's reply instructions, injected into the prompt verbatim
(`prompts.policy_block`). For most of this project's life there was exactly one,
named by `watch.policy`, and the goal was typed in beside it -- which is fine
until there are two apps. Then the pairing matters: the Hinge policy is only
correct under "work through Discover and reply to matches", and starting it under
the goal that was still in the box from the Instagram policy is a watch doing the
wrong thing carefully. The two were always one decision and nothing held them
together.

So the goal lives *in* the policy file, as front matter:

    ---
    goal: work through the Hinge feed and reply to anyone new
    ---

    # Hinge policy
    - Only ever like the first photo...

One file, portable, still a plain markdown document anyone can read. The front
matter is metadata about the policy, not part of it, so `instructions()` strips
it before the text ever reaches a prompt -- otherwise the goal would arrive twice
and the fence would arrive as an instruction.

Deliberately a hand-written parser rather than YAML: this is a `key: value` block
of at most a few keys, PyYAML is not a dependency of this project, and adding one
to read two fields is a poor trade. The parser is strict about what counts as
front matter and falls back to "this file has none" on anything it does not
recognise, because the failure that matters is a markdown horizontal rule at the
top of a policy being eaten as metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: The fence, on its own line. Three dashes exactly -- a longer rule is a rule.
FENCE = "---"

#: Suffixes a policy file may have. Markdown because a policy is prose with
#: structure; `.txt` because plenty of them are written without any.
SUFFIXES = (".md", ".markdown", ".txt")

#: Keys this module understands, in the order they are written back out. Anything
#: else found in the block is preserved and re-rendered after these, so a key
#: added by a later version of this file survives a save by an earlier one.
KNOWN_KEYS = ("goal", "title")

#: What a policy may be called when it is referred to by name rather than by
#: path. No separators and no dots, so a name can never escape the directory or
#: pick its own extension.
_NAME_OK = re.compile(r"[^A-Za-z0-9._-]+")

#: `key: value`, and the `key: |` that opens an indented block.
_KEY_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$")

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")


# ---------------------------------------------------------------------------
# front matter
# ---------------------------------------------------------------------------

def split_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """``(metadata, instructions)`` for one policy file's text.

    Returns ``({}, text)`` -- text untouched, not even stripped -- for a file
    with no front matter, which is every policy written before this existed.

    The block is only recognised when the file opens with the fence, a closing
    fence is found, and *every* non-blank line between them parses as a key or as
    the continuation of one. A policy that opens with a horizontal rule therefore
    reads as a policy with no metadata rather than as a mangled one: the
    alternative is silently eating the first paragraph of somebody's
    instructions, and the instructions are the part that talks to people.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FENCE:
        return {}, text

    close = next((i for i in range(1, len(lines))
                  if lines[i].strip() == FENCE), -1)
    if close < 0:
        return {}, text

    meta: Dict[str, str] = {}
    key = ""            # the key an indented continuation belongs to
    block: List[str] = []  # its lines so far

    def flush() -> None:
        if key:
            meta[key] = "\n".join(block).strip("\n")

    for raw in lines[1:close]:
        if key and (not raw.strip() or raw[:1] in (" ", "\t")):
            block.append(raw.strip())
            continue
        matched = _KEY_LINE.match(raw)
        if not matched:
            if not raw.strip():
                continue
            return {}, text     # not a front matter block after all
        flush()
        key, value = matched.group(1).lower(), matched.group(2).strip()
        if value in ("|", "|-", ">", ">-"):
            block = []          # a folded value: the lines below it
            continue
        meta[key] = _unquote(value)
        key, block = "", []
    flush()

    if not meta:
        return {}, text         # `---` twice with nothing in between is a rule
    return meta, "\n".join(lines[close + 1:]).lstrip("\n")


def with_front_matter(meta: Dict[str, str], body: str) -> str:
    """One policy file's text, from its metadata and its instructions.

    Empty metadata returns the body *exactly* as given -- no fence, no added
    newline. A policy with no goal saved from the editor is byte-for-byte the
    file that was typed, which is what anyone diffing it expects.
    """
    kept = {k: str(v) for k, v in meta.items() if str(v).strip()}
    if not kept:
        return body
    order = [k for k in KNOWN_KEYS if k in kept]
    order += sorted(k for k in kept if k not in KNOWN_KEYS)
    out = [FENCE]
    for k in order:
        value = kept[k].strip()
        if "\n" in value:
            out.append(f"{k}: |")
            out += [f"  {line}" if line.strip() else "" for line in value.splitlines()]
        else:
            out.append(f"{k}: {value}")
    out += [FENCE, ""]
    # A generated file ends in a newline. Only this branch: the empty-metadata
    # return above is the text somebody typed, given back unaltered.
    return "\n".join(out) + "\n" + body.lstrip("\n").rstrip("\n") + "\n"


def instructions(text: str) -> str:
    """The part of a policy file that goes into the prompt.

    The whole point of stripping: front matter is a note about the policy, and a
    model handed ``goal: ...`` inside its reply instructions has been told
    something nobody meant to tell it.
    """
    return split_front_matter(text)[1].strip()


def goal_in(text: str) -> str:
    """The goal a policy file was written for, or ""."""
    return split_front_matter(text)[0].get("goal", "").strip()


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def slug(name: str) -> str:
    """A file-safe policy name. "" when nothing usable is left.

    Not a general sanitiser: it is the answer to "the browser sent a name, what
    file does that mean", and every character that could make it mean a file
    somewhere else is gone rather than escaped.
    """
    cleaned = _NAME_OK.sub("-", (name or "").strip()).strip("-._")
    for suffix in SUFFIXES:
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned


def title_in(body: str) -> str:
    """The first markdown heading of a policy, as a name for it."""
    for line in body.splitlines():
        matched = _HEADING.match(line.strip())
        if matched:
            return matched.group(1).strip()
        if line.strip():
            break
    return ""


# ---------------------------------------------------------------------------
# one policy
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """One policy file, parsed. Exists or not: an unwritten one is normal."""

    path: Path
    #: How it is referred to when it is not a path: the file stem.
    name: str = ""
    #: The goal this policy was written for. The reason this module exists.
    goal: str = ""
    title: str = ""
    #: The instructions, front matter stripped. What reaches the prompt.
    body: str = ""
    exists: bool = False
    mtime: float = 0.0
    #: Front matter keys this version does not know about, kept for the save.
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """What to call it on screen, best available."""
        return self.title or title_in(self.body) or self.name or str(self.path)

    def meta(self) -> Dict[str, str]:
        return {**self.extra, "goal": self.goal, "title": self.title}

    def text(self) -> str:
        """The file, as it should be written."""
        return with_front_matter(self.meta(), self.body)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "path": str(self.path), "goal": self.goal,
                "title": self.title, "label": self.label, "body": self.body,
                "exists": self.exists, "mtime": self.mtime,
                "chars": len(self.body)}


def read(path) -> Policy:
    """Parse the policy at `path`. A missing file is a `Policy`, not an error.

    The editor asks for policies that have not been written yet -- that is what
    "new policy" is -- so absence travels in `exists` rather than as an
    exception. `watch.load_policy` is the one that refuses.
    """
    p = Path(path).expanduser()
    policy = Policy(path=p, name=p.stem)
    try:
        text = p.read_text(encoding="utf-8")
        policy.mtime = p.stat().st_mtime
        policy.exists = True
    except OSError:
        return policy
    meta, body = split_front_matter(text)
    policy.body = body.strip()
    policy.goal = meta.pop("goal", "").strip()
    policy.title = meta.pop("title", "").strip()
    policy.extra = meta
    return policy


# ---------------------------------------------------------------------------
# the directory of them
# ---------------------------------------------------------------------------

class PolicyStore:
    """The policies on disk: list them, name them, read and write them.

    A directory (`watch.policies_dir`) plus the one `watch.policy` names, which
    may live outside it -- a path configured before there was a directory still
    has to appear in the list, or the UI would offer to switch away from the
    policy that is actually configured and not offer to switch back.
    """

    def __init__(self, directory: str = "", current: str = "") -> None:
        self.dir: Optional[Path] = (Path(directory).expanduser()
                                   if str(directory).strip() else None)
        raw = str(current).strip()
        self.current: Optional[Path] = Path(raw).expanduser() if raw else None

    # -- naming ---------------------------------------------------------

    def path_for(self, name: str) -> Path:
        """The file a bare policy name means. Raises on a name that means none."""
        cleaned = slug(name)
        if not cleaned:
            raise ValueError(f"{name!r} is not a usable policy name")
        base = self.dir or Path(".")
        return base / f"{cleaned}.md"

    def resolve(self, ref: str = "") -> str:
        """The policy file `ref` refers to -- a path or a bare name -- or "".

        A string rather than a Path, because `Path("")` is `Path(".")`, whose
        truthiness makes every "no policy set" guard downstream silently pass
        and land a write on a directory.

        Three readings, in order: a path that exists, a bare name in the
        policies directory, and a path that does not exist yet (the editor's
        normal state before a save).
        """
        ref = (ref or "").strip()
        if not ref:
            return str(self.current) if self.current else ""
        p = Path(ref).expanduser()
        if p.is_file():
            return str(p)
        # `--policy hinge_policy.md` from the repo root, with the file in
        # `policies/`: it named a file, just not from here.
        if self.dir is not None and (self.dir / p.name).is_file():
            return str(self.dir / p.name)
        bare = len(p.parts) == 1 and not p.suffix
        if bare and self.dir is not None:
            return str(self.path_for(p.name))
        return str(p)

    def owns(self, path) -> bool:
        """Whether this store is allowed to write `path`.

        The guard on the write endpoint. The browser sends a path now -- it has
        to, since it picks between several -- and "a path the server will write
        whatever arrives to" is not something to leave open to a page that can
        be reached from another tab.
        """
        p = Path(path).expanduser()
        if self.current is not None and same_file(p, self.current):
            return True
        if self.dir is None:
            return False
        try:
            resolved_dir = self.dir.resolve()
            resolved = (p if p.is_absolute() else Path.cwd() / p).resolve()
        except OSError:
            return False
        return resolved.parent == resolved_dir and resolved.suffix.lower() in SUFFIXES

    # -- reading --------------------------------------------------------

    def read(self, ref: str = "") -> Optional[Policy]:
        """The policy `ref` names, or None when nothing names one."""
        found = self.resolve(ref)
        return read(found) if found else None

    def list(self) -> List[Policy]:
        """Every policy there is, by name, the configured one included.

        Sorted by name rather than by mtime: this is a list somebody picks from,
        and a list that reorders itself because you saved one is a list where
        the entry you want has moved.
        """
        found: Dict[str, Policy] = {}
        if self.dir is not None:
            try:
                entries = sorted(self.dir.iterdir())
            except OSError:      # no directory yet: the normal state on day one
                entries = []
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() in SUFFIXES:
                    found[str(entry.resolve())] = read(entry)
        # The configured policy, whether or not it has been written. It is what
        # a bare `adbagent watch` runs and what the UI opens on, so a picker that
        # left it out on the grounds that the file is not there yet would be a
        # picker with no way back to the policy that is actually configured.
        if self.current is not None:
            found.setdefault(str(self.current.resolve()), read(self.current))
        return sorted(found.values(), key=lambda p: (p.name.lower(), str(p.path)))

    # -- writing --------------------------------------------------------

    def save(self, policy: Policy) -> Path:
        """Write one policy, creating its directory if need be."""
        policy.path.parent.mkdir(parents=True, exist_ok=True)
        policy.path.write_text(policy.text(), encoding="utf-8")
        return policy.path

    def write(self, ref: str, body: str, *, goal: str = "",
              title: Optional[str] = None) -> Policy:
        """Save `body` (and the goal it belongs to) as the policy `ref` names.

        `goal` is authoritative -- an empty one clears it, because the editor
        always sends the box and an emptied box means what it says. `title` is
        left alone when omitted rather than cleared: nothing in the UI edits it,
        so a form that does not send one is not a request to drop the one
        somebody wrote by hand. Metadata this version does not know about is
        carried over from disk for the same reason.
        """
        found = self.resolve(ref)
        if not found:
            raise ValueError("no policy path given, and none in config")
        policy = read(found)
        policy.body = body.strip()
        policy.goal = goal.strip()
        if title is not None:
            policy.title = title.strip()
        self.save(policy)
        policy.exists = True
        return policy


def same_file(a, b) -> bool:
    """Whether two paths name one file, without needing either to exist.

    String comparison is not enough and never was: the config says
    ``policies/hinge.md``, the picker sends the absolute path the list gave it,
    and on Windows either may arrive with the other's separators.
    """
    if not a or not b:
        return False
    try:
        return (Path(a).expanduser().resolve()
                == Path(b).expanduser().resolve())
    except OSError:
        return str(a) == str(b)


def store_for(cfg) -> PolicyStore:
    """The store a `Config` describes."""
    return PolicyStore(cfg.watch.policies_dir, cfg.watch.policy)
