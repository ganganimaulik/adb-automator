"""The self-improving cache: anchors, admission gates, SQLite storage.

The contract the whole project rests on:

    The first time a screen is seen, an LLM decides what to do and we record it.
    Every later time that screen is recognised in the same goal context, we
    replay the recorded decision with no LLM call at all. The LLM comes back
    only when a replay fails.

Two things make that safe rather than reckless.

**Anchors, not coordinates.** We never store `(x, y)`. We store a weighted
description of the element -- its id, text, class, parent, position in its
scroller -- and re-resolve it against the live screen every time. A layout that
shifts by 40 pixels still resolves; a screen where the element is genuinely gone
resolves to nothing and falls back to the LLM.

**Three admission gates.** Fingerprint match, discriminative tokens, and anchor
resolvability. All three must pass. The middle one is the answer to the failure
mode that matters most -- a cache entry firing on a screen that merely *looks*
like the one it was learned on.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from . import trust
from .actions import AgentAction, Postcondition
from .config import Config
from .fingerprint import class_eq, hamming, mask_text, rid_norm
from .screen import Element, Screen

log = logging.getLogger("adbagent.memory")

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entry (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    app_key           TEXT NOT NULL,
    skeleton_id       TEXT NOT NULL,
    simhash           INTEGER NOT NULL,
    intent_id         TEXT NOT NULL,
    visit_ordinal     INTEGER NOT NULL DEFAULT 0,
    scope             TEXT NOT NULL DEFAULT 'step',

    anchor_json       TEXT,
    action_json       TEXT NOT NULL,
    postcondition_json TEXT,

    required_tokens   TEXT NOT NULL DEFAULT '[]',
    forbidden_tokens  TEXT NOT NULL DEFAULT '[]',

    next_skeleton_id  TEXT,
    alt_successors    TEXT NOT NULL DEFAULT '[]',

    state             TEXT NOT NULL DEFAULT 'probation',
    version           INTEGER NOT NULL DEFAULT 1,
    parent_id         INTEGER REFERENCES entry(id) ON DELETE SET NULL,

    n_success         REAL NOT NULL DEFAULT 0,
    n_failure         REAL NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,

    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    last_used_at      REAL
);

CREATE INDEX IF NOT EXISTS entry_lookup
    ON entry(app_key, skeleton_id, intent_id, visit_ordinal);
CREATE INDEX IF NOT EXISTS entry_app ON entry(app_key, state);

CREATE TABLE IF NOT EXISTS entry_outcome (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id       INTEGER NOT NULL REFERENCES entry(id) ON DELETE CASCADE,
    run_id         TEXT NOT NULL,
    grade          TEXT NOT NULL,
    match_distance INTEGER NOT NULL DEFAULT 0,
    anchor_score   REAL NOT NULL DEFAULT 0,
    reason         TEXT,
    at             REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS outcome_entry ON entry_outcome(entry_id, at);

CREATE TABLE IF NOT EXISTS transition (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    app_key      TEXT NOT NULL,
    from_skeleton TEXT NOT NULL,
    to_skeleton  TEXT NOT NULL,
    action_sig   TEXT NOT NULL,
    n_seen       INTEGER NOT NULL DEFAULT 1,
    updated_at   REAL NOT NULL,
    UNIQUE(app_key, from_skeleton, to_skeleton, action_sig)
);

CREATE TABLE IF NOT EXISTS screen_seen (
    app_key     TEXT NOT NULL,
    skeleton_id TEXT NOT NULL,
    token       TEXT NOT NULL,
    PRIMARY KEY (app_key, skeleton_id, token)
);

CREATE TABLE IF NOT EXISTS run (
    run_id      TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    intent_id   TEXT NOT NULL,
    outcome     TEXT,
    steps       INTEGER NOT NULL DEFAULT 0,
    llm_calls   INTEGER NOT NULL DEFAULT 0,
    cache_hits  INTEGER NOT NULL DEFAULT 0,
    usd         REAL NOT NULL DEFAULT 0,
    started_at  REAL NOT NULL,
    ended_at    REAL
);
"""


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

W_RID = 0.40
W_TEXT = 0.20
W_DESC = 0.15
W_CLASS = 0.10
W_PARENT = 0.10
W_SIBLING = 0.05


class Anchor(BaseModel):
    """A description of an element, stable enough to survive a layout shift."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str = ""
    resource_id_raw: str = ""
    text: str = ""
    content_desc: str = ""
    class_eq: str = ""
    class_raw: str = ""
    parent_class_eq: str = ""
    parent_resource_id: str = ""
    sibling_index: int = -1
    scroller_rid: str = ""
    #: (x, y, w, h) as fractions of the screen, so density does not matter.
    bounds_frac: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    depth: int = 0
    path: str = ""
    #: "attributed" when there is something to match on; "coordinate_only" when
    #: the element has no id, text or description at all (WebView, canvas).
    kind: str = "attributed"

    def describe(self) -> str:
        return (self.resource_id or self.text or self.content_desc
                or f"{self.class_eq}@{self.path}")


def build_anchor(el: Element, screen: Screen) -> Anchor:
    """Read a real element and record what makes it findable again.

    The model never writes an anchor -- it only ever picks a `#N`. This function
    reads the actual node, so anchors describe reality rather than a guess.
    """
    width = screen.width or 1
    height = screen.height or 1
    parent = el.parent
    scroller = el.scroller()
    siblings = parent.children if parent is not None else []
    try:
        sibling_index = siblings.index(el)
    except ValueError:
        sibling_index = -1

    has_identity = bool(el.resource_id or el.text or el.content_desc or el.label)
    l, t, r, b = el.bounds
    return Anchor(
        resource_id=rid_norm(el.resource_id),
        resource_id_raw=el.resource_id_raw,
        text=el.best_text[:120],
        content_desc=el.content_desc[:120],
        class_eq=class_eq(el.cls),
        class_raw=el.cls,
        parent_class_eq=class_eq(parent.cls) if parent is not None else "",
        parent_resource_id=rid_norm(parent.resource_id) if parent is not None else "",
        sibling_index=sibling_index,
        scroller_rid=rid_norm(scroller.resource_id) if scroller is not None else "",
        bounds_frac=(l / width, t / height, (r - l) / width, (b - t) / height),
        depth=el.depth,
        path=el.path,
        kind="attributed" if has_identity else "coordinate_only",
    )


def _tokens(s: str) -> set:
    return {t for t in mask_text(s).split() if t}


def _text_match(recorded: str, candidate: str) -> float:
    """Fuzzy, never `==`. Labels drift across versions, locales and pluralisation."""
    a, b = recorded.strip().lower(), candidate.strip().lower()
    if not a:
        return 0.0
    if a == b:
        return 1.0
    if a and b and (a in b or b in a):
        return 0.7
    ta, tb = _tokens(recorded), _tokens(candidate)
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard >= 0.6:
            return 0.5
    return 0.0


def score_anchor(anchor: Anchor, el: Element) -> float:
    """Weighted similarity in [0, 1], normalised over recorded features only.

    Normalising over what was actually recorded matters: an element with no
    resource-id must not be permanently penalised for lacking one.
    """
    total = 0.0
    got = 0.0

    if anchor.resource_id:
        total += W_RID
        if rid_norm(el.resource_id) == anchor.resource_id:
            # Full credit only when the fully-qualified id matches too; a
            # normalised match across packages is weaker evidence.
            got += W_RID * (1.0 if el.resource_id_raw == anchor.resource_id_raw
                            else 0.75)

    if anchor.text:
        total += W_TEXT
        got += W_TEXT * _text_match(anchor.text, el.best_text)

    if anchor.content_desc:
        total += W_DESC
        got += W_DESC * _text_match(anchor.content_desc, el.content_desc)

    if anchor.class_eq:
        total += W_CLASS
        if class_eq(el.cls) == anchor.class_eq:
            got += W_CLASS

    if anchor.parent_class_eq or anchor.parent_resource_id:
        total += W_PARENT
        parent = el.parent
        if parent is not None:
            hits = 0
            if anchor.parent_class_eq and class_eq(parent.cls) == anchor.parent_class_eq:
                hits += 1
            if (anchor.parent_resource_id
                    and rid_norm(parent.resource_id) == anchor.parent_resource_id):
                hits += 1
            wanted = bool(anchor.parent_class_eq) + bool(anchor.parent_resource_id)
            got += W_PARENT * (hits / wanted if wanted else 0.0)

    if anchor.sibling_index >= 0:
        total += W_SIBLING
        parent = el.parent
        if parent is not None:
            try:
                if parent.children.index(el) == anchor.sibling_index:
                    got += W_SIBLING
            except ValueError:
                pass

    return got / total if total else 0.0


@dataclass
class Resolution:
    element: Optional[Element]
    score: float = 0.0
    runner_up: float = 0.0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.element is not None


def resolve_anchor(anchor: Anchor, screen: Screen, *, threshold: float,
                   ambiguity_gap: float, dhash_ok: bool = False) -> Resolution:
    """Find the element this anchor describes on the live screen."""
    if not screen.elements:
        return Resolution(None, reason="screen has no elements")

    if anchor.kind == "coordinate_only":
        # The only path that replays raw geometry. Gated hard, because there is
        # nothing to verify against if it lands on the wrong thing.
        if not dhash_ok:
            return Resolution(None, reason="coordinate-only anchor needs an exact "
                                           "screen match")
        x = int(anchor.bounds_frac[0] * screen.width
                + anchor.bounds_frac[2] * screen.width / 2)
        y = int(anchor.bounds_frac[1] * screen.height
                + anchor.bounds_frac[3] * screen.height / 2)
        log.warning("replaying a coordinate-only anchor at (%d, %d)", x, y)
        synthetic = Element(cls=anchor.class_raw,
                            bounds=(x - 1, y - 1, x + 1, y + 1))
        return Resolution(synthetic, score=1.0, reason="coordinate-only")

    scored = sorted(((score_anchor(anchor, el), el) for el in screen.elements),
                    key=lambda pair: pair[0], reverse=True)
    top_score, top = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if top_score < threshold:
        return Resolution(None, top_score, runner_up,
                          f"best match scored {top_score:.2f} < {threshold:.2f}")

    if top_score - runner_up < ambiguity_gap:
        # Two candidates this close means we cannot tell them apart, and tapping
        # the wrong one of a pair is worse than asking the model again.
        return Resolution(None, top_score, runner_up,
                          f"ambiguous: {top_score:.2f} vs {runner_up:.2f}")

    if anchor.scroller_rid:
        # Inside a scroller, vertical position is meaningless -- the list moved.
        if not _within(anchor, top, screen, check_y=False):
            return Resolution(None, top_score, runner_up, "moved horizontally")
    elif not _within(anchor, top, screen, check_y=True):
        if top_score < 0.70:
            return Resolution(None, top_score, runner_up,
                              "moved a long way and the match is weak")

    if not _plausible_size(anchor, top, screen):
        return Resolution(None, top_score, runner_up,
                          "element is clipped or much smaller than recorded")

    return Resolution(top, top_score, runner_up)


def _within(anchor: Anchor, el: Element, screen: Screen, check_y: bool,
            tolerance: float = 0.15) -> bool:
    ax = anchor.bounds_frac[0] + anchor.bounds_frac[2] / 2
    ay = anchor.bounds_frac[1] + anchor.bounds_frac[3] / 2
    cx, cy = el.center
    if abs(cx / (screen.width or 1) - ax) > tolerance:
        return False
    if check_y and abs(cy / (screen.height or 1) - ay) > tolerance:
        return False
    return True


def _plausible_size(anchor: Anchor, el: Element, screen: Screen) -> bool:
    """Guard against a half-scrolled row whose visible box is clipped.

    u2 reports `getVisibleBoundsInScreen`, clipped to the display, so a partly
    scrolled element reports a small rect whose centre is not on the element.
    """
    recorded = anchor.bounds_frac[2] * anchor.bounds_frac[3]
    if recorded <= 0:
        return True
    live = (el.width / (screen.width or 1)) * (el.height / (screen.height or 1))
    return live >= recorded * 0.6 and el.width >= 8 and el.height >= 8


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

@dataclass
class CachedStep:
    id: int
    app_key: str
    skeleton_id: str
    simhash: int
    intent_id: str
    visit_ordinal: int
    scope: str
    anchor: Optional[Anchor]
    action: AgentAction
    postcondition: Optional[Postcondition]
    required_tokens: List[str]
    forbidden_tokens: List[str]
    next_skeleton_id: str
    alt_successors: List[str]
    state: str
    version: int
    stats: trust.Stats
    #: filled in by lookup(), for logging and calibration
    match_distance: int = 0
    anchor_score: float = 0.0

    def describe(self) -> str:
        return (f"entry#{self.id} v{self.version} [{self.state}] "
                f"{self.action.describe()}")


def intent_key(goal: str) -> str:
    """Normalised goal, so trivial rewording still hits the cache."""
    import hashlib
    normalised = " ".join(mask_text(goal).split())
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=8).hexdigest()


class Memory:
    """SQLite-backed store of learned steps."""

    def __init__(self, cfg: Config, path: Optional[Path] = None):
        self.cfg = cfg
        self.path = path or cfg.db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(DDL)
        self.db.execute(
            "INSERT INTO meta(k, v) VALUES('schema_version', ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(SCHEMA_VERSION),))
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- corpus, for discriminative tokens ---------------------------------

    def note_screen(self, screen: Screen) -> None:
        """Record which tokens this screen has, so IDF can be computed later."""
        rows = [(screen.package, screen.skeleton_id, token)
                for token in set(screen.tokens)]
        if not rows:
            return
        self.db.executemany(
            "INSERT OR IGNORE INTO screen_seen(app_key, skeleton_id, token) "
            "VALUES(?,?,?)", rows)
        self.db.commit()

    def idf(self, app_key: str) -> Dict[str, float]:
        import math
        total = self.db.execute(
            "SELECT COUNT(DISTINCT skeleton_id) AS n FROM screen_seen WHERE app_key=?",
            (app_key,)).fetchone()["n"] or 0
        if total <= 1:
            return {}
        out: Dict[str, float] = {}
        for row in self.db.execute(
                "SELECT token, COUNT(DISTINCT skeleton_id) AS df FROM screen_seen "
                "WHERE app_key=? GROUP BY token", (app_key,)):
            out[row["token"]] = math.log(total / max(1, row["df"]))
        return out

    # -- lookup ------------------------------------------------------------

    def lookup(self, screen: Screen, intent_id: str, visit: int,
               *, forbidden_now: Sequence[str] = (),
               banned_signatures: Sequence[str] = ()) -> Optional[CachedStep]:
        """Find a replayable step for this screen, or None to fall back to the LLM.

        Gate 1 is the fingerprint, gate 2 the discriminative tokens. Gate 3
        (anchor resolvability) is applied by the caller via `rehydrate`, because
        it may want to scroll and retry first.
        """
        if not self.cfg.memory.enabled:
            return None

        rows = self.db.execute(
            "SELECT * FROM entry WHERE app_key=? AND skeleton_id=? AND intent_id=? "
            "AND visit_ordinal=? AND state != 'retired' ORDER BY version DESC",
            (screen.package, screen.skeleton_id, intent_id, visit)).fetchall()

        if not rows:
            # Fallback tier: same app, near-identical SimHash, trusted only.
            rows = self.db.execute(
                "SELECT * FROM entry WHERE app_key=? AND intent_id=? "
                "AND visit_ordinal=? AND state='trusted'",
                (screen.package, intent_id, visit)).fetchall()
            rows = [r for r in rows
                    if hamming(r["simhash"], screen.simhash) <= self.cfg.memory.t_strict]

        present = set(screen.tokens)
        forbidden_live = set(forbidden_now)
        candidates: List[CachedStep] = []

        for row in rows:
            distance = hamming(row["simhash"], screen.simhash)
            if distance > self.cfg.memory.t_sim:
                continue
            entry = self._row_to_step(row)
            entry.match_distance = distance

            if not trust.may_replay(entry.state):
                continue
            if entry.action.signature() in banned_signatures:
                continue
            # Gate 2a: everything that made this screen distinctive must still
            # be here. This is what stops an entry firing on a look-alike.
            if not set(entry.required_tokens) <= present:
                log.debug("%s: required tokens missing", entry.describe())
                continue
            # Gate 2b: nothing that marks a *different* screen may be present,
            # and nothing irreversible may be on screen at all.
            if set(entry.forbidden_tokens) & (present | forbidden_live):
                log.debug("%s: forbidden token present", entry.describe())
                continue
            candidates.append(entry)

        if not candidates:
            return None
        candidates.sort(key=lambda e: (-e.stats.wilson(), e.match_distance))
        return candidates[0]

    def rehydrate(self, entry: CachedStep, screen: Screen,
                  minor_deviation: bool = False) -> Optional[AgentAction]:
        """Gate 3: bind the stored anchor to a live element.

        Returns an action whose target points at the element we actually found,
        so execution never relies on the index the model originally chose.
        """
        if entry.anchor is None:
            return entry.action

        threshold = trust.anchor_threshold(
            entry.state, self.cfg.memory.anchor_strict,
            self.cfg.memory.anchor_relaxed,
            minor_deviation=minor_deviation or entry.match_distance > self.cfg.memory.t_strict)

        resolution = resolve_anchor(
            entry.anchor, screen, threshold=threshold,
            ambiguity_gap=self.cfg.memory.ambiguity_gap,
            dhash_ok=(entry.match_distance == 0 and entry.state == "trusted"))
        entry.anchor_score = resolution.score

        if not resolution.ok:
            log.info("%s: anchor did not bind (%s)", entry.describe(),
                     resolution.reason)
            return None

        element = resolution.element
        assert element is not None
        bound = entry.action.model_copy(deep=True)
        if bound.target is not None:
            bound.target.index = element.index or None
            bound.target.resource_id = element.resource_id or None
            bound.target.text = element.best_text[:120] or None
            if bound.target.index is None and not (bound.target.resource_id
                                                   or bound.target.text):
                # A synthetic coordinate-only element has no index; fall back to
                # the recorded text so Target stays valid.
                bound.target.text = entry.anchor.text or entry.anchor.class_eq
        return bound

    # -- recording ---------------------------------------------------------

    def record(self, *, screen: Screen, intent_id: str, visit: int,
               action: AgentAction, element: Optional[Element],
               postcondition: Optional[Postcondition], after: Screen,
               run_id: str) -> CachedStep:
        """Learn a step that the LLM chose and that verified successfully."""
        from .fingerprint import destructive_tokens, required_tokens

        anchor = build_anchor(element, screen) if element is not None else None
        idf = self.idf(screen.package)
        required = required_tokens(screen.tokens, idf)
        forbidden = destructive_tokens(after)

        now = time.time()
        cursor = self.db.execute(
            "INSERT INTO entry(app_key, skeleton_id, simhash, intent_id, "
            " visit_ordinal, scope, anchor_json, action_json, postcondition_json,"
            " required_tokens, forbidden_tokens, next_skeleton_id, alt_successors,"
            " state, version, created_at, updated_at) "
            "VALUES(?,?,?,?,?,'step',?,?,?,?,?,?,'[]','probation',?,?,?)",
            (screen.package, screen.skeleton_id, screen.simhash, intent_id, visit,
             anchor.model_dump_json() if anchor else None,
             action.model_dump_json(),
             postcondition.model_dump_json() if postcondition else None,
             json.dumps(required), json.dumps(forbidden),
             after.skeleton_id,
             self._next_version(screen, intent_id, visit), now, now))
        self.db.commit()

        self.note_transition(screen, after, action)
        row = self.db.execute("SELECT * FROM entry WHERE id=?",
                              (cursor.lastrowid,)).fetchone()
        return self._row_to_step(row)

    def _next_version(self, screen: Screen, intent_id: str, visit: int) -> int:
        row = self.db.execute(
            "SELECT MAX(version) AS v FROM entry WHERE app_key=? AND skeleton_id=? "
            "AND intent_id=? AND visit_ordinal=?",
            (screen.package, screen.skeleton_id, intent_id, visit)).fetchone()
        return (row["v"] or 0) + 1

    def note_transition(self, before: Screen, after: Screen,
                        action: AgentAction) -> None:
        if before.skeleton_id == after.skeleton_id:
            return
        self.db.execute(
            "INSERT INTO transition(app_key, from_skeleton, to_skeleton, "
            " action_sig, updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(app_key, from_skeleton, to_skeleton, action_sig) "
            "DO UPDATE SET n_seen = n_seen + 1, updated_at = excluded.updated_at",
            (before.package, before.skeleton_id, after.skeleton_id,
             action.signature(), time.time()))
        self.db.commit()

    # -- outcomes ----------------------------------------------------------

    def mark(self, entry: CachedStep, grade: str, run_id: str,
             reason: str = "", observed_successor: str = "") -> None:
        """Record how a replay went and re-classify the entry."""
        success = grade in ("success", "soft_fail")
        weight = 1.0 if grade == "success" else (0.5 if grade == "soft_fail" else 0.0)

        if success:
            entry.stats.n_success += weight
            entry.stats.consecutive_failures = 0
        else:
            entry.stats.n_failure += 1.0
            entry.stats.consecutive_failures += 1

        alts = list(entry.alt_successors)
        if grade == "soft_fail" and observed_successor and observed_successor not in alts:
            # Screens legitimately fan out. Remember the new destination rather
            # than treating it as a failure next time.
            alts.append(observed_successor)

        new_state = trust.classify(entry.stats)
        if new_state != entry.state:
            log.info("%s: %s -> %s", entry.describe(), entry.state, new_state)
        entry.state = new_state
        entry.alt_successors = alts

        now = time.time()
        self.db.execute(
            "UPDATE entry SET n_success=?, n_failure=?, consecutive_failures=?, "
            " state=?, alt_successors=?, updated_at=?, last_used_at=? WHERE id=?",
            (entry.stats.n_success, entry.stats.n_failure,
             entry.stats.consecutive_failures, entry.state, json.dumps(alts),
             now, now, entry.id))
        self.db.execute(
            "INSERT INTO entry_outcome(entry_id, run_id, grade, match_distance, "
            " anchor_score, reason, at) VALUES(?,?,?,?,?,?,?)",
            (entry.id, run_id, grade, entry.match_distance, entry.anchor_score,
             reason, now))
        self.db.commit()

    # -- runs --------------------------------------------------------------

    def begin_run(self, run_id: str, goal: str, intent_id: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO run(run_id, goal, intent_id, started_at) "
            "VALUES(?,?,?,?)", (run_id, goal, intent_id, time.time()))
        self.db.commit()

    def end_run(self, run_id: str, outcome: str, steps: int, llm_calls: int,
                cache_hits: int, usd: float) -> None:
        self.db.execute(
            "UPDATE run SET outcome=?, steps=?, llm_calls=?, cache_hits=?, usd=?, "
            "ended_at=? WHERE run_id=?",
            (outcome, steps, llm_calls, cache_hits, usd, time.time(), run_id))
        self.db.commit()

    # -- inspection and maintenance ----------------------------------------

    def entries(self, app_key: str = "", state: str = "") -> List[CachedStep]:
        sql = "SELECT * FROM entry WHERE 1=1"
        args: List[Any] = []
        if app_key:
            sql += " AND app_key=?"
            args.append(app_key)
        if state:
            sql += " AND state=?"
            args.append(state)
        sql += " ORDER BY app_key, skeleton_id, visit_ordinal, version"
        return [self._row_to_step(r) for r in self.db.execute(sql, args)]

    def get(self, entry_id: int) -> Optional[CachedStep]:
        row = self.db.execute("SELECT * FROM entry WHERE id=?", (entry_id,)).fetchone()
        return self._row_to_step(row) if row else None

    def forget(self, *, entry_id: int = 0, app_key: str = "",
               state: str = "") -> int:
        sql = "DELETE FROM entry WHERE 1=1"
        args: List[Any] = []
        if entry_id:
            sql += " AND id=?"
            args.append(entry_id)
        if app_key:
            sql += " AND app_key=?"
            args.append(app_key)
        if state:
            sql += " AND state=?"
            args.append(state)
        n = self.db.execute(sql, args).rowcount
        self.db.commit()
        return n

    def gc(self, max_age_days: float = 90.0) -> int:
        """Retire entries that are stale or have proven themselves wrong."""
        cutoff = time.time() - max_age_days * 86400
        n = self.db.execute(
            "DELETE FROM entry WHERE state='quarantined' AND updated_at < ?",
            (time.time() - 30 * 86400,)).rowcount
        n += self.db.execute(
            "DELETE FROM entry WHERE COALESCE(last_used_at, created_at) < ? "
            "AND n_success < 2", (cutoff,)).rowcount
        # Keep at most MAX_VERSIONS live versions per (screen, intent, visit).
        n += self.db.execute(
            "DELETE FROM entry WHERE id IN ("
            "  SELECT id FROM ("
            "    SELECT id, ROW_NUMBER() OVER ("
            "      PARTITION BY app_key, skeleton_id, intent_id, visit_ordinal "
            "      ORDER BY version DESC) AS rn FROM entry"
            "  ) WHERE rn > ?)", (trust.MAX_VERSIONS,)).rowcount
        self.db.commit()
        return n

    def stats_summary(self) -> Dict[str, Any]:
        counts = {row["state"]: row["n"] for row in self.db.execute(
            "SELECT state, COUNT(*) AS n FROM entry GROUP BY state")}
        apps = self.db.execute(
            "SELECT COUNT(DISTINCT app_key) AS n FROM entry").fetchone()["n"]
        return {"entries": sum(counts.values()), "by_state": counts, "apps": apps}

    # -- row mapping -------------------------------------------------------

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> CachedStep:
        anchor_json = row["anchor_json"]
        post_json = row["postcondition_json"]
        age_days = max(0.0, (time.time() - row["updated_at"]) / 86400.0)
        return CachedStep(
            id=row["id"],
            app_key=row["app_key"],
            skeleton_id=row["skeleton_id"],
            simhash=row["simhash"],
            intent_id=row["intent_id"],
            visit_ordinal=row["visit_ordinal"],
            scope=row["scope"],
            anchor=Anchor.model_validate_json(anchor_json) if anchor_json else None,
            action=AgentAction.model_validate_json(row["action_json"]),
            postcondition=(Postcondition.model_validate_json(post_json)
                           if post_json else None),
            required_tokens=json.loads(row["required_tokens"]),
            forbidden_tokens=json.loads(row["forbidden_tokens"]),
            next_skeleton_id=row["next_skeleton_id"] or "",
            alt_successors=json.loads(row["alt_successors"]),
            state=row["state"],
            version=row["version"],
            stats=trust.Stats(
                n_success=row["n_success"],
                n_failure=row["n_failure"],
                consecutive_failures=row["consecutive_failures"],
                age_days=age_days,
            ),
        )
