"""Run tracking, dead ends and located points -- what the agent learns across runs.

Only what is read lives here. Anything written and never read was removed rather
than kept in case it became useful: a per-step SQLite commit feeding a table with
no reader is a cost with no upside, and it reads to the next person like a
feature.

Two kinds of cross-run memory, and they are keyed differently on purpose.

The dead-end table records that an action on a screen led nowhere, so the agent
does not rediscover the same dud control in every run. It is keyed by intent as
well as by screen, because "the Wi-Fi row does nothing" can be true of one goal
and false of another.

The locate table records where a *named* control sits on a screen, so a `tap_at`
that named one does not pay for the same vision call twice. It is deliberately
NOT keyed by intent: where a control is on a layout is a fact about the layout,
and the goal being pursued has no bearing on it.

Both expire. An app that was broken last night may be fixed this morning, and an
app that updates overnight may have moved the control.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional

from .config import Config
from .fingerprint import mask_goal, normalize_verb_polarity
from .screen import Screen

log = logging.getLogger("adbagent.memory")

SCHEMA_VERSION = 5

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    run_id      TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    intent_id   TEXT NOT NULL,
    outcome     TEXT,
    steps       INTEGER NOT NULL DEFAULT 0,
    llm_calls   INTEGER NOT NULL DEFAULT 0,
    usd         REAL NOT NULL DEFAULT 0,
    started_at  REAL NOT NULL,
    ended_at    REAL
);

CREATE TABLE IF NOT EXISTS dead_end (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    app_key     TEXT NOT NULL,
    skeleton_id TEXT NOT NULL,
    intent_id   TEXT NOT NULL,
    action_sig  TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    UNIQUE(app_key, skeleton_id, intent_id, action_sig)
);

CREATE INDEX IF NOT EXISTS dead_end_lookup
    ON dead_end(app_key, skeleton_id, intent_id);

CREATE TABLE IF NOT EXISTS locate (
    app_key     TEXT NOT NULL,
    skeleton_id TEXT NOT NULL,
    description TEXT NOT NULL,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    PRIMARY KEY(app_key, skeleton_id, description)
);
"""

#: Tables from schema versions that no longer have a reader. Dropped rather than
#: left in place: an unused table is indistinguishable from a broken feature to
#: whoever opens the database next.
_DROPPED_TABLES = (
    # v2: the fingerprint cache and its trust scoring, removed in 78f50a7.
    "entry_outcome", "entry", "app_tuning",
    # v3: written on every step and every transition, read by nothing.
    # `screen_seen` fed an IDF that no caller ever computed; `transition` had no
    # reader at all.
    "screen_seen", "transition",
)


def intent_key(goal: str) -> str:
    """Normalised goal, so trivial rewording still hits the cache.

    Incorporates verb polarity so that opposite-intent goals (e.g. "turn on
    WiFi" vs "turn off WiFi") produce different keys.
    """
    import hashlib
    polarity = normalize_verb_polarity(goal)
    normalised = " ".join(mask_goal(goal).split())
    combined = f"{polarity}\x1f{normalised}" if polarity else normalised
    return hashlib.blake2b(combined.encode("utf-8"), digest_size=8).hexdigest()


class Memory:
    """SQLite-backed store of what the agent learned across runs."""

    def __init__(self, cfg: Config, path: Optional[Path] = None):
        self.cfg = cfg
        self.path = path or cfg.db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables, migrate from older schema versions."""
        self.db.executescript(DDL)

        # -- Drop what older versions wrote and nothing reads ---------------
        row = self.db.execute(
            "SELECT v FROM meta WHERE k='schema_version'").fetchone()
        old_version = int(row["v"]) if row else 0

        if old_version < SCHEMA_VERSION:
            for table in _DROPPED_TABLES:
                try:
                    self.db.execute(f"DROP TABLE IF EXISTS {table}")
                except sqlite3.OperationalError:
                    pass

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

    # -- runs --------------------------------------------------------------

    def begin_run(self, run_id: str, goal: str, intent_id: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO run(run_id, goal, intent_id, started_at) "
            "VALUES(?,?,?,?)", (run_id, goal, intent_id, time.time()))
        self.db.commit()

    def end_run(self, run_id: str, outcome: str, steps: int, llm_calls: int,
                usd: float) -> None:
        self.db.execute(
            "UPDATE run SET outcome=?, steps=?, llm_calls=?, usd=?, "
            "ended_at=? WHERE run_id=?",
            (outcome, steps, llm_calls, usd, time.time(), run_id))
        self.db.commit()

    # -- dead ends ---------------------------------------------------------

    _DEAD_END_TTL_S = 86400  # 24 hours

    def record_dead_end(self, screen: Screen, intent_id: str,
                        action_sig: str, reason: str) -> None:
        """Remember that an action on this screen led nowhere.

        Read back by `dead_ends` for 24 hours, in this run and in any later one.
        """
        now = time.time()
        self.db.execute(
            "INSERT INTO dead_end(app_key, skeleton_id, intent_id, action_sig, "
            " reason, created_at, expires_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(app_key, skeleton_id, intent_id, action_sig) "
            "DO UPDATE SET reason=excluded.reason, created_at=excluded.created_at, "
            " expires_at=excluded.expires_at",
            (screen.package, screen.skeleton_id, intent_id, action_sig,
             reason, now, now + self._DEAD_END_TTL_S))
        self.db.commit()

    def dead_ends(self, screen: Screen, intent_id: str) -> Dict[str, str]:
        """Action signature -> why it led nowhere, for this screen and intent.

        Keyed by intent as well as by screen because "this row does nothing" can
        be true of one goal and false of another, and time-limited because an app
        that was broken last night may be fixed this morning.
        """
        rows = self.db.execute(
            "SELECT action_sig, reason FROM dead_end WHERE app_key=? "
            "AND skeleton_id=? AND intent_id=? AND expires_at > ?",
            (screen.package, screen.skeleton_id, intent_id,
             time.time())).fetchall()
        return {row["action_sig"]: row["reason"] for row in rows}

    # -- located points ----------------------------------------------------

    #: Shorter than the dead-end TTL. A dead end stays true until the app is
    #: fixed; a located point stops being true the moment the app updates and
    #: moves the control, and a stale point taps the wrong thing rather than
    #: merely wasting a turn. `forget_locate` is the real protection -- this is
    #: the backstop for a layout that changed without any tap proving it.
    _LOCATE_TTL_S = 43200  # 12 hours

    @staticmethod
    def _locate_key(description: str) -> str:
        """How a control's name is matched against an earlier one.

        Case and inner whitespace only. Nothing cleverer: the description is
        the model's own words for the control, and two spellings that differ by
        more than that ("the send pill", "Send Priority Like") may well be two
        different controls, so they get their own rows and their own locates.
        """
        return " ".join(description.lower().split())

    def recall_locate(self, screen: Screen,
                      description: str) -> Optional[tuple]:
        """Where this control was last found on this screen, as (x, y).

        Measured over the 169 runs in ``runs/``: 577 `tap_at` actions named a
        control, and they resolve to 94 distinct (skeleton, name) pairs. 211 of
        them (37%) repeat a pair already located earlier *in the same run*, and
        483 (84%) repeat one located in some earlier run -- "send priority like"
        on one Hinge skeleton was located 134 separate times, each a screenshot
        and a vision call for a point the harness had already been told.

        Not keyed by intent, unlike `dead_ends`: where a control sits is a
        property of the layout, and the goal has no bearing on it.
        """
        row = self.db.execute(
            "SELECT x, y FROM locate WHERE app_key=? AND skeleton_id=? "
            "AND description=? AND expires_at > ?",
            (screen.package, screen.skeleton_id,
             self._locate_key(description), time.time())).fetchone()
        return (row["x"], row["y"]) if row else None

    def record_locate(self, screen: Screen, description: str,
                      x: float, y: float) -> None:
        """Remember where a vision call placed this control."""
        now = time.time()
        self.db.execute(
            "INSERT INTO locate(app_key, skeleton_id, description, x, y, "
            " created_at, expires_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(app_key, skeleton_id, description) "
            "DO UPDATE SET x=excluded.x, y=excluded.y, "
            " created_at=excluded.created_at, expires_at=excluded.expires_at",
            (screen.package, screen.skeleton_id, self._locate_key(description),
             x, y, now, now + self._LOCATE_TTL_S))
        self.db.commit()

    def forget_locate(self, screen: Screen, description: str) -> None:
        """Drop a cached point that turned out to be wrong.

        Called when a tap at the cached point changed nothing. Without this the
        cache would be strictly worse than paying for the locate: a vision call
        that misses costs one turn, while a cached miss would cost every
        remaining turn that named the same control.
        """
        self.db.execute(
            "DELETE FROM locate WHERE app_key=? AND skeleton_id=? "
            "AND description=?",
            (screen.package, screen.skeleton_id,
             self._locate_key(description)))
        self.db.commit()
