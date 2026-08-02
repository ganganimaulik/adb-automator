"""Run tracking, screen corpus, transitions, and dead-end storage."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .fingerprint import mask_goal, normalize_verb_polarity
from .screen import Screen

log = logging.getLogger("adbagent.memory")

SCHEMA_VERSION = 3

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

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
"""

# -- Schema migration from v2 to v3 -----------------------------------------

_MIGRATE_V2_TO_V3 = [
    "DROP TABLE IF EXISTS entry_outcome",
    "DROP TABLE IF EXISTS entry", 
    "DROP TABLE IF EXISTS app_tuning",
]

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


def _to_sqlite_int(val: int) -> int:
    """Ensure 64-bit integer fits into SQLite signed 64-bit INTEGER bounds."""
    if val >= (1 << 63):
        return val - (1 << 64)
    return val


class Memory:
    """SQLite-backed store of learned steps."""

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

        # -- Detect existing schema version and migrate --------------------
        row = self.db.execute(
            "SELECT v FROM meta WHERE k='schema_version'").fetchone()
        old_version = int(row["v"]) if row else 0

        if old_version < 3:
            for stmt in _MIGRATE_V2_TO_V3:
                try:
                    self.db.execute(stmt)
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

    def note_transition(self, before: Screen, after: Screen,
                        action: Any) -> None:
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
        """Remember that an action on this screen is a dead end.

        Future lookups will automatically ban this action signature for 24 hours.
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

    def _dead_end_sigs(self, app_key: str, skeleton_id: str,
                       intent_id: str) -> set:
        """Active dead-end action signatures for a screen."""
        now = time.time()
        rows = self.db.execute(
            "SELECT action_sig FROM dead_end WHERE app_key=? AND skeleton_id=? "
            "AND intent_id=? AND expires_at > ?",
            (app_key, skeleton_id, intent_id, now)).fetchall()
        return {r["action_sig"] for r in rows}
