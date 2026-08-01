"""How much to trust a cached step. Pure functions, no I/O.

A cache that assumes its own entries are good is a liability: one stale entry
replays a wrong tap forever. Three ideas keep it honest.

* **Wilson lower bound**, not a success ratio. A raw ratio says a 1-of-1 entry
  is 100% reliable. The Wilson lower bound says ~0.21, which is the correct
  amount of confidence to have in a single observation, so new entries stay on
  probation until they have earned their way out.
* **Time decay**, because app updates are the dominant source of staleness. A
  14-day half-life roughly matches release cadence.
* **Versioning, never overwriting.** A failing entry spawns a new version beside
  the old one. Blind overwriting is how a cache degrades into being worse than
  no cache at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

State = Literal["probation", "active", "trusted", "quarantined", "retired"]

HALF_LIFE_DAYS = 14.0
Z = 1.96  # 95%

PROBATION_MIN_OBS = 3
ACTIVE_WILSON = 0.50          # reached at 4 consecutive successes
# 0.70 rather than 0.75 so a clean run of 10 earns trust. At 0.75 the bound is
# not crossed until 13 consecutive successes, which in practice means almost
# nothing is ever promoted and the relaxed-threshold path is dead code.
TRUSTED_WILSON = 0.70
TRUSTED_MIN_SUCCESS = 8
QUARANTINE_CONSECUTIVE_FAILURES = 3
QUARANTINE_WILSON = 0.25
QUARANTINE_MIN_OBS = 4

MAX_VERSIONS = 3


def wilson_lower_bound(successes: float, failures: float, z: float = Z) -> float:
    """Lower bound of the 95% confidence interval on the success rate."""
    n = successes + failures
    if n <= 0:
        return 0.0
    phat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    return max(0.0, (centre - margin) / denominator)


def decay_factor(age_days: float, half_life: float = HALF_LIFE_DAYS) -> float:
    """Weight of an observation `age_days` old."""
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life)


@dataclass
class Stats:
    n_success: float = 0.0
    n_failure: float = 0.0
    consecutive_failures: int = 0
    age_days: float = 0.0

    def decayed(self) -> "Stats":
        f = decay_factor(self.age_days)
        return Stats(self.n_success * f, self.n_failure * f,
                     self.consecutive_failures, self.age_days)

    @property
    def observations(self) -> float:
        return self.n_success + self.n_failure

    def wilson(self) -> float:
        d = self.decayed()
        return wilson_lower_bound(d.n_success, d.n_failure)


def classify(stats: Stats) -> State:
    """Map observations onto a trust state."""
    if stats.consecutive_failures >= QUARANTINE_CONSECUTIVE_FAILURES:
        return "quarantined"
    wilson = stats.wilson()
    if stats.observations >= QUARANTINE_MIN_OBS and wilson < QUARANTINE_WILSON:
        return "quarantined"
    if stats.observations < PROBATION_MIN_OBS:
        return "probation"
    if wilson >= TRUSTED_WILSON and stats.n_success >= TRUSTED_MIN_SUCCESS:
        return "trusted"
    if wilson >= ACTIVE_WILSON:
        return "active"
    return "probation"


def may_replay(state: State) -> bool:
    return state in ("probation", "active", "trusted")


def must_verify(state: State) -> bool:
    """Probation entries are always verified; trusted ones still usually are."""
    return state != "trusted"


def anchor_threshold(state: State, strict: float, relaxed: float,
                     minor_deviation: bool = False) -> float:
    """How well an anchor must match before we act on it."""
    if minor_deviation or state == "probation":
        return strict
    return relaxed if state == "trusted" else strict


def shadow_audit_rate(state: State, probation: float, active: float,
                      trusted: float) -> float:
    """Fraction of cache hits that also ask the LLM, to measure agreement.

    This is what turns "we think the cache is right" into a number.
    """
    return {"probation": probation, "active": active,
            "trusted": trusted}.get(state, 0.0)
