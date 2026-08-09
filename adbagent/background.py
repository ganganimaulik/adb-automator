"""One thing: start slow work now, collect it later.

Both of the agent's expensive waits are round trips to something else -- the
model over the internet, the phone over adb -- and for most of the run they were
strictly alternating: talk to the model, then talk to the phone, then talk to the
model. Anything whose *input* is already complete before the next wait begins can
overlap it instead, and the step costs the slower of the two rather than the sum.

It lives in its own module because both callers are now peers: `llm` overlaps a
per-item vision read with the gesture that follows it, and `device` overlaps the
two independent adb round trips an observation makes. Neither should have to
import the other to get a thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

log = logging.getLogger("adbagent.background")


class Prefetch:
    """A call started now and collected later, so it overlaps other work.

    A failure is swallowed and reported through `result`'s default: a prefetched
    call is an optimisation, and losing one must degrade its answer rather than
    abort the work it was overlapping. Streaming callbacks are deliberately not
    plumbed through -- two threads writing to the same live terminal panel
    interleave into nonsense.
    """

    def __init__(self, fn: Callable[[], Any]):
        self._value: Any = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, args=(fn,),
                                        daemon=True, name="adbagent-prefetch")
        self._thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            self._value = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced via result()
            self._error = exc

    @property
    def failed(self) -> bool:
        return self._error is not None

    def result(self, default: Any = "", timeout: Optional[float] = None) -> Any:
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("prefetched call did not finish within %.1fs", timeout)
            return default
        if self._error is not None:
            log.warning("prefetched call failed: %s", self._error)
            return default
        return default if self._value is None else self._value
