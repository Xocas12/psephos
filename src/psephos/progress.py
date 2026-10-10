"""Progress reporting for the long checks.

The integer-percentage null is a Monte Carlo over every unit, so on a national dataset one
check runs for seconds and a full audit for minutes. A silent wait is not merely unpleasant:
it invites someone to kill the run and try again with fewer replicates, and ``n_mc`` sets the
Monte Carlo p-value floor at ``1 / (n_mc + 1)``, against which the strong flag is defined. A
run abandoned for being quiet comes back as a run that flags differently.

So progress goes to stderr, never stdout: the report is stdout's, and a progress line written
into it would corrupt a redirected report.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import TextIO


class StderrProgress:
    """Report check progress on one rewritten stderr line.

    ``quiet`` returns a reporter that does nothing, so the caller does not branch. Output is
    rate-limited, because a block of replicates can finish in microseconds on a small dataset
    and the flicker would cost more than the work.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        min_interval: float = 0.2,
        enabled: bool = True,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval = min_interval
        self.enabled = enabled
        self._last = 0.0
        self._width = 0

    def for_check(self, label: str) -> Callable[[int, int], None] | None:
        """A ``progress(done, total)`` callable for one named check, or None when disabled.

        None rather than a no-op function so the check can skip the call altogether.
        """
        if not self.enabled:
            return None

        def report(done: int, total: int) -> None:
            now = time.monotonic()
            if done < total and now - self._last < self.min_interval:
                return
            self._last = now
            pct = 100.0 * done / total if total else 100.0
            line = f"  {label}: {done}/{total} replicates ({pct:.0f}%)"
            # Pad to erase a longer previous line rather than leaving its tail behind.
            self.stream.write("\r" + line.ljust(self._width))
            self.stream.flush()
            self._width = max(self._width, len(line))

        return report

    def done(self) -> None:
        """Clear the line, so the report that follows starts at column zero."""
        if self.enabled and self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
            self._width = 0
