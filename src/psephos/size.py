"""Precinct-size stratification, which is not optional.

A polling station with 100 registered voters can honestly report exactly 70 per cent turnout,
because with 100 voters every attainable turnout IS a whole percentage. A station with 2,500
voters almost never lands on one by chance. So a test for "too many round percentages" that
ignores station size is measuring arithmetic, and it will indict small rural and institutional
precincts in every country on earth.

Every psephos check that involves a percentage runs stratified by size, and the report shows
the strata. This module decides the strata.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

#: Default boundaries for size bands, in registered voters. Chosen so the first band is where
#: integer percentages are common by arithmetic alone and the last is where they are rare.
DEFAULT_EDGES: tuple[float, ...] = (0, 100, 250, 500, 1000, 2000, np.inf)


@dataclass(frozen=True)
class Stratum:
    """One size band."""

    name: str
    lo: float
    hi: float
    mask: np.ndarray

    @property
    def n(self) -> int:
        return int(self.mask.sum())


def integer_reachability(denominator: float, tolerance: float = 0.05) -> float:
    """Fraction of attainable percentages that land within ``tolerance`` of a whole number.

    With ``d`` registered voters there are ``d + 1`` attainable counts and so at most ``d + 1``
    attainable percentages. This returns the share of them that are within ``tolerance`` of an
    integer, which is the probability of a round percentage under the crudest possible null of
    a uniformly chosen count.

    It is a diagnostic, not a test. Its purpose is to make the size trap a number you can look
    at rather than a warning you can ignore: at ``d = 100`` it is 1.0, and it falls away
    quickly as ``d`` grows.
    """
    d = float(denominator)
    if not np.isfinite(d) or d <= 0:
        return float("nan")
    counts = np.arange(int(d) + 1, dtype=float)
    pct = 100.0 * counts / d
    return float(np.mean(np.abs(pct - np.round(pct)) <= tolerance))


def stratify(
    denominators: np.ndarray,
    edges: tuple[float, ...] = DEFAULT_EDGES,
) -> list[Stratum]:
    """Split units into size bands. Units with a non-positive or missing denominator go
    nowhere and are simply absent from every stratum, which the caller can detect by comparing
    the stratum sizes against the total."""
    den = np.asarray(denominators, dtype=float)
    out: list[Stratum] = []
    for lo, hi in pairwise(edges):
        mask = np.isfinite(den) & (den > 0) & (den >= lo) & (den < hi)
        hi_label = "and up" if not np.isfinite(hi) else f"to {int(hi) - 1}"
        name = f"registered {int(lo)} {hi_label}"
        out.append(Stratum(name=name, lo=float(lo), hi=float(hi), mask=mask))
    return out


def size_warning(denominators: np.ndarray, min_denominator: int) -> str | None:
    """A sentence for the report when many units sit below the analysis threshold, or None."""
    den = np.asarray(denominators, dtype=float)
    usable = np.isfinite(den) & (den > 0)
    if not usable.any():
        return "no unit has a usable denominator"
    small = int((usable & (den < min_denominator)).sum())
    total = int(usable.sum())
    if small == 0:
        return None
    pct = 100.0 * small / total
    return (
        f"{small} of {total} units ({pct:.1f} per cent) have fewer than {min_denominator} "
        "registered voters and are excluded from percentage-based checks, because at that "
        "size round percentages arise by arithmetic."
    )
