"""Excess mass at integer percentages.

If someone writes down a target rather than a count, the target tends to be a round number.
So percentages computed from reported figures pile up on whole numbers more often than counting
noise allows. This is the strongest single check psephos implements, and it is also the one
most easily faked by small precincts, which is why the size threshold is not optional.

Method
------
For each unit with denominator ``d`` and numerator ``k``, the percentage is ``100 k / d``. Count
the units whose percentage lies within ``tolerance`` of a whole number.

The null is that each unit's count is binomial noise around its own observed rate: draw
``k* ~ Binomial(d, k / d)``, recompute, and recount. Repeating that gives the distribution of
the integer count under "no rounding, same underlying rates, same precinct sizes", which is the
right null because it holds precinct size fixed. A null that ignored size would flag any
dataset with many small precincts.

Reference: this is the estimator described by Kobak, Shpilkin and Pshenichnikov, "Integer
percentages as electoral falsification fingerprints", Annals of Applied Statistics 10(1), 2016.
The implementation here follows the description of the method; it has not been checked line by
line against the authors' own code, and issue #6 tracks that validation.
"""

from __future__ import annotations

import numpy as np

from psephos._types import Finding, Flag
from psephos.size import size_warning

#: Below this many registered voters, whole-number percentages arise by arithmetic often enough
#: that the test cannot separate them from rounding. Units below it are excluded and counted.
DEFAULT_MIN_DENOMINATOR = 250

CONFOUNDS = [
    "Small precincts, where whole-number percentages are common by arithmetic alone. The "
    "min_denominator threshold controls this and the result should be checked across a range "
    "of thresholds before it is believed.",
    "Institutional precincts such as hospitals, barracks, ships and prisons, which are small "
    "and whose electorates are often round numbers to begin with.",
    "A reported figure that was genuinely computed as a percentage of a target, for example a "
    "quota, without any misreporting of the count.",
]


def _integer_count(pct: np.ndarray, tolerance: float) -> int:
    return int(np.sum(np.abs(pct - np.round(pct)) <= tolerance))


def integer_excess(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    tolerance: float = 0.05,
    min_denominator: int = DEFAULT_MIN_DENOMINATOR,
    n_mc: int = 500,
    seed: int | None = None,
    slice_name: str = "all",
    label: str = "value",
) -> Finding:
    """Test for excess mass at whole-number percentages.

    Parameters
    ----------
    numerator, denominator : array-like
        Counts and their bases, one per unit.
    tolerance : float
        How close to a whole number counts as landing on it, in percentage points.
    min_denominator : int
        Units with a smaller base are excluded. See :data:`DEFAULT_MIN_DENOMINATOR`.
    n_mc : int
        Monte Carlo replicates for the null. 500 is enough for a p-value near 0.01.
    seed : int, optional
        Makes the result reproducible.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.shape != den.shape:
        raise ValueError(f"numerator and denominator differ in shape: {num.shape} vs {den.shape}")

    usable = (
        np.isfinite(num) & np.isfinite(den) & (den >= min_denominator) & (num >= 0) & (num <= den)
    )
    n_used = int(usable.sum())
    n_excluded = int(usable.size - n_used)

    if n_used < 100:
        return Finding(
            check="integer_percentage",
            title=(
                f"Only {n_used} units have at least {min_denominator} in the denominator, "
                "which is too few for this test to say anything."
            ),
            flag=Flag.UNDERPOWERED,
            n_used=n_used,
            n_excluded=n_excluded,
            slice_name=slice_name,
            details={"min_denominator": min_denominator, "label": label},
        )

    k = num[usable]
    d = den[usable]
    pct = 100.0 * k / d
    observed = _integer_count(pct, tolerance)

    rng = np.random.default_rng(seed)
    p = np.clip(k / d, 0.0, 1.0)
    d_int = np.rint(d).astype(np.int64)
    null = np.empty(n_mc, dtype=float)
    for i in range(n_mc):
        draw = rng.binomial(d_int, p)
        null[i] = _integer_count(100.0 * draw / d, tolerance)

    mean = float(null.mean())
    sd = float(null.std(ddof=1))
    excess = observed - mean
    z = excess / sd if sd > 0 else float("nan")
    # Add-one Monte Carlo p-value: never reports exactly zero, which would overstate certainty.
    mc_p = float((1 + np.sum(null >= observed)) / (n_mc + 1))
    # Excess as a share of the units tested, which is the number a reader can interpret.
    effect = excess / n_used

    # The Monte Carlo p-value cannot go below 1 / (n_mc + 1), so a fixed cutoff such as 0.001
    # would be unreachable at the default n_mc and the strong flag would never fire. Compare
    # against the attainable floor instead, and require a large standardised departure too, so
    # that "no replicate came close" alone is not enough.
    p_floor = 1.0 / (n_mc + 1)
    if not np.isfinite(z):
        flag = Flag.UNDERPOWERED
    elif mc_p <= 2 * p_floor and z >= 5:
        flag = Flag.STRONG
    elif mc_p <= 0.05:
        flag = Flag.NOTABLE
    else:
        flag = Flag.OK

    pct_excess = 100.0 * effect
    title = (
        f"{observed} of {n_used} units land on a whole-number {label} percentage; "
        f"binomial noise predicts about {mean:.0f}. "
        f"Excess {excess:+.0f} units ({pct_excess:+.2f} per cent of those tested)."
    )

    return Finding(
        check="integer_percentage",
        title=title,
        flag=flag,
        statistic=float(z),
        pvalue=mc_p,
        effect=float(effect),
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "observed": observed,
            "null_mean": mean,
            "null_sd": sd,
            "excess": float(excess),
            "tolerance": tolerance,
            "min_denominator": min_denominator,
            "n_mc": n_mc,
            "mc_pvalue_floor": 1.0 / (n_mc + 1),
            "seed": seed,
            "label": label,
            "excluded_because": (
                f"denominator below {min_denominator}, missing, or numerator outside [0, denominator]"
            ),
            "size_note": size_warning(den, min_denominator),
        },
    )


def integer_excess_by_threshold(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    thresholds: tuple[int, ...] = (100, 250, 500, 1000),
    **kwargs: object,
) -> list[Finding]:
    """Run the test at several size thresholds.

    A real rounding signal survives raising the threshold; an artefact of small precincts dies.
    Reporting the sweep rather than one threshold is what stops the parameter being tuned until
    the answer is interesting, so the audit runs this rather than a single call.
    """
    out = []
    for t in thresholds:
        kw = dict(kwargs)
        kw["min_denominator"] = t
        kw["slice_name"] = f"min_denominator={t}"
        out.append(integer_excess(numerator, denominator, **kw))  # type: ignore[arg-type]
    return out
