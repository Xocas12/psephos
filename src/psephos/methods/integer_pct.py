"""Excess mass at integer percentages.

If someone writes down a target rather than a count, the target tends to be a round number.
So percentages computed from reported figures pile up on whole numbers more often than counting
noise allows. This is the strongest single check psephos implements.

Method
------
For each unit with denominator ``d`` and numerator ``k``, the percentage is ``100 k / d``. Count
the units whose percentage lies within ``tolerance`` of a whole number.

The null is that each unit's count is binomial noise around its own observed rate: draw
``k* ~ Binomial(d, k / d)``, recompute, and recount. Repeating that gives the distribution of
the integer count under "no rounding, same underlying rates, same precinct sizes", which is the
right null because it holds precinct size fixed. A null that ignored size would flag any
dataset with many small precincts.

This one does not, and small precincts cannot fake a signal under it. The arithmetic that puts a
small or round-sized precinct on a whole number is in the null as well as in the data: a
100-voter unit lands on one with probability 1 in both, a 200-voter unit with probability one
half in both. Averaged over turnout spread evenly from 40 to 80 per cent, an honest unit of any
size from 20 to 499 adds between -0.02 and +0.01 of a unit to the expected excess. Where the
plug-in null does misfit small units, at turnout close to 100 per cent as in some institutional
precincts, it predicts more whole numbers than honest data produce (up to 0.14 of a unit each,
averaged over turnout from 90 to 100 per cent), so those units pull the statistic down. Small
precincts can hide rounding; they do not manufacture it.

What the size threshold controls is power. Below about 250 voters the count nearest a
whole-number percentage is usually not on one, so rounding there barely moves the count: at 55
per cent turnout, rounding a 100-249 voter precinct raises its chance of landing on a whole
number from about 0.11 to 0.18, and a 20-99 voter one not at all, against 0.10 to 1.0 at 1,000
voters and above. Including those units mostly dilutes the test.

The threshold sweep therefore says where in the size range an excess sits, not whether it is
rounding. An excess that fades as the threshold rises is an excess in the smaller precincts the
higher thresholds drop, which is where rounding is often alleged; it is still an excess over a
null that already holds their size fixed.

The null is drawn one replicate at a time, and vectorising that loop is the obvious thing to try
and does not pay. 95 per cent of a check is inside NumPy's per-unit binomial sampling, which
happens in C either way, and the Python loop is 0.01 per cent, so there is almost nothing there
to win. Drawing a block of replicates as one two-dimensional array was measured and its timing
difference does not survive the run-to-run noise on the machine it was measured on; what it does
cost is working memory, 96 MB against 1.5 MB at 64 replicates and about 760 MB for a national
run. That is why the loop stays. Both measurements, and a benchmark that reproduces them, are in
docs/performance.md. The progress callback stays for the other reason: minutes of silence is the
thing that actually makes someone rerun this with fewer replicates.

Reference: this is the estimator described by Kobak, Shpilkin and Pshenichnikov, "Integer
percentages as electoral falsification fingerprints", Annals of Applied Statistics 10(1), 2016.
The implementation here follows the description of the method; it has not been checked line by
line against the authors' own code, and issue #6 tracks that validation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from psephos._types import Finding, Flag
from psephos.interval import DEFAULT_N_BOOT, bootstrap_ci
from psephos.size import size_warning

#: Below this many registered voters, rounding barely changes how often a unit lands on a
#: whole-number percentage, so those units add noise and almost no power (see the module
#: docstring). It is a power threshold, not a guard against false alarms: the null already holds
#: each unit's size fixed. Units below it are excluded and counted.
DEFAULT_MIN_DENOMINATOR = 250

#: Innocent explanations printed beside a flagged finding. Each one has to be a mechanism that
#: puts the reported count itself on a whole-number percentage, because only that can raise the
#: observed count above a null that holds every unit's size fixed. Precinct size cannot: the
#: arithmetic that puts a small or round-sized precinct on a whole number is in the null too.
#: tests/test_small_precincts.py simulates each mechanism and checks it raises the count.
CONFOUNDS = [
    "Counts that were estimated or rounded rather than tallied, for example a figure reported "
    "to the nearest ten. Where the base is also a round number, such as 500 or 1,000, every "
    "such count is a whole-number percentage. A rounded registered count on its own does not "
    "do this, since the null takes the base as given.",
    "A count back-calculated from a published percentage that had already been rounded, by the "
    "commission, by whoever compiled the file, or by an earlier processing step, so that it "
    "lands on a whole number by construction.",
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
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = 0.95,
    progress: Callable[[int, int], None] | None = None,
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
    n_boot : int
        Bootstrap resamples for the interval on the effect, ``0`` to skip it. The resample is
        over units within size bands and costs one recount each, not a second Monte Carlo; see
        :mod:`psephos.interval`.
    ci_level : float
        Coverage of that interval.
    progress : callable, optional
        Called as ``progress(replicates_done, n_mc)`` while the null is drawn. A national
        dataset takes tens of seconds for one check and minutes for an audit, and a silent wait
        invites someone to kill the run and retry with fewer replicates. ``n_mc`` sets the Monte
        Carlo p-value floor at ``1 / (n_mc + 1)``, which is what the strong flag is defined
        against, so a run abandoned for being quiet comes back as a run that flags differently.
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
    # How often each unit individually landed on a whole number across the replicates, which is
    # the per-unit null probability the bootstrap needs. Accumulating it here is one add per
    # replicate; deriving it later would mean re-running the Monte Carlo for every resample.
    unit_hits = np.zeros(n_used, dtype=np.int64)
    # Reported every 10 replicates rather than every one: the callback rate-limits by time, but
    # on a small dataset even the call overhead is worth not paying 500 times.
    report_every = max(1, n_mc // 50)
    for i in range(n_mc):
        pct_draw = 100.0 * rng.binomial(d_int, p) / d
        hit = np.abs(pct_draw - np.round(pct_draw)) <= tolerance
        null[i] = float(hit.sum())
        unit_hits += hit
        if progress is not None and ((i + 1) % report_every == 0 or i + 1 == n_mc):
            progress(i + 1, n_mc)

    mean = float(null.mean())
    sd = float(null.std(ddof=1))
    excess = observed - mean
    z = excess / sd if sd > 0 else float("nan")
    # Add-one Monte Carlo p-value: never reports exactly zero, which would overstate certainty.
    mc_p = float((1 + np.sum(null >= observed)) / (n_mc + 1))
    # Excess as a share of the units tested, which is the number a reader can interpret.
    effect = excess / n_used

    # The effect is a mean of per-unit contributions: unit i either landed on a whole number or
    # did not, against the probability q_i that it would under the null. Writing it that way is
    # what makes the bootstrap affordable — each resample is a mean over an index array rather
    # than another Monte Carlo — and it is the same number, because
    # sum_i(hit_i - q_i) / n = (observed - null_mean) / n.
    hit = np.abs(pct - np.round(pct)) <= tolerance
    contribution = hit.astype(float) - unit_hits / n_mc
    ci_low, ci_high, ci_details = bootstrap_ci(
        lambda idx: float(contribution[idx].mean()),
        n_used,
        estimate=float(effect),
        sizes=d,
        n_boot=n_boot,
        level=ci_level,
        # Derived from the check's own seed so the interval is reproducible with the finding,
        # and offset so the resamples are not the null draws over again.
        seed=None if seed is None else seed + 101,
    )

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
    interval = (
        ""
        if ci_low is None or ci_high is None
        # The coverage is stated from ci_level, not hardcoded: --ci-level is plumbed through, and
        # a title claiming 95 per cent under --ci-level 0.8 would contradict the report's own
        # EFFECT INTERVALS section two sections below it.
        else (f", {100 * ci_level:g} per cent interval [{100 * ci_low:+.2f}, {100 * ci_high:+.2f}]")
    )
    title = (
        f"{observed} of {n_used} units land on a whole-number {label} percentage; "
        f"binomial noise predicts about {mean:.0f}. "
        f"Excess {excess:+.0f} units ({pct_excess:+.2f} per cent of those tested"
        f"{interval})."
    )

    return Finding(
        check="integer_percentage",
        title=title,
        flag=flag,
        statistic=float(z),
        pvalue=mc_p,
        effect=float(effect),
        ci_low=ci_low,
        ci_high=ci_high,
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
            **ci_details,
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

    The sweep locates an excess by precinct size. It is not what separates rounding from
    arithmetic; the null at every threshold already does that, by holding each unit's size
    fixed. An excess that fades as the threshold rises sits in the smaller precincts the higher
    thresholds drop, and it is still an excess there; one that holds at the strictest threshold
    sits in the large ones. Reporting the sweep rather than one threshold is what stops the
    parameter being tuned until the answer is interesting, so the audit runs this rather than a
    single call.
    """
    out = []
    for t in thresholds:
        kw = dict(kwargs)
        kw["min_denominator"] = t
        kw["slice_name"] = f"min_denominator={t}"
        out.append(integer_excess(numerator, denominator, **kw))  # type: ignore[arg-type]
    return out
