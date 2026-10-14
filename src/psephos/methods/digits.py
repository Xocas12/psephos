"""Digit tests on vote counts.

Last digits of genuine vote counts should be close to uniform, because the last digit of a
count in the hundreds or thousands is not something a real process controls. Numbers invented
by people are not uniform: people under-produce repeated digits such as 33 and over-produce
adjacent ones such as 34.

Reference: Beber and Scacco, "What the Numbers Say: A Digit-Based Test for Election Fraud",
Political Analysis 20(2), 2012. The uniform null and the chi-square test below follow the
standard construction. The DIRECTION of the human-invention signature, that repeated pairs are
under-produced, is reported as a descriptive detail rather than as a test, because psephos has
not validated the direction against the paper; issue #7 tracks that.

The cut. A count below about a hundred has a last digit that is not free, so both checks test
only counts of at least ``min_count``. That cut is not innocent. Keeping the counts at or above
a value makes their last digits uneven by themselves wherever counts are dense at that value:
with the cut at 100, digit 0 collects 100, 110, 120 and so on and digit 9 collects 109, 119,
129, so where counts are falling across each ten the low endings get more of them than the high
ones. An honest minor party whose counts straddle 100 has exactly that shape. The departure is
fixed while the chi-square noise shrinks with the number of precincts, so a test against an
exact uniform flags it more often the more precincts there are. On the clean generator's
smallest party it fired in 25 of 200 elections at 4,000 precincts and 21 of 40 at 20,000, and
on a minor party concentrated just above 100 in 90 of 200 and 40 of 40, most of those strong
(issue #25; tests/test_digit_calibration.py has the elections). Moving ``min_count`` only
moves the cut to another dense point.

So neither check compares against an exact uniform. The expected share of each ending is the
one a smooth count density produces once it is cut at ``min_count``, which
:func:`expected_shares_after_cut` derives. It depends only on how many counts sit in the block
of values just below the cut and the block just above it, ten values for the last digit and a
hundred for the last two, and with no counts near the cut it is exactly uniform. Block totals
do not depend on the last digit when the cut is a multiple of the block, so inventing last
digits does not move the expected shares at all.

On Benford's law: the first-digit test is deliberately NOT implemented here. Precinct vote
counts do not span enough orders of magnitude for Benford to hold under honest conditions, so
it produces confident nonsense on exactly this kind of data. docs/why_no_benford.md records
the reasoning, with a demonstration on honest data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from psephos._types import Finding, Flag
from psephos.interval import DEFAULT_N_BOOT, bootstrap_ci, expected_tvd_under_uniform

#: Counts below this have too few digits for a free last digit.
MIN_COUNT_FOR_LAST_DIGIT = 100

CONFOUNDS_LAST_DIGIT = [
    "Counts dense at the min_count cut. Leaving out counts below min_count, which is needed "
    "because a small count's last digit is not free, makes the last digits uneven by "
    "themselves wherever many counts sit near the cut: the endings just above it collect more "
    "counts than those furthest from it. The expected shares allow for that, estimated from "
    "the counts just either side of the cut on the assumption that the count density is "
    "smooth there, so counts that change steeply within a few votes of the cut can still "
    "depart from them without any number being invented.",
    "Vote counts that are systematically rounded during aggregation or transcription, which "
    "produces the same non-uniformity without anyone inventing a number.",
    "Optical-character recognition or transcription error, if the data came from scans, which "
    "distorts digit frequencies on its own.",
]


def expected_shares_after_cut(
    counts: np.ndarray, min_count: int, period: int = 10
) -> tuple[np.ndarray, int, int]:
    """Expected share of each ending among the counts of at least ``min_count``.

    ``period`` is 10 for the last digit and 100 for the last two. Returns the shares, indexed
    by ending, and the two block totals they were derived from: the number of counts in the
    ``period`` values just below the cut and in the ``period`` values from the cut up.

    The null is that the count density is smooth and nothing else favours an ending. Split the
    counts at or above the cut into blocks of ``period`` consecutive values starting at the cut,
    and let ``D_j`` be the number of counts in block j, with ``D_-1`` the block just below the
    cut. A quadratic density through blocks j-1, j and j+1 puts, at position p of block j
    (p = 0 at the block's first value, centre c = (period - 1) / 2),

        D_j / period + b_j (p - c) + g_j ((p - c)^2 - (period^2 - 1) / 12)

    with ``b_j = (D_j+1 - D_j-1) / (2 period^2)`` and
    ``g_j = (D_j+1 - 2 D_j + D_j-1) / (2 period^3)``. Every value in position p of any block has
    the same ending, so the expected number of counts with that ending is a sum over blocks,
    and both sums telescope: the slopes sum to ``-(D_0 + D_-1) / (2 period^2)`` and the
    curvatures to ``(D_-1 - D_0) / (2 period^3)``. Only the two blocks either side of the cut
    are left.

    Checked against digit distributions computed exactly from known densities. For lognormal
    counts like the clean generator's parties and the binomial mixture of the near-the-cut test
    generator, the shares agree with the exact ones to within about 0.0002 at a cut of 100 and
    0.0006 at 105, against departures from uniform of 0.007 to 0.013. The error grows as the
    density narrows around the cut: a normal density with a standard deviation of 15 votes,
    peaking just above the cut, is off by up to 0.0012, and one of 7 votes by 0.003 to 0.005,
    a third of the correction, because a quadratic over blocks of ten cannot follow it.
    Counting noise alone spreads a minor party's count of about a hundred by ten either side,
    and differing precinct sizes spread it further.
    """
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x)]
    n_used = int((x >= min_count).sum())
    below = int(((x >= min_count - period) & (x < min_count)).sum())
    first = int(((x >= min_count) & (x < min_count + period)).sum())
    if n_used == 0:
        return np.full(period, 1.0 / period), below, first

    centre = (period - 1) / 2.0
    # Position of each ending within a block that starts at the cut, measured from the centre.
    offset = (np.arange(period) - min_count) % period - centre
    slope = -(first + below) / (2.0 * period**2)
    curve = (below - first) / (2.0 * period**3)
    expected = n_used / period + slope * offset + curve * (offset**2 - (period**2 - 1) / 12.0)
    return expected / n_used, below, first


def _too_close_to_the_cut(
    check: str,
    label: str,
    min_count: int,
    shares: np.ndarray,
    n_used: int,
    n_excluded: int,
    slice_name: str,
) -> Finding:
    """The cut moves some ending's expected share by more than half: the endings are not free.

    Past that point the counts that pass the cut sit within a block or so of it, which is
    the small-count problem the cut exists to avoid, and the smooth-density shares are not
    to be trusted either. Reported as underpowered rather than tested.
    """
    return Finding(
        check=check,
        title=(
            f"Counts of {label} sit too close to the cut at {min_count} to test their endings: "
            f"the cut alone would put between {100 * shares.min():.1f} and "
            f"{100 * shares.max():.1f} per cent on an ending that should get "
            f"{100 / shares.size:.1f}."
        ),
        flag=Flag.UNDERPOWERED,
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        details={
            "min_count": min_count,
            "label": label,
            "expected_proportions": shares.tolist(),
        },
    )


def last_digit_uniformity(
    counts: np.ndarray,
    *,
    min_count: int = MIN_COUNT_FOR_LAST_DIGIT,
    slice_name: str = "all",
    label: str = "votes",
    sizes: np.ndarray | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = 0.95,
    seed: int | None = 0,
) -> Finding:
    """Chi-square test of the last digits on 0 to 9 against what honest counts give.

    Honest here means uniform last digits, cut at ``min_count``: the expected share of each
    digit is :func:`expected_shares_after_cut`, which is exactly uniform when no counts sit near
    the cut and departs from it, in a fixed and predictable direction, when many do. The module
    docstring has why comparing with an exact uniform is wrong for any count that straddles the
    cut.

    The reported effect is the total variation distance from those expected shares, with a
    bootstrap interval over units. ``sizes`` gives the precinct size per unit so that resample
    can be stratified; without it the resample is unstratified and the finding says so. The
    expected shares are held fixed across the resamples: they rest on the counts within ten of
    the cut, and their own noise is a few per cent of the correction, itself a fraction of one
    percentage point.

    A total variation distance cannot be negative, so its interval never contains zero, however
    honest the data. ``details["null_expected_tvd"]`` is what this statistic takes on honest
    digits at this sample size, which is the comparison a reader actually needs, and it is
    reported in the title beside the observed value.
    """
    x = np.asarray(counts, dtype=float)
    usable = np.isfinite(x) & (x >= min_count)
    n_used = int(usable.sum())
    n_excluded = int(usable.size - n_used)

    if n_used < 100:
        return Finding(
            check="last_digit",
            title=(
                f"Only {n_used} counts are at least {min_count}, too few to test the last digit."
            ),
            flag=Flag.UNDERPOWERED,
            n_used=n_used,
            n_excluded=n_excluded,
            slice_name=slice_name,
            details={"min_count": min_count, "label": label},
        )

    shares, below, first = expected_shares_after_cut(x, min_count, 10)
    if shares.min() < 0.05:
        return _too_close_to_the_cut(
            "last_digit", label, min_count, shares, n_used, n_excluded, slice_name
        )

    digits = np.rint(x[usable]).astype(np.int64) % 10
    observed = np.bincount(digits, minlength=10).astype(float)
    expected = n_used * shares
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(chi2, df=9))

    props = observed / n_used
    # Total variation distance from the expected shares: an interpretable effect size, unlike
    # chi-square, which grows with sample size for a fixed departure.
    tvd = float(0.5 * np.abs(props - shares).sum())

    # Resample units and recount the histogram. Cheap: an index into the digits array.
    ci_low, ci_high, ci_details = bootstrap_ci(
        lambda idx: float(
            0.5 * np.abs(np.bincount(digits[idx], minlength=10) / idx.size - shares).sum()
        ),
        n_used,
        estimate=tvd,
        # A distance cannot be negative.
        floor=0.0,
        sizes=None if sizes is None else np.asarray(sizes, dtype=float)[usable],
        n_boot=n_boot,
        level=ci_level,
        seed=None if seed is None else seed + 202,
    )
    expected_tvd = expected_tvd_under_uniform(n_used, 10)

    if pvalue <= 0.001 and tvd >= 0.02:
        flag = Flag.STRONG
    elif pvalue <= 0.05:
        flag = Flag.NOTABLE
    else:
        flag = Flag.OK

    over = int(np.argmax(observed))
    under = int(np.argmin(observed))
    interval = "" if ci_low is None or ci_high is None else f" [{ci_low:.4f}, {ci_high:.4f}]"
    lo, hi = 100 * shares.min(), 100 * shares.max()
    honest = (
        "honest digits would be 10.0 per cent each"
        if f"{lo:.1f}" == f"{hi:.1f}"
        else (
            f"honest digits would be {lo:.1f} to {hi:.1f} per cent each, uneven only because "
            f"counts below {min_count} are left out"
        )
    )
    title = (
        f"Last digit of {label} across {n_used} units: most common {over} "
        f"({100 * props[over]:.1f} per cent), least common {under} "
        f"({100 * props[under]:.1f} per cent); {honest}. "
        f"Distance from honest digits {tvd:.4f}{interval}, against {expected_tvd:.4f} "
        "from sampling noise alone at this sample size."
    )

    return Finding(
        check="last_digit",
        title=title,
        flag=flag,
        statistic=chi2,
        pvalue=pvalue,
        effect=tvd,
        ci_low=ci_low,
        ci_high=ci_high,
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS_LAST_DIGIT if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "observed": observed.astype(int).tolist(),
            "proportions": props.tolist(),
            "expected_proportions": shares.tolist(),
            "counts_just_below_cut": below,
            "counts_just_above_cut": first,
            "df": 9,
            "min_count": min_count,
            "label": label,
            "excluded_because": f"count below {min_count} or missing",
            "null_expected_tvd": expected_tvd,
            "tvd_is_non_negative": (
                "A total variation distance cannot be negative, so the interval never contains "
                "zero and excluding zero means nothing. Compare the observed distance against "
                "null_expected_tvd, which is what honest digits produce at this sample size."
            ),
            **ci_details,
        },
    )


def last_two_digit_pairs(
    counts: np.ndarray,
    *,
    min_count: int = 1000,
    slice_name: str = "all",
    label: str = "votes",
    sizes: np.ndarray | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = 0.95,
    seed: int | None = 0,
) -> Finding:
    """Descriptive summary of the last two digits: repeated pairs against adjacent pairs.

    Under a uniform null, 10 of the 100 pairs are repeats (00, 11, ... 99) and 18 are adjacent
    (differing by one, in either direction). Reported as a description with a chi-square on the
    full 100-cell table. The interpretation of the direction is deliberately left to the reader
    until issue #7 confirms it against the source.

    The cut at ``min_count`` does to the last two digits what the last-digit cut does to the
    last one, over blocks of a hundred rather than ten, so the table is compared with
    :func:`expected_shares_after_cut` at ``period=100`` rather than with an exact uniform. On
    honest counts that straddle 1000 the exact uniform was rejected in 37 of 200 elections at
    4,000 precincts and 34 of 40 at 20,000 (issue #25). The shift barely touches the repeated and adjacent shares,
    whose cells sit spread across the hundred, but it is the whole table that is tested.
    """
    x = np.asarray(counts, dtype=float)
    usable = np.isfinite(x) & (x >= min_count)
    n_used = int(usable.sum())
    n_excluded = int(usable.size - n_used)

    if n_used < 500:
        return Finding(
            check="last_two_digits",
            title=f"Only {n_used} counts are at least {min_count}, too few for a 100-cell table.",
            flag=Flag.UNDERPOWERED,
            n_used=n_used,
            n_excluded=n_excluded,
            slice_name=slice_name,
            details={"min_count": min_count, "label": label},
        )

    shares, below, first = expected_shares_after_cut(x, min_count, 100)
    if shares.min() < 0.005:
        return _too_close_to_the_cut(
            "last_two_digits", label, min_count, shares, n_used, n_excluded, slice_name
        )

    vals = np.rint(x[usable]).astype(np.int64) % 100
    observed = np.bincount(vals, minlength=100).astype(float)
    expected = n_used * shares
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(chi2, df=99))

    tens, ones = np.divmod(np.arange(100), 10)
    is_repeated = tens == ones
    is_adjacent = np.abs(tens - ones) == 1
    repeated = observed[is_repeated].sum() / n_used
    adjacent = observed[is_adjacent].sum() / n_used
    expected_repeated = float(shares[is_repeated].sum())
    expected_adjacent = float(shares[is_adjacent].sum())

    # The reported effect is the repeated-pair share above its expected share, which is the
    # uniform 0.10 moved by the cut: at the default cut of 1000, by less than 0.0001. Signed, so
    # unlike the total variation distance in the check above it has no floor and its interval
    # can straddle zero.
    def _repeated_excess(idx: np.ndarray) -> float:
        counts_i = np.bincount(vals[idx], minlength=100).astype(float)
        return float(counts_i[is_repeated].sum() / idx.size - expected_repeated)

    ci_low, ci_high, ci_details = bootstrap_ci(
        _repeated_excess,
        n_used,
        estimate=float(repeated - expected_repeated),
        sizes=None if sizes is None else np.asarray(sizes, dtype=float)[usable],
        n_boot=n_boot,
        level=ci_level,
        seed=None if seed is None else seed + 505,
    )

    flag = Flag.NOTABLE if pvalue <= 0.05 else Flag.OK
    title = (
        f"Last two digits of {label}: repeated pairs {100 * repeated:.1f} per cent "
        f"(honest {100 * expected_repeated:.1f}), adjacent pairs {100 * adjacent:.1f} per cent "
        f"(honest {100 * expected_adjacent:.1f}), over {n_used} units."
    )
    return Finding(
        check="last_two_digits",
        title=title,
        flag=flag,
        statistic=chi2,
        pvalue=pvalue,
        effect=float(repeated - expected_repeated),
        ci_low=ci_low,
        ci_high=ci_high,
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS_LAST_DIGIT if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "repeated_share": float(repeated),
            "adjacent_share": float(adjacent),
            "uniform_repeated": 0.10,
            "uniform_adjacent": 0.18,
            "expected_repeated": expected_repeated,
            "expected_adjacent": expected_adjacent,
            "counts_just_below_cut": below,
            "counts_just_above_cut": first,
            "df": 99,
            "min_count": min_count,
            "label": label,
            "direction_unverified": True,
            "excluded_because": f"count below {min_count} or missing",
            **ci_details,
        },
    )
