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
:func:`expected_shares_after_cut` derives from how many counts sit in the two blocks of values
just below the cut and the two just above it, ten values a block for the last digit and a
hundred for the last two. With no counts near the cut it is exactly uniform. Block totals do
not depend on the last digit when the cut is a multiple of the block, so inventing last digits
does not move the expected shares at all.

That rests on the count density being smooth across a block, and it need not be. Counts
spread by little more than counting noise, about ten votes either side of a hundred, change
too fast within a block of ten for the totals of whole blocks to say how they are spread over
the endings; so do counts that rise or fall within less than a block, such as the edge of a
narrow band of precinct sizes. Each check therefore also fits a second smooth shape to the same
block totals and refuses to test, as underpowered, when the two disagree by more than the
sample can tolerate (:data:`MAX_SHAPE_DISAGREEMENT`). That limit depends on the number of
counts, so a column tested at 4,000 precincts can be refused at 20,000.

On Benford's law: the first-digit test is deliberately NOT implemented here. Precinct vote
counts do not span enough orders of magnitude for Benford to hold under honest conditions, so
it produces confident nonsense on exactly this kind of data. docs/why_no_benford.md records
the reasoning, with a demonstration on honest data.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

from psephos._types import Finding, Flag
from psephos.interval import DEFAULT_N_BOOT, bootstrap_ci, expected_tvd

#: Counts below this have too few digits for a free last digit.
MIN_COUNT_FOR_LAST_DIGIT = 100

#: How far the expected shares may be in doubt before a check refuses to test. The doubt is
#: ``n_used * sum((other - shares) ** 2 / shares)``, where ``shares`` are the tested shares and
#: ``other`` the shares a second smooth shape gives from the same block totals: what testing
#: against ``shares`` would add to the chi-square's expected value were ``other`` the truth.
#: At 0.5 that would move a 5 per cent test to 6.6 per cent on the last digit's nine degrees of
#: freedom and to 5.4 per cent on the pair table's 99.
MAX_SHAPE_DISAGREEMENT = 0.5

#: Blocks with fewer counts than this are left out of the second shape: an exponential fitted
#: to a handful of counts says nothing about the density, and such blocks hold too few counts
#: to move the shares.
MIN_COUNTS_FOR_SHAPE = 10

CONFOUNDS_LAST_DIGIT = [
    "Counts dense at the min_count cut. Leaving out counts below min_count, which is needed "
    "because a small count's last digit is not free, makes the last digits uneven by "
    "themselves wherever many counts sit near the cut: the endings just above it collect more "
    "counts than those furthest from it. The expected shares allow for that, estimated from "
    "the counts just either side of the cut on the assumption that the count density is "
    "smooth there, and the check declines to test when two smooth shapes disagree; counts "
    "that change steeply within a few votes of the cut can still depart from them without any "
    "number being invented.",
    "Vote counts that are systematically rounded during aggregation or transcription, which "
    "produces the same non-uniformity without anyone inventing a number.",
    "Optical-character recognition or transcription error, if the data came from scans, which "
    "distorts digit frequencies on its own.",
]


def _block_totals(x: np.ndarray, min_count: int, period: int, first: int, stop: int) -> np.ndarray:
    """Counts in blocks ``first`` to ``stop - 1``, block j holding the ``period`` values from
    ``min_count + j * period`` up."""
    j = np.floor((x - min_count) / period).astype(np.int64)
    j = j[(j >= first) & (j < stop)]
    return np.bincount(j - first, minlength=stop - first).astype(float)


@lru_cache(maxsize=8)
def _quartic_weights(period: int) -> np.ndarray:
    """``a[i, p]``: a quartic density whose totals over five consecutive blocks are D_-2 to D_2
    puts ``sum_i a[i, p] * D_(i - 2)`` at position p of the middle block."""
    centre = (period - 1) / 2.0
    p = np.arange(period)
    # moments[i, d]: the sum over block i - 2 of ((position - centre) / period) ** d.
    moments = np.array(
        [
            [(((i * period + p - centre) / period) ** d).sum() for d in range(5)]
            for i in range(-2, 3)
        ]
    )
    powers = np.array([((p - centre) / period) ** d for d in range(5)])
    weights = np.linalg.solve(moments.T, powers)
    weights.setflags(write=False)
    return weights


def _log_quadratic_deviation(totals: np.ndarray, period: int) -> np.ndarray:
    """Within the middle block of each row of three block totals, how far a density
    ``exp(b u + c u^2)`` matching all three sits from flat, position by position. NaN rows did
    not converge."""
    rows = totals.shape[0]
    u = (np.arange(-period, 2 * period) - (period - 1) / 2.0) / period
    u3 = u.reshape(3, period)
    y_low = np.log(totals[:, 0] / totals[:, 1])
    y_high = np.log(totals[:, 2] / totals[:, 1])
    b = (y_high - y_low) / 2.0
    c = (y_high + y_low) / 2.0
    converged = np.zeros(rows, dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(60):
            e = b[:, None] * u + c[:, None] * u * u
            w = np.exp(e - e.max(axis=1, keepdims=True)).reshape(rows, 3, period)
            s = w.sum(axis=2)
            r_low = np.log(s[:, 0] / s[:, 1]) - y_low
            r_high = np.log(s[:, 2] / s[:, 1]) - y_high
            converged = (np.abs(r_low) < 1e-10) & (np.abs(r_high) < 1e-10)
            if converged.all():
                break
            m1 = (w * u3).sum(axis=2) / s
            m2 = (w * u3 * u3).sum(axis=2) / s
            j11, j12 = m1[:, 0] - m1[:, 1], m2[:, 0] - m2[:, 1]
            j21, j22 = m1[:, 2] - m1[:, 1], m2[:, 2] - m2[:, 1]
            det = j11 * j22 - j12 * j21
            b = b - (r_low * j22 - r_high * j12) / det
            c = c - (j11 * r_high - j21 * r_low) / det
        middle = w[:, 1, :]
        deviation = totals[:, 1:2] * (middle / middle.sum(axis=1, keepdims=True) - 1.0 / period)
    deviation[~(converged & np.isfinite(deviation).all(axis=1))] = np.nan
    return deviation


def _shares_after_cut(
    counts: np.ndarray, min_count: int, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tested shares, the second shape's shares, and the totals of blocks -2 to 1."""
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x)]
    n_used = int((x >= min_count).sum())
    if n_used == 0:
        flat = np.full(period, 1.0 / period)
        return flat, flat.copy(), np.zeros(4)
    weights = _quartic_weights(period)
    n_blocks = int((x.max() - min_count) // period) + 1
    # totals[k] is block k - 2, from two below the cut to two past the last count.
    totals = _block_totals(x, min_count, period, -2, n_blocks + 2)

    # Summed over every block from the cut up, the quartic telescopes; see
    # expected_shares_after_cut.
    d_m2, d_m1, d_0, d_1 = totals[:4]
    shifts = np.array([d_m2 + d_m1, d_m1, 0.0, -d_0, -d_0 - d_1])
    by_position = n_used / period + shifts @ weights

    # The second shape, block by block: an exponential of a quadratic through each block and
    # its neighbours, swapped in for the quartic wherever a block has counts enough to fit.
    windows = np.stack([totals[k : k + n_blocks] for k in range(5)], axis=1)
    quartic = windows @ weights - windows[:, 2:3] / period
    fit = windows[:, 2] >= MIN_COUNTS_FOR_SHAPE
    other = by_position.copy()
    if fit.any():
        log_quadratic = _log_quadratic_deviation(np.maximum(windows[fit, 1:4], 0.5), period)
        good = ~np.isnan(log_quadratic).any(axis=1)
        other += (log_quadratic[good] - quartic[fit][good]).sum(axis=0)

    # Ending e sits at position (e - min_count) % period of every block that starts at the cut.
    position = (np.arange(period) - min_count) % period
    return by_position[position] / n_used, other[position] / n_used, totals[:4]


def expected_shares_after_cut(counts: np.ndarray, min_count: int, period: int = 10) -> np.ndarray:
    """Expected share of each ending among the counts of at least ``min_count``.

    ``period`` is 10 for the last digit and 100 for the last two; the shares are indexed by
    ending.

    The null is that the count density is smooth and nothing else favours an ending. Split the
    counts into blocks of ``period`` consecutive values with block 0 starting at the cut, and let
    ``D_j`` be the number of counts in block j. Within block j take the quartic density whose
    totals over blocks j-2 to j+2 are those five ``D``: at position p of the block it puts
    ``sum_i a_i(p) D_(j+i)``, with weights ``a_i(p)`` that depend only on p and ``period``, and
    which sum over i to ``1 / period`` because a flat density must come out flat. Every value at
    position p of any block has the same ending, so the expected number of counts with that
    ending is the sum over blocks j >= 0, which is ``sum_i a_i(p) S_i`` with ``S_i`` the counts
    in blocks i and up. Each ``S_i`` is ``n_used`` plus or minus the blocks between the cut and
    i, so only ``D_-2``, ``D_-1``, ``D_0`` and ``D_1`` are left:

        E(p) = n_used / period + a_-2(p) (D_-2 + D_-1) + a_-1(p) D_-1
                                - a_1(p) D_0 - a_2(p) (D_0 + D_1)

    The same construction with a quadratic through three blocks leaves only ``D_-1`` and
    ``D_0``; the quartic follows a density that bends within a few blocks of the cut more
    closely (tests/test_digit_calibration.py has the comparison).

    Checked against digit distributions computed exactly from known densities, at a cut of
    100. Lognormal counts like the clean generator's parties: within 0.00001, against departures
    from uniform of 0.003 to 0.012. The minor party of near_the_cut_election, binomial with
    registered voters uniform on 1000 to 1400: 0.00007 against 0.008. Narrowing that band to
    1000-1200, 1000-1100, 950-1050 and 900-1000 leaves the counts spread by little more than
    counting noise, about ten votes, around 109, 104, 99 and 94; the error grows to 0.0003,
    0.0014, 0.0025 and 0.0020 against departures of 0.016 to 0.06, where the quadratic was off
    by 0.0012, 0.0036, 0.0060 and 0.0069. A normal density peaking at the cut is off by 0.0001
    with a standard deviation of 20 votes, 0.0007 with 13, 0.002 with 10, 0.008 with 7 and 0.02
    with 5.

    Those errors matter in proportion to the number of counts, and the quartic does not remove
    them; the shape check in each test is what fences them. The case is counts spread by about
    counting noise with their mass within a block or so of the cut. With the quadratic, honest
    minor parties in the 1000-1100, 950-1050 and 900-1000 bands were flagged in 9, 32 and 15 of
    40 elections at 20,000 precincts, 20 of them strong. Now 1000-1100 is flagged in 1 of the 34
    tested and refused in 6, 950-1050 is refused in all 40, and 900-1000 is flagged in 4 of 35
    and refused in 5. At 4,000 precincts most are tested and 4.5 to 9.5 per cent of those are
    flagged; at 95,000, 1000-1100 is refused every time. tests/test_digit_calibration.py holds
    these bands to the nominal rate among the elections tested.
    """
    return _shares_after_cut(counts, min_count, period)[0]


def _blocks_near_cut(min_count: int, period: int, totals: np.ndarray) -> dict[str, int]:
    """The four block totals the expected shares rest on, keyed by the values each block holds."""
    out = {}
    for j, total in zip(range(-2, 2), totals, strict=True):
        lo = min_count + j * period
        out[f"{lo}-{lo + period - 1}"] = int(total)
    return out


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


def _shape_in_doubt(
    check: str,
    label: str,
    min_count: int,
    shares: np.ndarray,
    other: np.ndarray,
    doubt: float,
    n_used: int,
    n_excluded: int,
    slice_name: str,
) -> Finding:
    """Two smooth shapes that fit the same block totals disagree by more than the sample can
    bear: the endings' null is not known precisely enough to test this many counts against."""
    period = shares.size
    return Finding(
        check=check,
        title=(
            f"Counts of {label} change too fast within a block of {period} values for the "
            f"share each ending should get to be known as precisely as {n_used} counts would "
            f"test it: two smooth shapes that fit the same counts per block put up to "
            f"{100 * np.abs(other - shares).max():.2f} percentage points apart on one ending, "
            f"which on its own would add {doubt:.1f} to the chi-square (the limit is "
            f"{MAX_SHAPE_DISAGREEMENT})."
        ),
        flag=Flag.UNDERPOWERED,
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        details={
            "min_count": min_count,
            "label": label,
            "expected_proportions": shares.tolist(),
            "second_shape_proportions": other.tolist(),
            "shape_disagreement": doubt,
            "max_shape_disagreement": MAX_SHAPE_DISAGREEMENT,
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
    cut. When a second smooth shape fitted to the same counts per ten disagrees with those
    shares by more than :data:`MAX_SHAPE_DISAGREEMENT` allows at this sample size, the check
    reports UNDERPOWERED instead of testing: the null itself is not known well enough.

    The reported effect is the total variation distance from those expected shares, with a
    bootstrap interval over units. ``sizes`` gives the precinct size per unit so that resample
    can be stratified; without it the resample is unstratified and the finding says so. The
    expected shares are held fixed across the resamples: they rest on the counts within twenty
    of the cut, and their own noise is a few per cent of the correction, itself a fraction of
    one percentage point.

    A total variation distance cannot be negative, so its interval never contains zero, however
    honest the data. ``details["null_expected_tvd"]`` is the mean distance from the expected
    shares of digits drawn from those shares at this sample size, which is the comparison a
    reader actually needs, and it is reported in the title beside the observed value.
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

    shares, other, near_cut = _shares_after_cut(x, min_count, 10)
    if shares.min() < 0.05:
        return _too_close_to_the_cut(
            "last_digit", label, min_count, shares, n_used, n_excluded, slice_name
        )
    doubt = n_used * float(((other - shares) ** 2 / shares).sum())
    if doubt > MAX_SHAPE_DISAGREEMENT:
        return _shape_in_doubt(
            "last_digit", label, min_count, shares, other, doubt, n_used, n_excluded, slice_name
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
    null_tvd = expected_tvd(n_used, shares)

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
        f"Distance from honest digits {tvd:.4f}{interval}, against {null_tvd:.4f} "
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
            "counts_by_block_near_cut": _blocks_near_cut(min_count, 10, near_cut),
            "shape_disagreement": doubt,
            "df": 9,
            "min_count": min_count,
            "label": label,
            "excluded_because": f"count below {min_count} or missing",
            "null_expected_tvd": null_tvd,
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
    4,000 precincts and 34 of 40 at 20,000 (issue #25). The shift barely touches the repeated
    and adjacent shares, whose cells sit spread across the hundred, but it is the whole table
    that is tested.

    A block here is a hundred values, so the counts have to be smooth over a hundred, and a
    winner whose counts sit within a hundred or two of 1000 is not: a standard deviation under
    about a hundred votes, as when precincts are much of a size and the winner's share hardly
    varies, leaves the last two digits unfree with nothing invented. With the quadratic of the
    first version of this correction, winners centred on 1000 with standard deviations of 62
    and 90 votes were flagged in 31 and 21 of 200 elections at 4,000 precincts and 31 and 18 of
    40 at 20,000. Every one of those elections is now refused as underpowered, by the shape
    check or, where the expected shares fall below half their flat value, as too close to the
    cut.
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

    shares, other, near_cut = _shares_after_cut(x, min_count, 100)
    if shares.min() < 0.005:
        return _too_close_to_the_cut(
            "last_two_digits", label, min_count, shares, n_used, n_excluded, slice_name
        )
    doubt = n_used * float(((other - shares) ** 2 / shares).sum())
    if doubt > MAX_SHAPE_DISAGREEMENT:
        return _shape_in_doubt(
            "last_two_digits",
            label,
            min_count,
            shares,
            other,
            doubt,
            n_used,
            n_excluded,
            slice_name,
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
            "counts_by_block_near_cut": _blocks_near_cut(min_count, 100, near_cut),
            "shape_disagreement": doubt,
            "df": 99,
            "min_count": min_count,
            "label": label,
            "direction_unverified": True,
            "excluded_because": f"count below {min_count} or missing",
            **ci_details,
        },
    )
