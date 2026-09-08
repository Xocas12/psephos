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

On Benford's law: the first-digit test is deliberately NOT implemented here. Precinct vote
counts do not span enough orders of magnitude for Benford to hold under honest conditions, so
it produces confident nonsense on exactly this kind of data. docs/why_no_benford.md records
the reasoning, with a demonstration on honest data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from psephos._types import Finding, Flag

#: Counts below this have too few digits for a free last digit.
MIN_COUNT_FOR_LAST_DIGIT = 100

CONFOUNDS_LAST_DIGIT = [
    "Small counts, where the last digit is not free. The min_count threshold controls this.",
    "Vote counts that are systematically rounded during aggregation or transcription, which "
    "produces the same non-uniformity without anyone inventing a number.",
    "Optical-character recognition or transcription error, if the data came from scans, which "
    "distorts digit frequencies on its own.",
]


def last_digit_uniformity(
    counts: np.ndarray,
    *,
    min_count: int = MIN_COUNT_FOR_LAST_DIGIT,
    slice_name: str = "all",
    label: str = "votes",
) -> Finding:
    """Chi-square test that last digits are uniform on 0 to 9."""
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

    digits = np.rint(x[usable]).astype(np.int64) % 10
    observed = np.bincount(digits, minlength=10).astype(float)
    expected = np.full(10, n_used / 10.0)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(chi2, df=9))

    props = observed / n_used
    # Total variation distance from uniform: an interpretable effect size, unlike chi-square,
    # which grows with sample size for a fixed departure.
    tvd = float(0.5 * np.abs(props - 0.1).sum())

    if pvalue <= 0.001 and tvd >= 0.02:
        flag = Flag.STRONG
    elif pvalue <= 0.05:
        flag = Flag.NOTABLE
    else:
        flag = Flag.OK

    over = int(np.argmax(observed))
    under = int(np.argmin(observed))
    title = (
        f"Last digit of {label} across {n_used} units: most common {over} "
        f"({100 * props[over]:.1f} per cent), least common {under} "
        f"({100 * props[under]:.1f} per cent); uniform would be 10.0 per cent."
    )

    return Finding(
        check="last_digit",
        title=title,
        flag=flag,
        statistic=chi2,
        pvalue=pvalue,
        effect=tvd,
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS_LAST_DIGIT if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "observed": observed.astype(int).tolist(),
            "proportions": props.tolist(),
            "df": 9,
            "min_count": min_count,
            "label": label,
            "excluded_because": f"count below {min_count} or missing",
        },
    )


def last_two_digit_pairs(
    counts: np.ndarray,
    *,
    min_count: int = 1000,
    slice_name: str = "all",
    label: str = "votes",
) -> Finding:
    """Descriptive summary of the last two digits: repeated pairs against adjacent pairs.

    Under a uniform null, 10 of the 100 pairs are repeats (00, 11, ... 99) and 18 are adjacent
    (differing by one, in either direction). Reported as a description with a chi-square on the
    full 100-cell table. The interpretation of the direction is deliberately left to the reader
    until issue #7 confirms it against the source.
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

    vals = np.rint(x[usable]).astype(np.int64) % 100
    observed = np.bincount(vals, minlength=100).astype(float)
    expected = np.full(100, n_used / 100.0)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(chi2, df=99))

    tens, ones = np.divmod(np.arange(100), 10)
    repeated = observed[tens == ones].sum() / n_used
    adjacent = observed[np.abs(tens - ones) == 1].sum() / n_used

    flag = Flag.NOTABLE if pvalue <= 0.05 else Flag.OK
    title = (
        f"Last two digits of {label}: repeated pairs {100 * repeated:.1f} per cent "
        f"(uniform 10.0), adjacent pairs {100 * adjacent:.1f} per cent (uniform 18.0), "
        f"over {n_used} units."
    )
    return Finding(
        check="last_two_digits",
        title=title,
        flag=flag,
        statistic=chi2,
        pvalue=pvalue,
        effect=float(repeated - 0.10),
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS_LAST_DIGIT if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "repeated_share": float(repeated),
            "adjacent_share": float(adjacent),
            "uniform_repeated": 0.10,
            "uniform_adjacent": 0.18,
            "df": 99,
            "min_count": min_count,
            "label": label,
            "direction_unverified": True,
            "excluded_because": f"count below {min_count} or missing",
        },
    )
