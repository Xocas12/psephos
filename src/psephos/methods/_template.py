"""A working template for new checks. Copy the shape, not the statistic.

This module is a teaching example and is deliberately unreachable from a user's output: it is
not exported from ``psephos.methods``, ``psephos.audit`` never calls it, and nothing it computes
can appear in any report. Its only job is to demonstrate, on a statistic simple enough to hold
in your head, the six things CONTRIBUTING.md requires of every new check, with the tests that
prove them in ``tests/test_template.py``. Read the two side by side: the docstring here states
the null and the citation decision, and the tests hold the check to both.

The statistic is the share of vote counts whose last digit is 0 or 5.

The null, stated in one sentence as the guide requires: under the null, the last digit of every
usable count is uniform on 0 to 9, so 20 per cent of counts end in 0 or 5. The one-sentence bar
exists because a null you cannot state that briefly is a null you do not understand well enough
to implement.

The confounds live in ``CONFOUNDS`` below and travel with every flagged finding. That part is
not optional: ``Finding.__post_init__`` raises if a finding flagged notable or strong carries
none.

The tests map onto the guide. The quiet test runs this check on ``clean_election()`` and demands
``Flag.OK``. The loud test runs it on output of ``with_invented_last_digits``, a named generator
in ``tests/conftest.py`` that rewrites a known fraction of counts with human-looking endings
(heavy on 0 and 5), and demands the measured effect rise with that fraction. A third covers the
underpowered path: with too few usable units the check returns ``Flag.UNDERPOWERED`` and no
p-value, never a reassuring null.

Citation. The idea that the last digits of genuine counts are uniform, and that invented ones
are not, comes from the digit-testing literature; psephos's real check in that family is
:mod:`psephos.methods.digits`, following Beber and Scacco, "What the Numbers Say: A Digit-Based
Test for Election Fraud", Political Analysis 20(2), 2012. The 0-or-5 statistic itself is NOT
from that paper or any other: it was invented for this template because it is the simplest null
with a closed form. It carries no authority, must not be cited, and must not be registered as a
real check; the digit family is already covered.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from psephos._types import Finding, Flag

#: Counts below this are too small for the last digit to be free, the same reasoning as the
#: real digit check. Units below it are excluded and counted.
DEFAULT_MIN_COUNT = 100

#: Below this many usable units the test cannot say anything either way, and reports that
#: rather than a p-value.
MINIMUM_USABLE = 100

#: The share of counts ending in 0 or 5 under the null of uniform last digits.
NULL_SHARE = 0.20

#: An excess at least this large is required for the strong flag, so that a large sample cannot
#: turn a trivial departure into a strong one.
STRONG_EFFECT_FLOOR = 0.02

CONFOUNDS = [
    "Counts that are round because of allocation rather than invention: ballot papers, funds or "
    "seats distributed per station in round amounts produce counts ending in 0 without anyone "
    "inventing a number.",
    "Rounding during aggregation or transcription, which produces the same endings on counts "
    "that were honestly collected.",
    "Small counts, where the last digit is not free. The min_count threshold controls this.",
]


def zero_five_ending_share(
    counts: np.ndarray,
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    slice_name: str = "all",
    label: str = "votes",
) -> Finding:
    """Test the share of counts ending in 0 or 5 against a uniform-last-digit null.

    Parameters
    ----------
    counts : array-like
        One count per unit.
    min_count : int
        Units with a smaller count are excluded. See :data:`DEFAULT_MIN_COUNT`.
    slice_name : str
        Which subset this was computed on, for the report.
    label : str
        What the counts are, for the title.
    """
    x = np.asarray(counts, dtype=float)
    usable = np.isfinite(x) & (x >= min_count)
    n_used = int(usable.sum())
    n_excluded = int(usable.size - n_used)

    if n_used < MINIMUM_USABLE:
        return Finding(
            check="template_zero_five_endings",
            title=(
                f"Only {n_used} counts are at least {min_count}, "
                "which is too few for this test to say anything."
            ),
            flag=Flag.UNDERPOWERED,
            n_used=n_used,
            n_excluded=n_excluded,
            slice_name=slice_name,
            details={"min_count": min_count, "label": label},
        )

    last_digits = np.rint(x[usable]).astype(np.int64) % 10
    observed = int(np.isin(last_digits, (0, 5)).sum())
    share = observed / n_used
    excess = share - NULL_SHARE

    # The null has a closed form: under uniform last digits, the number of counts ending in 0
    # or 5 is Binomial(n_used, 0.20), so the p-value below is exact. No Monte Carlo, no seed.
    # A template for a null without a closed form would copy the simulation loop in
    # integer_pct.py instead.
    pvalue = float(stats.binom.sf(observed - 1, n_used, NULL_SHARE))
    sd = float(np.sqrt(n_used * NULL_SHARE * (1.0 - NULL_SHARE)))
    z = (observed - n_used * NULL_SHARE) / sd

    # Two conditions for the strong flag, as in the real checks: a p-value alone grows more
    # impressive as the sample grows at a fixed departure, so a large excess is required too.
    if pvalue <= 0.001 and excess >= STRONG_EFFECT_FLOOR:
        flag = Flag.STRONG
    elif pvalue <= 0.05:
        flag = Flag.NOTABLE
    else:
        flag = Flag.OK

    title = (
        f"{observed} of {n_used} {label} counts end in 0 or 5; "
        f"uniform last digits predict about {n_used * NULL_SHARE:.0f}. "
        f"Excess {100 * excess:+.2f} percentage points."
    )

    return Finding(
        check="template_zero_five_endings",
        title=title,
        flag=flag,
        statistic=float(z),
        pvalue=pvalue,
        effect=float(excess),
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "observed_endings": observed,
            "share": share,
            "null_share": NULL_SHARE,
            "expected_endings": n_used * NULL_SHARE,
            "min_count": min_count,
            "label": label,
            "excluded_because": f"count below {min_count} or missing",
        },
    )
