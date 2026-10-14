"""The template check, exercised the way CONTRIBUTING.md requires of every check.

This file and ``src/psephos/methods/_template.py`` are the worked example behind the guide:
quiet on clean data, at a rate near its nominal level across many clean elections, loud on the
output of a named generator, monotone in the injected amount, and honest about being
underpowered. Copy this shape for a new check's tests, including the test at the bottom that
keeps the template out of user-facing output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import (
    CLEAN_RATE_SEEDS,
    assert_clean_rate,
    clean_election,
    with_invented_last_digits,
)

import psephos.methods
from psephos._types import Flag
from psephos.methods._template import zero_five_ending_share

# ---------------------------------------------------------------- quiet


def test_template_quiet_on_clean_data(clean_df):
    f = zero_five_ending_share(clean_df["party_incumbent"].to_numpy(float))
    assert f.flag is Flag.OK, f"clean data was flagged: {f}"
    assert f.pvalue > 0.05


@pytest.mark.xfail(
    strict=False,
    reason=(
        "the 0-or-5 share inherits the cut of issue #25: counts dense at min_count end in 0 "
        "more often than a flat uniform allows, so clean counts flag at about 7 per cent"
    ),
)
def test_template_false_positive_rate_on_clean_data():
    """How often the check fires on clean data is a rate, and one clean election cannot show
    it: an honest check at alpha 0.05 flags about one in twenty. So run it on the 400 clean
    elections of CLEAN_RATE_SEEDS and let assert_clean_rate hold the count to its bound, at most
    30 flags, which a check at twice the nominal rate fails with probability 0.95. A real check
    gets one of these for every column the audit runs it on, in test_false_positive_rate.py.

    Intervals are off (``n_boot=0``) because no flag depends on one, and they are most of the
    cost.

    This one fails, and is recorded as an expected failure rather than loosened. Keeping only
    counts from 100 up and comparing their endings with a flat uniform is the mistake issue #25
    describes for the real last-digit check: wherever counts are dense at the cut, the endings
    above it lean toward 0. Pooled over these 400 elections, 20.11 per cent of the incumbent's
    counts from 100 up end in 0 or 5, not 20.00, which moves this one-sided test from 5 to
    about 7 per cent. That is too close to the bound for 400 elections to call every time: it
    flags 31 of them under numpy 2.5 and 29 under numpy 2.2 and 2.4, whose random streams
    differ, so the marker is not strict here. Where a miscalibration is far past the bound, as
    for the minor parties in test_false_positive_rate.py, it should be. Either way, an expected
    failure with its cause in the reason is how a known miscalibration is recorded: never a
    wider bound or a hand-picked seed."""
    findings = [
        zero_five_ending_share(clean_election(seed=s)["party_incumbent"].to_numpy(float), n_boot=0)
        for s in CLEAN_RATE_SEEDS
    ]
    assert_clean_rate(findings, "template on party_incumbent")


# ---------------------------------------------------------------- loud


def test_template_finds_invented_numbers(clean_df):
    dirty = with_invented_last_digits(clean_df, fraction=0.3)
    f = zero_five_ending_share(dirty["party_incumbent"].to_numpy(float))
    assert f.flag is Flag.STRONG, f
    assert f.effect > 0.02
    assert f.confounds, "a flagged finding must carry confounds"


def test_template_effect_rises_with_injected_fraction(clean_df):
    effects = []
    for frac in (0.0, 0.10, 0.20, 0.35):
        d = with_invented_last_digits(clean_df, fraction=frac) if frac else clean_df
        f = zero_five_ending_share(d["party_incumbent"].to_numpy(float))
        effects.append(f.effect)
    assert effects == sorted(effects), f"effect not monotone in injected fraction: {effects}"


# ---------------------------------------------------------------- underpowered and edges


def test_template_reports_underpowered_rather_than_guessing():
    f = zero_five_ending_share(np.full(50, 250.0))
    assert f.flag is Flag.UNDERPOWERED
    assert f.pvalue is None
    assert f.n_used == 50
    assert f.n_excluded == 0


def test_template_excludes_small_counts_and_says_so(clean_df):
    counts = clean_df["party_incumbent"].to_numpy(float)
    f = zero_five_ending_share(counts, min_count=500)
    assert f.n_used + f.n_excluded == len(counts)
    assert f.n_excluded > 0
    assert "count below 500" in f.details["excluded_because"]


def test_template_hand_computed_on_a_perfect_uniform():
    """Forty copies of each ending 1000 to 1009: exactly 20 per cent land on 0 or 5."""
    counts = np.array([1000 + d for d in range(10)] * 40, dtype=float)
    f = zero_five_ending_share(counts)
    assert f.details["observed_endings"] == 80
    assert f.effect == pytest.approx(0.0)
    assert f.flag is Flag.OK


# ---------------------------------------------------------------- not registered


def test_template_is_not_part_of_any_report():
    """The template is a teaching example. It must stay unreachable from a user's output:
    not exported from the methods package, and never called by the audit. The audit module
    itself is read back, because ``psephos.audit`` as an attribute is the audit function, not
    the module."""
    audit_src = (Path(psephos.__file__).parent / "audit.py").read_text(encoding="utf-8")
    assert "_template" not in psephos.methods.__all__
    assert "_template" not in audit_src
