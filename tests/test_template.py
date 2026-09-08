"""The template check, exercised the way CONTRIBUTING.md requires of every check.

This file and ``src/psephos/methods/_template.py`` are the worked example behind the guide:
quiet on clean data, loud on the output of a named generator, monotone in the injected amount,
and honest about being underpowered. Copy this shape for a new check's tests, including the
test at the bottom that keeps the template out of user-facing output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import clean_election, with_invented_last_digits

import psephos.methods
from psephos._types import Flag
from psephos.methods._template import zero_five_ending_share

# ---------------------------------------------------------------- quiet


def test_template_quiet_on_clean_data(clean_df):
    f = zero_five_ending_share(clean_df["party_incumbent"].to_numpy(float))
    assert f.flag is Flag.OK, f"clean data was flagged: {f}"
    assert f.pvalue > 0.05


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_template_false_positive_rate_across_seeds(seed):
    """Across independent clean elections the check should rarely fire. As with the real
    checks, this is the closest thing to a false-positive rate the suite can assert cheaply."""
    df = clean_election(n=3000, seed=100 + seed)
    f = zero_five_ending_share(df["party_incumbent"].to_numpy(float))
    assert f.flag in (Flag.OK, Flag.NOTABLE)


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
