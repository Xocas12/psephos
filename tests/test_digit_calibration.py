"""The digit checks must flag honest counts at their nominal rate, wherever those counts sit.

Issue #25. Both digit checks test only counts at or above a cut, and cutting a count
distribution where it is dense makes its endings uneven with nothing invented. Against an exact
uniform, an honest minor party concentrated just above 100 was flagged in 90 of the 200
elections below at 4,000 precincts and in all 40 at 20,000, and the clean generator's smallest
party in 25 of 200 and 21 of 40. The rate is the thing to test, so these tests run the digit
checks exactly as the audit runs them, through ``run_digit_checks``, on hundreds of independent
honest elections, and hold every column's flag rate to a binomial test against 5 per cent at
the 0.01 level.

``n_boot=0`` because the flag does not depend on the interval, which is the slow part;
``test_the_flag_does_not_depend_on_the_interval`` pins that.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import clean_election, near_the_cut_election, with_invented_last_digits
from scipy import stats

from psephos._types import Flag
from psephos.audit import run_digit_checks
from psephos.methods import digits
from psephos.methods.digits import (
    CONFOUNDS_LAST_DIGIT,
    MIN_COUNT_FOR_LAST_DIGIT,
    last_digit_uniformity,
    last_two_digit_pairs,
)
from psephos.schema import ColumnMap, ElectionData

CLEAN = ColumnMap(
    votes={c: c for c in ("party_incumbent", "party_second", "party_third")},
    registered="registered",
    ballots_cast="ballots_cast",
    precinct_id="precinct",
)
NEAR_THE_CUT = ColumnMap(
    votes={c: c for c in ("party_major", "party_minor")},
    registered="registered",
    ballots_cast="ballots_cast",
    precinct_id="precinct",
)

#: Larger precincts than clean_election's default and a stronger winner, so the winner's
#: counts straddle the last-two-digit cut at 1000 with about half of them above it.
STRADDLING_1000 = {"min_size": 2000, "max_size": 8000, "turnout_mean": 0.65, "winner_mean": 0.6}

#: (precincts, elections): the two scales the issue asked for.
SCALES = [(4000, 200), (20_000, 40)]


def _flagged(f) -> bool:
    return f.flag in (Flag.NOTABLE, Flag.STRONG)


def _assert_nominal(tally: dict[str, list], n_elections: int) -> None:
    """Every column's flag rate passes a two-sided binomial test against 5% at 0.01, and the
    strong flags, pooled over columns, one against 0.1% at 0.01."""
    lines = []
    failed = []
    strong = total = 0
    for column, findings in tally.items():
        assert len(findings) == n_elections
        assert not [f for f in findings if f.flag is Flag.UNDERPOWERED], (
            f"{column}: an underpowered election tests nothing"
        )
        k = sum(_flagged(f) for f in findings)
        p = stats.binomtest(k, n_elections, 0.05).pvalue
        lines.append(f"{column}: {k}/{n_elections} flagged, binomial p={p:.2g}")
        if p < 0.01:
            failed.append(column)
        strong += sum(f.flag is Flag.STRONG for f in findings)
        total += n_elections
    p_strong = stats.binomtest(strong, total, 0.001, alternative="greater").pvalue
    lines.append(f"strong: {strong}/{total}, binomial p={p_strong:.2g}")
    assert not failed and p_strong >= 0.01, "\n".join(lines)


def _digit_findings(frame, columns: ColumnMap, check: str) -> dict:
    data = ElectionData(frame=frame, columns=columns, source="synthetic-honest")
    return {f.slice_name: f for f in run_digit_checks(data, n_boot=0, seed=0) if f.check == check}


# ------------------------------------------------------------------- the calibration


@pytest.mark.parametrize(("n", "n_elections"), SCALES)
def test_last_digit_flag_rate_is_nominal_on_every_contestant_column(n, n_elections):
    """#25: every contestant column of clean_election, plus an honest minor party concentrated
    just above min_count, flagged at the nominal rate at 4,000 and at 20,000 precincts."""
    tally: dict[str, list] = {}
    for s in range(n_elections):
        frames = (
            (clean_election(n=n, seed=2500 + s), CLEAN),
            (near_the_cut_election(n=n, seed=2500 + s), NEAR_THE_CUT),
        )
        for frame, columns in frames:
            for label, f in _digit_findings(frame, columns, "last_digit").items():
                tally.setdefault(label, []).append(f)
    assert set(tally) == {
        "party_incumbent",
        "party_second",
        "party_third",
        "party_major",
        "party_minor",
    }
    _assert_nominal(tally, n_elections)


@pytest.mark.parametrize(("n", "n_elections"), SCALES)
def test_last_two_digits_flag_rate_is_nominal_on_counts_straddling_1000(n, n_elections):
    """#25, the same construction at the 1000 cut: the winner's last two digits, as the audit
    tests them, on honest counts about half of which are at least 1000."""
    findings = []
    for s in range(n_elections):
        frame = clean_election(n=n, seed=2500 + s, **STRADDLING_1000)
        [f] = _digit_findings(frame, CLEAN, "last_two_digits").values()
        findings.append(f)
    # The fixture must actually straddle the cut, or the test says nothing about it.
    kept = [f.n_used / (f.n_used + f.n_excluded) for f in findings]
    assert 0.3 < float(np.median(kept)) < 0.7, f"median share kept {np.median(kept):.2f}"
    _assert_nominal({"party_incumbent": findings}, n_elections)


def test_the_flag_does_not_depend_on_the_interval():
    """The calibration runs with n_boot=0. That is only the audit's call if the flag ignores
    the bootstrap, which is resampled after the flag is decided."""
    frame = near_the_cut_election(n=4000, seed=7)
    data = ElectionData(frame=frame, columns=NEAR_THE_CUT, source="synthetic-honest")
    fast = run_digit_checks(data, n_boot=0, seed=0)
    full = run_digit_checks(data, n_boot=200, seed=0)
    assert [(f.check, f.slice_name, f.flag, f.pvalue) for f in fast] == [
        (f.check, f.slice_name, f.flag, f.pvalue) for f in full
    ]


# ------------------------------------------------------------------- the expected shares


def _near_the_cut_pmf() -> np.ndarray:
    """The exact distribution of near_the_cut_election's minor-party count: binomial with
    probability 0.55 * 0.18 given registered voters, uniform on 1000 to 1400."""
    registered = np.arange(1000, 1401)
    return stats.binom.pmf(np.arange(400)[:, None], registered[None, :], 0.55 * 0.18).mean(axis=1)


def _exact_shares(pmf: np.ndarray, min_count: int, period: int) -> np.ndarray:
    values = np.arange(pmf.size)
    kept = values >= min_count
    shares = np.bincount(values[kept] % period, weights=pmf[kept], minlength=period)
    return shares / shares.sum()


def _counts_with(pmf: np.ndarray) -> np.ndarray:
    """A million counts whose histogram is the distribution, to within rounding: the shares
    are a function of counts, so this is how the exact density goes in."""
    return np.repeat(np.arange(pmf.size), np.rint(pmf / pmf.sum() * 1e6).astype(int)).astype(float)


@pytest.mark.parametrize("min_count", [100, 105])
def test_expected_shares_match_the_exact_ones_for_the_near_the_cut_party(min_count):
    """#25: the correction is a formula, so check it against the digit distribution computed
    exactly from the density of near_the_cut_election's minor party rather than against another
    simulation. With the cut at 105 the excess belongs on digits 5 to 9, where the issue saw it
    move."""
    pmf = _near_the_cut_pmf()
    exact = _exact_shares(pmf, min_count, 10)
    shares, below, first = digits.expected_shares_after_cut(_counts_with(pmf), min_count, 10)
    assert below > 0 and first > 0
    lowest_ending_above_the_cut = min_count % 10
    assert np.argmax(exact) == np.argmax(shares) == lowest_ending_above_the_cut
    # The cut moves the exact shares well away from uniform...
    departure = np.abs(exact - 0.1).max()
    assert departure > 0.007
    # ...and the formula follows them to within a few per cent of that.
    assert np.abs(shares - exact).max() < 0.06 * departure, np.round(shares - exact, 6)


def test_expected_shares_are_uniform_when_nothing_sits_near_the_cut():
    counts = np.arange(1000, 3000, dtype=float)
    shares, below, first = digits.expected_shares_after_cut(counts, 100, 10)
    assert (below, first) == (0, 0)
    assert shares == pytest.approx(np.full(10, 0.1))


def test_invented_last_digits_do_not_move_the_expected_shares():
    """The correction rests on how many counts sit in each ten, which inventing a last digit
    does not change, so it cannot absorb the signal the check exists for."""
    clean = near_the_cut_election(n=4000, seed=3)
    dirty = with_invented_last_digits(clean, fraction=0.3, column="party_minor")
    a = last_digit_uniformity(clean["party_minor"].to_numpy(float), n_boot=0)
    b = last_digit_uniformity(dirty["party_minor"].to_numpy(float), n_boot=0)
    assert a.details["expected_proportions"] == b.details["expected_proportions"]


def test_invented_digits_are_still_found_where_the_correction_is_largest():
    clean = near_the_cut_election(n=4000, seed=3)
    effects = []
    for frac in (0.0, 0.1, 0.2, 0.35):
        frame = (
            with_invented_last_digits(clean, fraction=frac, column="party_minor") if frac else clean
        )
        f = last_digit_uniformity(frame["party_minor"].to_numpy(float), n_boot=0)
        effects.append(f.effect)
    assert effects == sorted(effects), f"effect not monotone in injected fraction: {effects}"
    assert f.flag is Flag.STRONG, f
    assert f.confounds


def test_counts_piled_at_the_cut_are_underpowered_not_tested():
    """Counts that barely clear the cut have no free last digit, and the smooth-density
    shares cannot be trusted there either. Say so instead of testing."""
    rng = np.random.default_rng(0)
    counts = np.concatenate([rng.integers(90, 100, 1000), rng.integers(100, 110, 150)])
    f = last_digit_uniformity(counts.astype(float), n_boot=0)
    assert f.flag is Flag.UNDERPOWERED
    assert f.pvalue is None
    assert f.n_excluded == 1000
    assert "too close to the cut" in f.title


def test_the_repeated_pair_share_barely_moves_at_the_default_cut():
    """The comment on last_two_digit_pairs' effect says the cut moves the expected repeated
    share by less than 0.0001 at 1000, so the effect still reads as an excess over 0.10."""
    for s in range(5):
        frame = clean_election(n=4000, seed=2500 + s, **STRADDLING_1000)
        f = last_two_digit_pairs(frame["party_incumbent"].to_numpy(float), n_boot=0)
        assert abs(f.details["expected_repeated"] - 0.10) < 1e-4


# ------------------------------------------------------------------- the words


def test_the_cut_confound_names_the_truncation_not_a_control():
    """#25: the threshold causes this departure, so the confound must not offer it as the
    control, and must say what the cut does."""
    text = " ".join(CONFOUNDS_LAST_DIGIT)
    assert "controls this" not in text
    assert "min_count threshold controls" not in text
    first = CONFOUNDS_LAST_DIGIT[0]
    assert "min_count" in first and "cut" in first and "uneven" in first


def test_a_flagged_finding_reads_against_the_cut_not_a_flat_uniform():
    """The title's comparison is the honest shares after the cut, which a minor party's
    counts make visibly uneven, not a flat 10 per cent."""
    frame = near_the_cut_election(n=4000, seed=3)
    f = last_digit_uniformity(frame["party_minor"].to_numpy(float), n_boot=0)
    assert f"counts below {MIN_COUNT_FOR_LAST_DIGIT} are left out" in f.title
    assert "genuinely uniform" not in f.title
