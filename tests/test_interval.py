"""Effect-size intervals: what they bracket, how they narrow, and the bias they correct.

Issue #16 asks for an interval on every reported effect, resampling units within precinct-size
bands, printed as the headline instead of the p-value. The tests that matter here are the ones
that would pass on a broken implementation of that: an interval that does not contain its own
estimate, an interval whose width ignores the sample size, and a resample that quietly changes
the size mix the estimate came from.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import clean_election, with_invented_last_digits, with_rounded_turnout

from psephos._types import Finding, Flag
from psephos.interval import (
    MIN_UNITS_FOR_INTERVAL,
    bootstrap_ci,
    expected_tvd_under_uniform,
    stratum_codes,
)
from psephos.methods.digits import last_digit_uniformity
from psephos.methods.integer_pct import integer_excess
from psephos.methods.turnout import turnout_share_dependence

N_BOOT = 300  # enough for these assertions, small enough to keep the suite quick


# --------------------------------------------------------------------- the machinery


def test_an_interval_contains_its_own_estimate():
    x = np.random.default_rng(0).normal(5.0, 2.0, 800)
    lo, hi, details = bootstrap_ci(
        lambda idx: float(x[idx].mean()),
        x.size,
        estimate=float(x.mean()),
        n_boot=N_BOOT,
        seed=1,
    )
    assert lo is not None and hi is not None
    assert lo <= x.mean() <= hi
    assert details["ci_level"] == 0.95
    assert details["ci_n_boot"] == N_BOOT


def test_the_interval_narrows_as_the_sample_grows():
    """The point of the issue: the same effect on 300 units and on 30,000 is not the same
    finding, and before this the output could not tell them apart."""
    widths = []
    for n in (300, 3_000, 30_000):
        x = np.random.default_rng(4).normal(5.0, 2.0, n)
        lo, hi, _ = bootstrap_ci(
            lambda idx: float(x[idx].mean()),  # noqa: B023
            n,
            estimate=float(x.mean()),
            n_boot=N_BOOT,
            seed=1,
        )
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    # Roughly 1/sqrt(n): a hundredfold in n should be about tenfold in width.
    assert 5.0 < widths[0] / widths[2] < 20.0


def test_resampling_is_stratified_so_the_size_mix_is_held_fixed():
    """Each band contributes exactly as many units as it has, in every resample.

    Without this the resample draws size mixes the election did not have, and the interval
    widens with variation in a quantity that is fixed and known.
    """
    sizes = np.concatenate([np.full(400, 150.0), np.full(100, 1500.0)])
    seen: list[tuple[int, int]] = []

    def statistic(idx: np.ndarray) -> float:
        small = int((sizes[idx] < 1000).sum())
        seen.append((small, int(idx.size - small)))
        return float(sizes[idx].mean())

    bootstrap_ci(
        statistic, sizes.size, estimate=float(sizes.mean()), sizes=sizes, n_boot=50, seed=2
    )
    assert seen, "the statistic should have been called"
    assert set(seen) == {(400, 100)}, "every resample must keep 400 small and 100 large units"


def test_one_size_band_is_not_reported_as_stratified():
    """A per-size-band finding passes the sizes of one band, so every unit is in one stratum.

    Claiming "stratified by precinct size" there describes the argument, not the method: there
    is nothing to hold fixed. This is a normal path - the audit produces one such finding per
    size band on every run - so the label has to tell the truth on it.
    """
    sizes = np.full(400, 150.0)  # all inside the 100-249 band
    _, _, details = bootstrap_ci(
        lambda idx: float(sizes[idx].mean()),
        sizes.size,
        estimate=150.0,
        sizes=sizes,
        n_boot=60,
        seed=0,
    )
    assert details["ci_n_strata"] == 1
    assert details["ci_stratified_by"] is None, "one band is not stratification"
    assert "unstratified in effect" in str(details["ci_note"])


def test_several_size_bands_are_reported_as_stratified():
    sizes = np.concatenate([np.full(200, 150.0), np.full(200, 1500.0)])
    _, _, details = bootstrap_ci(
        lambda idx: float(sizes[idx].mean()),
        sizes.size,
        estimate=float(sizes.mean()),
        sizes=sizes,
        n_boot=60,
        seed=0,
    )
    assert details["ci_n_strata"] == 2
    assert details["ci_stratified_by"] == "precinct size"


def test_the_bootstrap_uses_the_size_bands_size_py_defines():
    """docs/effect_intervals.md says the bands are the ones size.py uses. Make that true by
    construction rather than by two lists that happen to match today."""
    import inspect

    from psephos.size import DEFAULT_EDGES

    assert inspect.signature(bootstrap_ci).parameters["edges"].default is DEFAULT_EDGES


def test_unstratified_resampling_is_reported_as_such():
    x = np.random.default_rng(0).normal(0.0, 1.0, 500)
    _, _, details = bootstrap_ci(
        lambda idx: float(x[idx].mean()), x.size, estimate=float(x.mean()), n_boot=50, seed=3
    )
    assert details["ci_stratified_by"] is None
    assert "unstratified" in str(details["ci_note"])


def test_stratum_codes_put_units_in_bands_and_exclude_impossible_sizes():
    edges = (0.0, 100.0, 250.0, np.inf)
    sizes = np.array([50.0, 150.0, 900.0, np.nan, 0.0, -5.0])
    codes = stratum_codes(sizes, sizes.size, edges)
    assert codes[0] == 0 and codes[1] == 1 and codes[2] == 2
    assert (codes[3:] == -1).all(), "missing and non-positive sizes belong to no band"
    assert (stratum_codes(None, 4, edges) == 0).all(), "no sizes means one band"


def test_same_seed_reproduces_the_interval_and_another_seed_does_not():
    x = np.random.default_rng(0).normal(0.0, 1.0, 600)
    args = dict(n_boot=N_BOOT, estimate=float(x.mean()))
    a = bootstrap_ci(lambda idx: float(x[idx].mean()), x.size, seed=7, **args)
    b = bootstrap_ci(lambda idx: float(x[idx].mean()), x.size, seed=7, **args)
    c = bootstrap_ci(lambda idx: float(x[idx].mean()), x.size, seed=8, **args)
    assert a[:2] == b[:2]
    assert a[:2] != c[:2]


def test_no_interval_below_the_minimum_units_and_the_finding_says_why():
    n = MIN_UNITS_FOR_INTERVAL - 1
    x = np.random.default_rng(0).normal(0.0, 1.0, n)
    lo, hi, details = bootstrap_ci(
        lambda idx: float(x[idx].mean()), n, estimate=float(x.mean()), n_boot=N_BOOT, seed=0
    )
    assert lo is None and hi is None
    assert str(MIN_UNITS_FOR_INTERVAL) in str(details["ci_note"])


def test_n_boot_zero_turns_intervals_off():
    lo, hi, details = bootstrap_ci(lambda idx: 0.0, 500, estimate=0.0, n_boot=0)
    assert (lo, hi) == (None, None)
    assert "disabled" in str(details["ci_note"])


def test_a_statistic_that_is_mostly_undefined_gets_no_interval():
    """Quantiles taken from the few resamples that worked would be drawn from a biased subset.

    The condition has to actually be sometimes-false: an earlier version of this test used
    `idx[0] % 1 == 0`, which is true for every integer, so the statistic was always NaN and the
    threshold below was never the thing being exercised.
    """
    calls = []

    def mostly_nan(idx):
        calls.append(1)
        return 1.0 if len(calls) <= 10 else float("nan")

    lo, hi, details = bootstrap_ci(mostly_nan, 500, estimate=1.0, n_boot=100, seed=0)
    assert 10 < len(calls), "the statistic must have returned both finite values and NaN"
    assert lo is None and hi is None
    assert "only 10 of 100" in str(details["ci_note"])


def test_some_undefined_resamples_above_the_threshold_are_reported_not_hidden():
    """Above the halfway threshold the quantiles are still taken, so say what they came from.

    Silence here is the failure mode: an interval computed from 60 of 100 resamples looks
    exactly like one computed from all 100, and is that much less certain than its width says.
    """
    calls = []

    def sometimes_nan(idx):
        calls.append(1)
        # 70 finite, 30 NaN: above the n_boot // 2 threshold, so an interval IS returned.
        return float(len(calls)) if len(calls) <= 70 else float("nan")

    lo, hi, details = bootstrap_ci(sometimes_nan, 500, estimate=35.0, n_boot=100, seed=0)
    assert lo is not None and hi is not None
    note = str(details["ci_note"])
    assert "30 of 100 resamples produced no finite value" in note
    assert details["ci_resamples_used"] == 70


def test_level_must_be_a_probability():
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci(lambda idx: 0.0, 500, estimate=0.0, level=1.0, n_boot=10)


# --------------------------------------------------------------------- the bias correction


def _digits_of(counts: np.ndarray, min_count: int = 100) -> np.ndarray:
    x = counts[np.isfinite(counts) & (counts >= min_count)]
    return np.rint(x).astype(np.int64) % 10


def _tvd(digits: np.ndarray, n: int) -> float:
    return float(0.5 * np.abs(np.bincount(digits, minlength=10) / n - 0.1).sum())


def test_resampling_always_inflates_a_distance_statistic():
    """The defect the bias shift exists for, measured across twenty clean elections.

    A total variation distance is a sum of absolute deviations, so resampling can only push it
    up: the bootstrap distribution sits above the estimate every time, by a third of the
    interval's own width at the median. That is what tips the plain percentile interval off the
    estimate, and it does not shrink away with sample size.
    """
    shares = []
    for seed in range(20):
        df = clean_election(3000, seed=seed)
        for col in ("party_incumbent", "party_second"):
            digits = _digits_of(df[col].to_numpy(float))
            n = digits.size
            estimate = _tvd(digits, n)
            rng = np.random.default_rng(seed)
            boot = np.array([_tvd(digits[rng.integers(0, n, n)], n) for _ in range(200)])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            shares.append((boot.mean() - estimate) / (hi - lo))

    assert min(shares) > 0.0, "the inflation is one-directional, in all 40 cases"
    assert np.median(shares) > 0.1, "and it is a material fraction of the interval width"


def test_the_plain_percentile_interval_can_exclude_the_estimate_where_the_shipped_one_does_not():
    """A case where the uncorrected method states something false.

    Found by scanning the clean generator: the plain interval misses the estimate in 5 of 40
    seed-and-column combinations, and this is one of them. The point is not the frequency but
    that the failure is reachable on honest data with no injected signal at all.
    """
    digits = _digits_of(clean_election(3000, seed=5)["party_incumbent"].to_numpy(float))
    n = digits.size
    estimate = _tvd(digits, n)

    rng = np.random.default_rng(0)
    boot = np.array([_tvd(digits[rng.integers(0, n, n)], n) for _ in range(300)])
    raw_lo, raw_hi = np.quantile(boot, [0.025, 0.975])
    assert not raw_lo <= estimate <= raw_hi, "this fixture is here because the plain method fails"

    lo, hi, details = bootstrap_ci(
        lambda idx: _tvd(digits[idx], idx.size),
        n,
        estimate=estimate,
        floor=0.0,
        n_boot=300,
        seed=0,
    )
    assert lo is not None and lo <= estimate <= hi
    assert details["ci_bias_shift"] > 0


def test_the_bias_shift_is_negligible_for_a_mean():
    """It must not move an interval that was already right."""
    x = np.random.default_rng(0).normal(5.0, 2.0, 2000)
    _, _, details = bootstrap_ci(
        lambda idx: float(x[idx].mean()),
        x.size,
        estimate=float(x.mean()),
        n_boot=500,
        seed=0,
    )
    assert abs(float(details["ci_bias_shift"])) < 0.05 * x.std()


def test_a_distance_interval_is_floored_at_zero_and_says_so():
    """A negative distance is not a value, and an interval reaching below zero invites being
    read as a test against zero."""
    digits = np.random.default_rng(3).integers(0, 10, 4000)
    n = digits.size
    lo, _hi, details = bootstrap_ci(
        lambda idx: _tvd(digits[idx], idx.size),
        n,
        estimate=_tvd(digits, n),
        floor=0.0,
        n_boot=400,
        seed=0,
    )
    assert lo is not None and lo >= 0.0
    if lo == 0.0:
        assert details["ci_floored_at"] == 0.0


def test_expected_tvd_under_uniform_shrinks_with_n_and_is_not_zero():
    """The number a reader needs beside a distance, since its interval never contains zero."""
    big, small = expected_tvd_under_uniform(20_000), expected_tvd_under_uniform(500)
    assert 0.0 < big < small
    # Roughly 1 / sqrt(n).
    assert 3.0 < small / big < 10.0


# --------------------------------------------------------------------- the checks


def test_integer_percentage_interval_brackets_the_effect_and_matches_the_excess():
    df = clean_election(3000)
    f = integer_excess(
        df["ballots_cast"].to_numpy(float),
        df["registered"].to_numpy(float),
        n_mc=120,
        seed=0,
        n_boot=N_BOOT,
    )
    assert f.ci_low is not None
    assert f.ci_low <= f.effect <= f.ci_high
    # The bootstrapped quantity is the same effect the title reports, per unit.
    assert f.effect == pytest.approx(f.details["excess"] / f.n_used)
    assert f.details["ci_stratified_by"] == "precinct size"
    assert f.details["ci_n_strata"] >= 2


def test_a_clean_election_interval_covers_zero_and_an_injected_one_does_not():
    """The interval has to be able to tell the two apart, or it is decoration."""
    df = clean_election(3000)
    clean = integer_excess(
        df["ballots_cast"].to_numpy(float),
        df["registered"].to_numpy(float),
        n_mc=120,
        seed=0,
        n_boot=N_BOOT,
    )
    assert clean.ci_low <= 0.0 <= clean.ci_high

    dirty_df = with_rounded_turnout(df, fraction=0.15)
    dirty = integer_excess(
        dirty_df["ballots_cast"].to_numpy(float),
        dirty_df["registered"].to_numpy(float),
        n_mc=120,
        seed=0,
        n_boot=N_BOOT,
    )
    assert dirty.ci_low > 0.0
    assert dirty.ci_low > clean.ci_high, "the two intervals should not overlap"


def test_last_digit_interval_brackets_the_effect_on_clean_and_invented_digits():
    df = clean_election(3000)
    for frame in (df, with_invented_last_digits(df, fraction=0.4)):
        f = last_digit_uniformity(
            frame["party_incumbent"].to_numpy(float),
            sizes=frame["registered"].to_numpy(float),
            n_boot=N_BOOT,
            seed=0,
        )
        assert f.ci_low is not None
        assert f.ci_low <= f.effect <= f.ci_high
        assert f.ci_low >= 0.0
        assert f.details["null_expected_tvd"] > 0.0
        assert f.details["ci_stratified_by"] == "precinct size"


def test_last_digit_distance_is_read_against_the_null_not_against_zero():
    """Clean digits sit near the expected distance; invented ones sit well above it."""
    df = clean_election(4000)
    clean = last_digit_uniformity(df["party_incumbent"].to_numpy(float), n_boot=N_BOOT, seed=0)
    dirty_df = with_invented_last_digits(df, fraction=0.4)
    dirty = last_digit_uniformity(
        dirty_df["party_incumbent"].to_numpy(float), n_boot=N_BOOT, seed=0
    )
    assert clean.ci_low <= clean.details["null_expected_tvd"] <= clean.ci_high
    assert dirty.ci_low > dirty.details["null_expected_tvd"]


def test_turnout_dependence_interval_brackets_the_gap():
    df = clean_election(3000)
    reg = df["registered"].to_numpy(float)
    cast = df["ballots_cast"].to_numpy(float)
    votes = df["party_incumbent"].to_numpy(float)
    f = turnout_share_dependence(
        100.0 * cast / reg,
        100.0 * votes / cast,
        weights=cast,
        sizes=reg,
        n_boot=N_BOOT,
        seed=0,
    )
    assert f.ci_low is not None
    assert f.ci_low <= f.effect <= f.ci_high
    assert f.details["ci_stratified_by"] == "precinct size"


def test_n_boot_zero_leaves_every_check_without_an_interval():
    df = clean_election(1500)
    f = integer_excess(
        df["ballots_cast"].to_numpy(float),
        df["registered"].to_numpy(float),
        n_mc=60,
        seed=0,
        n_boot=0,
    )
    g = last_digit_uniformity(df["party_incumbent"].to_numpy(float), n_boot=0)
    assert (f.ci_low, f.ci_high) == (None, None)
    assert (g.ci_low, g.ci_high) == (None, None)
    assert f.effect is not None and g.effect is not None, "the effect is still reported"


# --------------------------------------------------------------------- the Finding guard


def test_a_finding_refuses_an_interval_that_does_not_contain_its_effect():
    with pytest.raises(ValueError, match="does not contain"):
        Finding(check="x", title="t", flag=Flag.OK, effect=0.5, ci_low=0.6, ci_high=0.9)


def test_a_finding_refuses_a_backwards_interval():
    with pytest.raises(ValueError, match="wrong way round"):
        Finding(check="x", title="t", flag=Flag.OK, effect=0.5, ci_low=0.9, ci_high=0.1)


def test_a_finding_refuses_half_an_interval():
    with pytest.raises(ValueError, match="both ends"):
        Finding(check="x", title="t", flag=Flag.OK, effect=0.5, ci_low=0.1)


def test_an_interval_without_an_effect_is_allowed_through():
    """Nothing to bracket, so nothing to check."""
    f = Finding(check="x", title="t", flag=Flag.OK, ci_low=0.1, ci_high=0.9)
    assert f.ci_low == 0.1


# --------------------------------------------------------------------- the quantile shortcut


def test_the_selection_quantile_agrees_with_numpy_to_within_two_ulp():
    """`turnout._quantile` replaces a sort with a selection, inside the bootstrap's hot loop.

    It is allowed to be faster, not different. It is not bit-identical to ``np.quantile`` --
    the two arrange the same interpolation differently -- so what is pinned is the size of the
    disagreement and, in the test below, that it moves no unit across the cut.
    """
    from psephos.methods.turnout import _quantile

    rng = np.random.default_rng(0)
    worst_ulps = 0.0
    for _ in range(2_000):
        n = int(rng.integers(5, 500))
        a = rng.normal(0.0, 1.0, n)
        # Measured against the magnitude of the data, not of the quantile: a quantile that
        # happens to sit near zero would make any relative bound meaningless.
        ulp = float(np.spacing(np.abs(a).max()))
        for q in (0.9, 0.5, 0.1, 0.975, 0.025):
            mine, theirs = _quantile(a, q), float(np.quantile(a, q))
            worst_ulps = max(worst_ulps, abs(mine - theirs) / ulp)
            if ((n - 1) * q).is_integer():
                assert mine == theirs, "an order statistic has no interpolation to round"
    assert worst_ulps <= 2.0, f"worst disagreement {worst_ulps:.2f} ulp"


def test_the_selection_quantile_picks_the_same_top_decile():
    """The cut only matters through the mask it produces, and that must be identical.

    Turnout percentages, the real input, with ties included on purpose: a cut sitting exactly on
    a repeated value is the one case where an epsilon could move a unit.
    """
    from psephos.methods.turnout import _quantile

    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(50, 600))
        a = np.round(rng.uniform(20.0, 95.0, n), 1)  # one decimal place, so ties are common
        mask_mine = a >= _quantile(a, 0.9)
        mask_numpy = a >= float(np.quantile(a, 0.9))
        assert np.array_equal(mask_mine, mask_numpy)


def test_the_selection_quantile_handles_degenerate_arrays():
    from psephos.methods.turnout import _quantile

    assert _quantile(np.array([3.0]), 0.9) == 3.0
    assert _quantile(np.full(10, 2.5), 0.9) == 2.5


def test_resamples_are_streamed_rather_than_held_at_once():
    """A list of a thousand index arrays is 760 MB on a national dataset, for arrays used once."""
    from psephos.interval import _resample_indices

    codes = np.zeros(50, dtype=np.intp)
    out = _resample_indices(codes, np.random.default_rng(0), 1000)
    assert not isinstance(out, list)
    first = next(iter(out))
    assert first.shape == (50,)
