"""Each check must find what it claims to find, and stay quiet otherwise.

The quiet half matters more than the loud half. A check that fires on clean data would make
psephos an engine for laundering suspicion.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    clean_election,
    with_invented_last_digits,
    with_rounded_turnout,
    with_stuffing,
)

from psephos._types import Flag
from psephos.methods.digits import last_digit_uniformity, last_two_digit_pairs
from psephos.methods.integer_pct import integer_excess
from psephos.methods.turnout import turnout_roundness, turnout_share_dependence
from psephos.schema import ElectionData

# ---------------------------------------------------------------- integer percentages


def test_integer_excess_quiet_on_clean_data(clean_df):
    f = integer_excess(
        clean_df["ballots_cast"].to_numpy(float),
        clean_df["registered"].to_numpy(float),
        seed=0,
    )
    assert f.flag is Flag.OK, f"clean data was flagged: {f}"
    assert f.pvalue > 0.05


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_integer_excess_false_positive_rate_across_seeds(seed):
    """Across independent clean elections the check should rarely fire. This is the closest
    thing to a false-positive rate the test suite can assert cheaply."""
    df = clean_election(n=3000, seed=100 + seed)
    f = integer_excess(
        df["ballots_cast"].to_numpy(float), df["registered"].to_numpy(float), seed=seed
    )
    assert f.flag in (Flag.OK, Flag.NOTABLE)


def test_integer_excess_finds_injected_rounding(clean_df):
    dirty = with_rounded_turnout(clean_df, fraction=0.15)
    f = integer_excess(
        dirty["ballots_cast"].to_numpy(float), dirty["registered"].to_numpy(float), seed=0
    )
    assert f.flag is Flag.STRONG, f
    assert f.effect > 0.05
    assert f.confounds, "a flagged finding must carry confounds"


def test_integer_excess_effect_rises_with_injected_fraction(clean_df):
    effects = []
    for frac in (0.0, 0.05, 0.15, 0.30):
        d = with_rounded_turnout(clean_df, fraction=frac) if frac else clean_df
        f = integer_excess(
            d["ballots_cast"].to_numpy(float), d["registered"].to_numpy(float), seed=0
        )
        effects.append(f.effect)
    assert effects == sorted(effects), f"effect not monotone in injected fraction: {effects}"


def test_integer_excess_excludes_small_precincts_and_says_so():
    rng = np.random.default_rng(3)
    reg = np.concatenate([np.full(500, 60), rng.integers(400, 3000, size=1500)])
    cast = rng.binomial(reg, 0.55)
    f = integer_excess(cast, reg, min_denominator=250, seed=0)
    assert f.n_excluded >= 500
    assert "denominator below 250" in f.details["excluded_because"]


def test_integer_excess_reports_underpowered_rather_than_guessing():
    rng = np.random.default_rng(4)
    reg = rng.integers(400, 900, size=40)
    cast = rng.binomial(reg, 0.5)
    f = integer_excess(cast, reg, seed=0)
    assert f.flag is Flag.UNDERPOWERED
    assert f.pvalue is None


def test_integer_excess_is_reproducible(clean_df):
    args = (clean_df["ballots_cast"].to_numpy(float), clean_df["registered"].to_numpy(float))
    a = integer_excess(*args, seed=42)
    b = integer_excess(*args, seed=42)
    assert a.statistic == b.statistic and a.pvalue == b.pvalue


def test_integer_excess_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        integer_excess(np.ones(5), np.ones(6))


# ---------------------------------------------------------------- digits


def test_last_digit_false_positive_rate_is_near_nominal(clean_df):
    """Quietness is a RATE, not a single draw.

    Demanding Flag.OK on one seed asserts that a correct check got lucky: at alpha 0.05 any
    honest check flags about one clean election in twenty. So measure the rate over many
    independent clean elections instead. Measured here: 4 notable and 0 strong out of 40.
    """
    flags = [
        last_digit_uniformity(
            clean_election(n=4000, seed=200 + s)["party_incumbent"].to_numpy(float)
        ).flag
        for s in range(40)
    ]
    strong = sum(f is Flag.STRONG for f in flags)
    flagged = sum(f in (Flag.NOTABLE, Flag.STRONG) for f in flags)
    assert strong == 0, "a strong flag on clean data means the check is badly calibrated"
    assert flagged <= 8, f"{flagged}/40 clean elections flagged; nominal is about 2"


def test_last_digit_finds_invented_numbers(clean_df):
    dirty = with_invented_last_digits(clean_df, fraction=0.35)
    f = last_digit_uniformity(dirty["party_incumbent"].to_numpy(float))
    assert f.flag is Flag.STRONG, f
    assert f.effect > 0.02
    assert f.confounds


def test_last_digit_hand_computed_on_a_perfect_uniform():
    counts = np.array([1000 + d for d in range(10)] * 40, dtype=float)
    f = last_digit_uniformity(counts)
    # Exactly uniform: chi-square is 0 and the total variation distance is 0.
    assert f.statistic == pytest.approx(0.0)
    assert f.effect == pytest.approx(0.0)
    assert f.flag is Flag.OK


def test_last_digit_underpowered_on_small_counts():
    f = last_digit_uniformity(np.full(500, 12.0), min_count=100)
    assert f.flag is Flag.UNDERPOWERED


def test_last_two_digits_runs_and_reports_shares(clean_df):
    f = last_two_digit_pairs(clean_df["party_incumbent"].to_numpy(float), min_count=200)
    assert f.check == "last_two_digits"
    if f.flag is not Flag.UNDERPOWERED:
        assert 0.0 <= f.details["repeated_share"] <= 1.0
        assert f.details["direction_unverified"] is True


# ---------------------------------------------------------------- turnout


def test_turnout_dependence_quiet_when_independent(clean_data):
    f = turnout_share_dependence(
        clean_data.turnout(),
        clean_data.share("party_incumbent"),
        weights=clean_data.votes_frame().sum(axis=1).to_numpy(float),
    )
    assert f.flag is Flag.OK, f


def test_turnout_dependence_finds_stuffing(clean_df, column_map):
    dirty = ElectionData(
        frame=with_stuffing(clean_df, fraction=0.15, boost=0.5),
        columns=column_map,
        source="synthetic-stuffed",
    )
    f = turnout_share_dependence(
        dirty.turnout(),
        dirty.share("party_incumbent"),
        weights=dirty.votes_frame().sum(axis=1).to_numpy(float),
    )
    assert f.flag in (Flag.NOTABLE, Flag.STRONG), f
    assert f.statistic > 0
    assert f.confounds


def _share_moving_with_turnout(slope: float, *, n: int = 3000, seed: int = 0):
    """Honest precincts whose leader's rate moves with turnout by ``slope``.

    Nothing is added to anyone's pile: every count is a binomial draw around a rate that tilts
    with turnout, the way geography or mobilisation tilts it. The reproduction in issue #27.
    """
    rng = np.random.default_rng(seed)
    reg = rng.integers(500, 2000, n)
    t = np.clip(rng.normal(0.55, 0.08, n), 0.2, 0.95)
    cast = rng.binomial(reg, t)
    rate = np.clip(0.75 + slope * (t - 0.55) + rng.normal(0, 0.03, n), 0.05, 0.95)
    win = rng.binomial(cast, rate)
    return 100.0 * cast / reg, 100.0 * win / cast, cast.astype(float)


def test_turnout_dependence_says_falls_when_the_share_falls():
    """Issue #27. The test is two-sided, so a falling share flags, and the headline has to say
    it falls. It said "rises" over rho -0.782 and then printed the gap as "-8.9 points ...
    lower than elsewhere", a double negative that literally says higher."""
    turnout, share, weights = _share_moving_with_turnout(-0.6)
    f = turnout_share_dependence(turnout, share, weights=weights, winner="party_a", n_boot=200)

    assert f.flag is Flag.STRONG, f
    assert f.statistic < 0 and f.effect < 0
    assert f.title.startswith("Share for party_a falls with turnout: Spearman rho -0."), f.title
    assert "rises" not in f.title
    # The gap is signed, top decile minus the rest, with no direction word to fight the sign.
    assert f"minus its share elsewhere is {f.effect:+.1f} points" in f.title, f.title
    assert "lower" not in f.title and "higher" not in f.title, f.title
    # The mobilisation confound is the rival's campaign, not the contestant's own.
    assert len(f.confounds) == 3
    assert not any("its own turnout and its own share" in c for c in f.confounds), f.confounds
    assert any("rival's well-organised campaign" in c for c in f.confounds), f.confounds


def test_turnout_dependence_wording_follows_the_sign_of_rho():
    """Both directions, at both flag levels: the headline word, the signed gap and the
    mobilisation confound all agree with rho. Slope -0.15 is the NOTABLE falling case."""
    seen = set()
    for slope in (-0.6, -0.15, 0.15, 0.6):
        turnout, share, weights = _share_moving_with_turnout(slope)
        f = turnout_share_dependence(turnout, share, weights=weights, n_boot=100)
        assert f.flag in (Flag.NOTABLE, Flag.STRONG), (slope, f)
        assert (f.statistic > 0) == (slope > 0), (slope, f.statistic)
        seen.add((f.flag, f.statistic > 0))

        word, other = ("rises", "falls") if f.statistic > 0 else ("falls", "rises")
        assert f.title.startswith(f"Share for the leading contestant {word} with turnout"), (
            slope,
            f.title,
        )
        assert other not in f.title, (slope, f.title)
        assert f"is {f.effect:+.1f} points" in f.title, (slope, f.title)
        rival = any("rival's well-organised campaign" in c for c in f.confounds)
        assert rival == (f.statistic < 0), (slope, f.confounds)
    assert (Flag.NOTABLE, False) in seen, "the sweep should include a NOTABLE falling share"


def test_turnout_dependence_reads_a_rivals_stuffing_as_a_falling_share(clean_df, column_map):
    """Why a falling share flags at all: votes added to one contestant's pile lower every other
    contestant's share in exactly the precincts where they raise turnout. The incumbent is
    stuffed here, as in the test above, and the share read is party_second's."""
    dirty = ElectionData(
        frame=with_stuffing(clean_df, fraction=0.15, boost=0.5),
        columns=column_map,
        source="synthetic-stuffed",
    )
    f = turnout_share_dependence(
        dirty.turnout(),
        dirty.share("party_second"),
        weights=dirty.votes_frame().sum(axis=1).to_numpy(float),
        winner="party_second",
        n_boot=200,
    )
    assert f.flag in (Flag.NOTABLE, Flag.STRONG), f
    assert f.statistic < 0
    assert f.title.startswith("Share for party_second falls with turnout"), f.title


def test_turnout_dependence_refuses_to_estimate_anomalous_votes(clean_data):
    f = turnout_share_dependence(clean_data.turnout(), clean_data.share("party_incumbent"))
    assert "no_anomalous_vote_estimate" in f.details


def test_turnout_distribution_is_descriptive_only(clean_data):
    f = turnout_roundness(clean_data.turnout())
    assert f.flag is Flag.OK
    assert f.details["descriptive_only"] is True
    assert len(f.details["deciles"]) == 11
