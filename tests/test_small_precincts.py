"""Small precincts under the integer-percentage null.

integer_excess redraws every unit at its own size, so the arithmetic that puts a small or
round-sized precinct on a whole-number percentage is in the null as well as in the data. A
100-voter unit lands on one with probability 1 in both. Small precincts therefore cannot raise
the observed count above the null, and the reading guidance has to say so rather than blame them
for an excess. These tests keep the three consequences true:

- an honest election full of small and round-sized precincts, including institutional ones at
  or near complete turnout, stays calibrated with the size threshold switched off;
- every confound the check prints is a mechanism that puts the count itself on a whole-number
  percentage, and each one is simulated here and shown to raise the count;
- real rounding confined to smaller precincts fades as the threshold rises, so no reader-facing
  text may call that pattern an artefact of precinct size.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import clean_election, with_rounded_turnout

from psephos._types import Flag
from psephos.methods.integer_pct import CONFOUNDS, integer_excess, integer_excess_by_threshold

ROOT = Path(__file__).resolve().parents[1]
FLAGGED = (Flag.NOTABLE, Flag.STRONG)


def small_precinct_election(n: int = 6000, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """An honest election in which small and institutional precincts are common.

    40 per cent of units have 20 to 249 registered voters at ordinary turnout. 15 per cent are
    institutional precincts at round sizes from 50 to 300 with 90 to 100 per cent turnout, a
    fifth of them at exactly 100 per cent. The rest are ordinary precincts of 250 to 3,000.
    Every count is a binomial draw; nothing is rounded. Returns ``(ballots_cast, registered)``.
    """
    rng = np.random.default_rng(seed)
    n_small = int(0.40 * n)
    n_inst = int(0.15 * n)
    n_rest = n - n_small - n_inst
    small = rng.integers(20, 250, size=n_small)
    inst = rng.choice([50, 100, 150, 200, 250, 300], size=n_inst)
    rest = rng.lognormal(6.8, 0.6, size=n_rest)
    out = (rest < 250) | (rest > 3000)
    while out.any():
        rest[out] = rng.lognormal(6.8, 0.6, size=int(out.sum()))
        out = (rest < 250) | (rest > 3000)
    reg = np.concatenate([small, inst, np.rint(rest).astype(int)])
    rate = np.concatenate(
        [
            np.clip(rng.normal(0.55, 0.10, n_small), 0.05, 0.98),
            rng.uniform(0.90, 1.00, n_inst),
            np.clip(rng.normal(0.55, 0.10, n_rest), 0.05, 0.98),
        ]
    )
    cast = rng.binomial(reg, rate)
    inst_idx = np.arange(n_small, n_small + n_inst)
    full = rng.choice(inst_idx, size=n_inst // 5, replace=False)
    cast[full] = reg[full]
    return cast.astype(float), reg.astype(float)


# ---------------------------------------------------------------- calibration


def test_integer_excess_stays_calibrated_with_many_small_and_round_sized_precincts():
    """Ten honest elections, each tested with no size threshold at all and at 100.

    Measured when this test was written: all twenty checks OK, z between -2.06 and +0.84, mean
    -0.62. The mean is below zero because the plug-in null over-predicts whole numbers for
    precincts near complete turnout, which is the one place it misfits small units, and it
    misfits them in the direction that hides rounding rather than inventing it.
    """
    zs: list[float] = []
    flagged: list[str] = []
    for s in range(10):
        cast, reg = small_precinct_election(seed=1000 + s)
        assert np.mean(reg < 250) > 0.45, "the generator must actually produce small precincts"
        for t in (1, 100):
            f = integer_excess(cast, reg, min_denominator=t, n_mc=200, seed=s, n_boot=0)
            if t == 1:
                assert f.n_used == reg.size, "with no threshold every unit has to be tested"
            zs.append(f.statistic)
            if f.flag in FLAGGED:
                flagged.append(f"seed {s} min_denominator={t}: {f.flag.value} z={f.statistic:+.2f}")

    assert not [x for x in flagged if "strong" in x], f"STRONG on honest data: {flagged}"
    # Twenty checks at a nominal 5 per cent: one flag is expected, three would be a rate the
    # null does not explain.
    assert len(flagged) <= 2, f"honest small precincts flagged: {flagged}"
    assert float(np.mean(zs)) < 0.5, f"mean z {np.mean(zs):+.2f} on honest small precincts"


# ---------------------------------------------------------------- confounds


def _counts_to_the_nearest_ten_at_round_bases(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One station in twenty has a round electorate and reports turnout to the nearest ten."""
    rng = np.random.default_rng(seed)
    df = clean_election(n=4000, seed=seed)
    reg = df["registered"].to_numpy(dtype=float)
    cast = df["ballots_cast"].to_numpy(dtype=float)
    idx = rng.choice(reg.size, size=reg.size // 20, replace=False)
    rate = cast[idx] / reg[idx]
    reg[idx] = rng.choice([500.0, 1000.0], size=idx.size)
    honest = rng.binomial(reg[idx].astype(np.int64), rate)
    cast[idx] = np.minimum(10.0 * np.rint(honest / 10.0), reg[idx])
    return cast, reg


def _back_calculated_from_a_rounded_percentage(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One station in ten has its count rebuilt from its published, rounded turnout."""
    df = with_rounded_turnout(clean_election(n=4000, seed=seed), fraction=0.10, seed=seed)
    return df["ballots_cast"].to_numpy(dtype=float), df["registered"].to_numpy(dtype=float)


def _computed_against_a_quota(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One station in ten reports a whole-number quota of its electorate, not its count."""
    rng = np.random.default_rng(seed)
    df = clean_election(n=4000, seed=seed)
    reg = df["registered"].to_numpy(dtype=float)
    cast = df["ballots_cast"].to_numpy(dtype=float)
    idx = rng.choice(reg.size, size=reg.size // 10, replace=False)
    quota = rng.integers(50, 81, size=idx.size)
    cast[idx] = np.rint(reg[idx] * quota / 100.0)
    return cast, reg


#: Every confound integer_excess prints, keyed by a phrase that identifies it, with a simulation
#: of the mechanism it names. A confound that cannot be simulated raising the count has no
#: business being printed beside a flag.
MECHANISMS: dict[str, Callable[[int], tuple[np.ndarray, np.ndarray]]] = {
    "rounded rather than tallied": _counts_to_the_nearest_ten_at_round_bases,
    "back-calculated from a published percentage": _back_calculated_from_a_rounded_percentage,
    "percentage of a target": _computed_against_a_quota,
}


def test_every_integer_confound_names_a_mechanism_simulated_here():
    for confound in CONFOUNDS:
        named = [phrase for phrase in MECHANISMS if phrase in confound]
        assert len(named) == 1, f"confound with no simulated mechanism: {confound!r}"
    assert len(CONFOUNDS) == len(MECHANISMS)


def test_integer_confounds_do_not_blame_precinct_size():
    """Size is in the null. A small or round-sized precinct cannot raise the count above it."""
    for confound in CONFOUNDS:
        text = confound.lower()
        assert "small precinct" not in text, confound
        assert "institutional" not in text, confound
        assert "threshold" not in text, confound


@pytest.mark.parametrize("phrase", list(MECHANISMS))
def test_each_integer_confound_raises_the_count_above_the_size_fixed_null(phrase):
    cast, reg = MECHANISMS[phrase](7)
    f = integer_excess(cast, reg, n_mc=200, seed=0, n_boot=0)
    assert f.flag in FLAGGED, f"{phrase}: {f.flag.value}, {f.title}"
    assert f.effect > 0
    assert any(phrase in c for c in f.confounds), f.confounds


def test_a_round_electorate_alone_does_not_raise_the_count():
    """What the first confound says it excludes: round bases with honestly tallied counts.

    The null takes the base as given, so a station of exactly 500 or 1,000 is expected to land
    on a whole number as often as it does. This is why a round-electorate institutional
    precinct is not printed as an explanation.
    """
    rng = np.random.default_rng(7)
    df = clean_election(n=4000, seed=7)
    reg = df["registered"].to_numpy(dtype=float)
    cast = df["ballots_cast"].to_numpy(dtype=float)
    idx = rng.choice(reg.size, size=reg.size // 5, replace=False)
    rate = cast[idx] / reg[idx]
    reg[idx] = rng.choice([500.0, 1000.0], size=idx.size)
    cast[idx] = rng.binomial(reg[idx].astype(np.int64), rate)
    f = integer_excess(cast, reg, n_mc=200, seed=0, n_boot=0)
    assert f.flag is Flag.OK, f.title


# ---------------------------------------------------------------- the sweep


def _round_some_of_band(
    cast: np.ndarray, reg: np.ndarray, lo: int, hi: int, fraction: float, seed: int
) -> np.ndarray:
    """Move a fraction of the units with lo <= registered < hi onto a whole-number turnout."""
    rng = np.random.default_rng(seed)
    out = cast.copy()
    band = np.flatnonzero((reg >= lo) & (reg < hi))
    idx = rng.choice(band, size=int(fraction * band.size), replace=False)
    target = np.round(100.0 * out[idx] / reg[idx])
    out[idx] = np.clip(np.rint(reg[idx] * target / 100.0), 0, reg[idx])
    return out


def test_real_rounding_in_smaller_precincts_fades_as_the_threshold_rises():
    """The pattern the sweep used to be read as an artefact of small precincts, made by rounding.

    Turnout is rounded in 30 per cent of the 250-499 voter precincts of an honest election and
    nowhere else. The excess is there at 250 and gone at 1,000, because the 1,000 threshold has
    dropped every rounded unit, not because the excess was arithmetic.
    """
    cast, reg = small_precinct_election(seed=1000)
    rounded = _round_some_of_band(cast, reg, 250, 500, 0.30, seed=2000)
    at_250, at_1000 = integer_excess_by_threshold(
        rounded, reg, thresholds=(250, 1000), n_mc=200, seed=0, n_boot=0
    )
    assert at_250.flag in FLAGGED, at_250.title
    assert at_250.effect > 0
    assert at_1000.flag is Flag.OK, at_1000.title
    # The 1,000 threshold sees exactly the units an unrounded run sees.
    clean_1000 = integer_excess(cast, reg, min_denominator=1000, n_mc=200, seed=0, n_boot=0)
    assert at_1000.details["observed"] == clean_1000.details["observed"]


# ---------------------------------------------------------------- the reading guidance

#: Where the reading rule for the sweep, and the confounds, are written for a reader.
READING = (
    "INTERPRETATION.md",
    "README.md",
    "src/psephos/report.py",
    "src/psephos/plots.py",
    "src/psephos/methods/integer_pct.py",
    "tests/test_audit.py",
)

#: Ways of saying that an excess which fades at higher thresholds is an artefact of small
#: precincts, which the size-fixed null rules out.
SMALL_PRECINCT_ARTEFACT = (
    r"(artefact|artifact)s? of small precincts",
    r"small[- ]precinct (artefact|artifact)",
    r"signal about small precincts",
    r"lives in small precincts",
    r"faked by small precincts",
    r"survives raising the threshold",
    r"small precincts, where whole-number percentages are common",
)


def _prose(path: str) -> str:
    """The file's text with wrapping undone: implicit string concatenation joined, comment
    markers at line starts dropped, whitespace collapsed, lower-cased."""
    text = (ROOT / path).read_text(encoding="utf-8")
    text = re.sub(r'"\s*\n\s*"', "", text)
    text = re.sub(r"\n\s*#", " ", text)
    return re.sub(r"\s+", " ", text).lower()


@pytest.mark.parametrize("path", READING)
def test_no_reading_guidance_calls_a_fading_excess_a_small_precinct_artefact(path):
    prose = _prose(path)
    found = [p for p in SMALL_PRECINCT_ARTEFACT if re.search(p, prose)]
    assert not found, f"{path} still reads a fading excess as a small-precinct artefact: {found}"
