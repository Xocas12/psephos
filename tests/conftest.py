"""Synthetic election generators.

Everything here is fabricated. The point of the clean generator is that psephos must NOT flag
it: a tool whose false-positive rate is unknown is worse than no tool, because it launders a
prior into a number. The dirty generators inject a known, named distortion so a test can assert
that the check finds the thing it claims to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def clean_election(
    n: int = 4000,
    *,
    seed: int = 11,
    min_size: int = 300,
    max_size: int = 3000,
    turnout_mean: float = 0.55,
    winner_mean: float = 0.45,
) -> pd.DataFrame:
    """An election with no rounding, no stuffing and no invented numbers.

    Precinct sizes are lognormal, turnout and party support vary smoothly between precincts,
    and every count is a binomial draw. Turnout and the winner's share are independent here, so
    the dependence check should also come back quiet.
    """
    rng = np.random.default_rng(seed)
    size = np.clip(rng.lognormal(mean=6.8, sigma=0.6, size=n), min_size, max_size)
    registered = np.rint(size).astype(int)

    turnout_rate = np.clip(rng.normal(turnout_mean, 0.10, size=n), 0.05, 0.98)
    cast = rng.binomial(registered, turnout_rate)

    winner_rate = np.clip(rng.normal(winner_mean, 0.12, size=n), 0.02, 0.95)
    winner = rng.binomial(np.maximum(cast, 0), winner_rate)
    rest = cast - winner
    second_rate = np.clip(rng.normal(0.55, 0.10, size=n), 0.05, 0.95)
    second = rng.binomial(np.maximum(rest, 0), second_rate)
    third = rest - second

    return pd.DataFrame(
        {
            "precinct": np.arange(1, n + 1),
            "region": rng.integers(1, 12, size=n),
            "registered": registered,
            "ballots_cast": cast,
            "party_incumbent": winner,
            "party_second": second,
            "party_third": third,
        }
    )


def with_rounded_turnout(
    df: pd.DataFrame, *, fraction: float = 0.15, seed: int = 5
) -> pd.DataFrame:
    """Move a fraction of precincts onto a whole-number turnout percentage.

    This is the mechanism the integer-percentage check exists to find: someone reports a target
    rather than a count. The ballots are adjusted, not invented from nothing, so totals stay
    coherent and the integrity checks stay quiet.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), size=int(fraction * len(out)), replace=False)
    reg = out.loc[out.index[idx], "registered"].to_numpy(dtype=float)
    cast = out.loc[out.index[idx], "ballots_cast"].to_numpy(dtype=float)
    target_pct = np.round(100.0 * cast / reg)
    new_cast = np.rint(reg * target_pct / 100.0).astype(int)
    new_cast = np.clip(new_cast, 0, reg.astype(int))
    delta = new_cast - cast.astype(int)
    out.loc[out.index[idx], "ballots_cast"] = new_cast
    # Push the change into the winner so the parties still sum to the ballots counted.
    winner = out.loc[out.index[idx], "party_incumbent"].to_numpy(dtype=int) + delta
    out.loc[out.index[idx], "party_incumbent"] = np.maximum(winner, 0)
    return out


def with_stuffing(
    df: pd.DataFrame, *, fraction: float = 0.12, boost: float = 0.35, seed: int = 7
) -> pd.DataFrame:
    """Add votes for the incumbent in a subset of precincts, raising turnout and share together.

    This is what the turnout-share dependence check exists to find.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), size=int(fraction * len(out)), replace=False)
    reg = out.loc[out.index[idx], "registered"].to_numpy(dtype=float)
    cast = out.loc[out.index[idx], "ballots_cast"].to_numpy(dtype=float)
    room = np.maximum(reg - cast, 0)
    added = np.rint(room * boost).astype(int)
    out.loc[out.index[idx], "ballots_cast"] = (cast + added).astype(int)
    out.loc[out.index[idx], "party_incumbent"] = (
        out.loc[out.index[idx], "party_incumbent"].to_numpy(dtype=int) + added
    )
    return out


def with_invented_last_digits(
    df: pd.DataFrame, *, fraction: float = 0.3, seed: int = 9
) -> pd.DataFrame:
    """Replace the incumbent's count in a subset with a human-looking number.

    People writing numbers avoid repeated digits and favour some endings, so the last digit
    stops being uniform.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), size=int(fraction * len(out)), replace=False)
    counts = out.loc[out.index[idx], "party_incumbent"].to_numpy(dtype=int)
    # A lumpy last-digit distribution: heavy on 0 and 5, light on 1 and 9.
    weights = np.array([0.22, 0.03, 0.10, 0.11, 0.10, 0.20, 0.09, 0.08, 0.04, 0.03])
    new_last = rng.choice(10, size=len(counts), p=weights)
    new_counts = (counts // 10) * 10 + new_last
    cast = out.loc[out.index[idx], "ballots_cast"].to_numpy(dtype=int)
    out.loc[out.index[idx], "party_incumbent"] = np.clip(new_counts, 0, cast)
    return out


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return clean_election()


@pytest.fixture
def column_map():
    from psephos.schema import ColumnMap

    return ColumnMap(
        votes={
            "party_incumbent": "party_incumbent",
            "party_second": "party_second",
            "party_third": "party_third",
        },
        registered="registered",
        ballots_cast="ballots_cast",
        precinct_id="precinct",
        region="region",
    )


@pytest.fixture
def clean_data(clean_df, column_map):
    from psephos.schema import ElectionData

    return ElectionData(frame=clean_df, columns=column_map, source="synthetic-clean")
