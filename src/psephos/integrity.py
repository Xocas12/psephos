"""Structural checks that run before any statistics.

These are not anomaly detection. They ask whether the table is internally coherent, and they
exist because most surprising statistical results in election data turn out to be a scraping
error, a column-order change between elections, or a units mismatch. Run them first, and take
a failure here as a fact about the file rather than about the election.
"""

from __future__ import annotations

import numpy as np

from psephos._types import Finding, Flag
from psephos.schema import ElectionData


def check_duplicate_units(data: ElectionData) -> Finding:
    """Precinct identifiers should be unique. Duplicates usually mean a partial re-scrape."""
    if not data.columns.precinct_id:
        return Finding(
            check="duplicate_units",
            title="No precinct identifier column is mapped, so duplicates cannot be checked.",
            flag=Flag.NOT_APPLICABLE,
        )
    keys = [data.columns.precinct_id]
    for role in ("region", "district"):
        col = getattr(data.columns, role)
        if col:
            keys.insert(0, col)
    dup = int(data.frame.duplicated(subset=keys).sum())
    return Finding(
        check="duplicate_units",
        title=(f"{dup} duplicate rows on {keys}." if dup else f"No duplicate rows on {keys}."),
        flag=Flag.STRONG if dup else Flag.OK,
        statistic=float(dup),
        n_used=data.n,
        confounds=(
            ["A legitimately repeated identifier across regions when the key is incomplete."]
            if dup
            else []
        ),
        details={"key": keys, "n_duplicates": dup},
    )


def check_negative_counts(data: ElectionData) -> Finding:
    """No count should be negative."""
    cols = list(data.columns.votes.values())
    for role in ("registered", "ballots_cast", "invalid"):
        col = getattr(data.columns, role)
        if col:
            cols.append(col)
    if not cols:
        return Finding(
            check="negative_counts",
            title="No numeric columns are mapped.",
            flag=Flag.NOT_APPLICABLE,
        )
    block = data.frame[cols].to_numpy(dtype=float)
    bad = int(np.nansum(block < 0))
    return Finding(
        check="negative_counts",
        title=f"{bad} negative counts." if bad else "No negative counts.",
        flag=Flag.STRONG if bad else Flag.OK,
        statistic=float(bad),
        n_used=data.n,
        confounds=["A sentinel value such as -1 used for missing data."] if bad else [],
        details={"columns_checked": cols},
    )


def check_turnout_over_100(data: ElectionData) -> Finding:
    """Ballots counted should not exceed registered voters."""
    try:
        turnout = data.turnout()
    except Exception as exc:
        return Finding(
            check="turnout_over_100",
            title=f"Turnout cannot be computed: {exc}",
            flag=Flag.NOT_APPLICABLE,
        )
    finite = np.isfinite(turnout)
    over = int((finite & (turnout > 100.0 + 1e-9)).sum())
    worst = float(np.nanmax(turnout)) if finite.any() else float("nan")
    return Finding(
        check="turnout_over_100",
        title=(
            f"{over} units report more ballots than registered voters (highest {worst:.1f} per cent)."
            if over
            else "No unit reports more ballots than registered voters."
        ),
        flag=Flag.STRONG if over else Flag.OK,
        statistic=float(over),
        n_used=int(finite.sum()),
        n_excluded=int((~finite).sum()),
        confounds=(
            [
                "Rules allowing registration on polling day, which make turnout above 100 "
                "per cent of the pre-election roll legitimate.",
                "A denominator column that means something other than registered voters.",
            ]
            if over
            else []
        ),
        details={"max_turnout_pct": worst, "n_over": over},
    )


def check_votes_exceed_ballots(data: ElectionData) -> Finding:
    """Contestant votes plus invalid ballots should not exceed the ballots counted."""
    if not data.columns.ballots_cast or not data.columns.votes:
        return Finding(
            check="votes_exceed_ballots",
            title="Needs both a ballots-cast column and contestant columns.",
            flag=Flag.NOT_APPLICABLE,
        )
    counted = data.frame[data.columns.ballots_cast].to_numpy(dtype=float)
    used = data.votes_frame().sum(axis=1).to_numpy(dtype=float)
    if data.columns.invalid:
        used = used + data.frame[data.columns.invalid].to_numpy(dtype=float)
    finite = np.isfinite(counted) & np.isfinite(used)
    bad = int((finite & (used > counted + 1e-6)).sum())
    return Finding(
        check="votes_exceed_ballots",
        title=(
            f"{bad} units where votes plus invalid ballots exceed ballots counted."
            if bad
            else "Votes never exceed ballots counted."
        ),
        flag=Flag.STRONG if bad else Flag.OK,
        statistic=float(bad),
        n_used=int(finite.sum()),
        n_excluded=int((~finite).sum()),
        confounds=(
            [
                "A multi-vote ballot, where one ballot carries several contestant votes.",
                "A ballots-cast column that counts something narrower than all ballots.",
            ]
            if bad
            else []
        ),
        details={"n_bad": bad},
    )


def check_missing(data: ElectionData) -> Finding:
    """How much of the mapped data is missing."""
    cols = list(data.columns.votes.values())
    for role in ("registered", "ballots_cast", "invalid"):
        col = getattr(data.columns, role)
        if col:
            cols.append(col)
    if not cols:
        return Finding(
            check="missing", title="No numeric columns are mapped.", flag=Flag.NOT_APPLICABLE
        )
    block = data.frame[cols]
    per_col = block.isna().sum()
    total = int(per_col.sum())
    worst = str(per_col.idxmax()) if len(per_col) else ""
    return Finding(
        check="missing",
        title=(
            f"{total} missing values across {len(cols)} mapped columns, most in {worst!r}."
            if total
            else "No missing values in the mapped columns."
        ),
        flag=Flag.NOTABLE if total else Flag.OK,
        statistic=float(total),
        n_used=data.n,
        confounds=["Precincts that reported nothing, which is a coverage fact, not an anomaly."]
        if total
        else [],
        details={"per_column": {str(k): int(v) for k, v in per_col.items()}},
    )


#: Every integrity check, in the order the report shows them.
INTEGRITY_CHECKS = (
    check_duplicate_units,
    check_negative_counts,
    check_missing,
    check_turnout_over_100,
    check_votes_exceed_ballots,
)


def run_integrity(data: ElectionData) -> list[Finding]:
    """Run every structural check. Never raises: a check that cannot run reports that."""
    out: list[Finding] = []
    for fn in INTEGRITY_CHECKS:
        try:
            out.append(fn(data))
        except Exception as exc:
            out.append(
                Finding(
                    check=fn.__name__.removeprefix("check_"),
                    title=f"Check failed to run: {type(exc).__name__}: {exc}",
                    flag=Flag.NOT_APPLICABLE,
                )
            )
    return out
