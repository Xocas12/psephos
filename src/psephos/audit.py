"""Run every applicable check and assemble the report.

The audit is deliberately opinionated about two things. It always runs the integrity checks
first, because a structural problem explains most surprising statistics. And it always runs the
percentage checks across a sweep of size thresholds rather than at one, because a single
threshold is a parameter that can be tuned until the answer is interesting.
"""

from __future__ import annotations

from dataclasses import replace

from psephos._types import AuditReport, Finding, Flag
from psephos.correction import apply_correction
from psephos.integrity import run_integrity
from psephos.methods import digits, integer_pct, turnout
from psephos.schema import ElectionData, SchemaError
from psephos.size import size_warning, stratify

#: Size thresholds the integer-percentage sweep uses.
DEFAULT_THRESHOLDS: tuple[int, ...] = (100, 250, 500, 1000)


def _safe(fn, *args, check: str, hypothesis: str | None = None, **kwargs) -> Finding:
    """Run one check; turn any failure into a reported finding rather than a crash.

    ``hypothesis`` tags the finding as a re-test of one hypothesis when the audit runs that
    hypothesis on several slices, which is what the multiple-testing correction counts once;
    see docs/multiple_testing.md.
    """
    try:
        finding = fn(*args, **kwargs)
    except SchemaError as exc:
        finding = Finding(check=check, title=f"Not run: {exc}", flag=Flag.NOT_APPLICABLE)
    except Exception as exc:
        finding = Finding(
            check=check,
            title=f"Check failed: {type(exc).__name__}: {exc}",
            flag=Flag.NOT_APPLICABLE,
        )
    if hypothesis is not None:
        return replace(finding, hypothesis=hypothesis)
    return finding


def audit(
    data: ElectionData,
    *,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    n_mc: int = 500,
    seed: int | None = 0,
    by_size: bool = True,
) -> AuditReport:
    """Audit one precinct table.

    Returns an :class:`~psephos._types.AuditReport`. Nothing here raises on statistical
    grounds: a check that cannot run says so and the audit continues.

    Before returning, the audit applies the multiple-testing correction of
    :mod:`psephos.correction`: every finding keeps its raw p-value and gains an adjusted one
    beside it, no flag is raised or withdrawn by the correction, and the number of hypotheses
    tested is recorded in the report meta. What counts as one family is argued in
    docs/multiple_testing.md.
    """
    report = AuditReport(
        source=data.source,
        n_units=data.n,
        meta={
            "columns": data.columns.to_dict(),
            "settings": {
                "thresholds": list(thresholds),
                "n_mc": n_mc,
                "seed": seed,
                "by_size": by_size,
            },
        },
    )
    report.integrity = run_integrity(data)

    has_registered = bool(data.columns.registered)
    has_votes = bool(data.columns.votes)

    if not has_votes:
        report.add(
            Finding(
                check="setup",
                title="No contestant columns are mapped, so no statistical check can run.",
                flag=Flag.NOT_APPLICABLE,
            )
        )
        return apply_correction(report)

    winner = data.winner_label()
    report.meta["winner"] = winner

    # ---------------------------------------------------------------- percentage checks
    if has_registered:
        reg = data.registered()
        cast = data.total_votes()
        note = size_warning(reg, min(thresholds))
        if note:
            report.meta["size_note"] = note

        for t in thresholds:
            report.add(
                _safe(
                    integer_pct.integer_excess,
                    cast,
                    reg,
                    check="integer_percentage",
                    min_denominator=t,
                    n_mc=n_mc,
                    seed=seed,
                    slice_name=f"turnout, min_denominator={t}",
                    label="turnout",
                    # One hypothesis at four thresholds, counted once by the correction.
                    hypothesis="integer_pct:turnout",
                )
            )

        winner_votes = data.frame[data.columns.votes[winner]].to_numpy(dtype=float)
        valid = data.votes_frame().sum(axis=1).to_numpy(dtype=float)
        for t in thresholds:
            report.add(
                _safe(
                    integer_pct.integer_excess,
                    winner_votes,
                    valid,
                    check="integer_percentage",
                    min_denominator=t,
                    n_mc=n_mc,
                    seed=seed,
                    slice_name=f"{winner} share, min_denominator={t}",
                    label=f"{winner} share",
                    hypothesis=f"integer_pct:{winner} share",
                )
            )

        report.add(
            _safe(
                turnout.turnout_share_dependence,
                data.turnout(),
                data.share(winner, of="valid"),
                check="turnout_share_dependence",
                weights=valid,
                winner=winner,
                hypothesis="turnout_share_dependence",
            )
        )
        report.add(_safe(turnout.turnout_roundness, data.turnout(), check="turnout_distribution"))
    else:
        report.add(
            Finding(
                check="integer_percentage",
                title=(
                    "No registered-voters column is mapped, so no percentage check can run. "
                    "This is the strongest family psephos has; map it if the data has it."
                ),
                flag=Flag.NOT_APPLICABLE,
            )
        )

    # ---------------------------------------------------------------- digit checks
    # One hypothesis per contestant column: each count's digits are a separate question, and
    # they are the members of the family closest to independent.
    for label, col in data.columns.votes.items():
        counts = data.frame[col].to_numpy(dtype=float)
        report.add(
            _safe(
                digits.last_digit_uniformity,
                counts,
                check="last_digit",
                slice_name=label,
                label=label,
                hypothesis=f"last_digit:{label}",
            )
        )
    report.add(
        _safe(
            digits.last_two_digit_pairs,
            data.frame[data.columns.votes[winner]].to_numpy(dtype=float),
            check="last_two_digits",
            slice_name=winner,
            label=winner,
            hypothesis=f"last_two_digits:{winner}",
        )
    )

    # ---------------------------------------------------------------- stratified view
    if by_size and has_registered:
        reg = data.registered()
        cast = data.total_votes()
        for stratum in stratify(reg):
            if stratum.n < 200:
                continue
            report.add(
                _safe(
                    integer_pct.integer_excess,
                    cast[stratum.mask],
                    reg[stratum.mask],
                    check="integer_percentage",
                    min_denominator=1,
                    n_mc=n_mc,
                    seed=seed,
                    slice_name=f"turnout, {stratum.name}",
                    label="turnout",
                    # Where the signal lives, not a new hypothesis: the same turnout claim
                    # re-tested within size bands, counted once by the correction.
                    hypothesis="integer_pct:turnout",
                )
            )

    return apply_correction(report)
