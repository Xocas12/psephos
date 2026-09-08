"""The relationship between turnout and vote share.

Where votes are added to a winner's pile, both the turnout and the winner's share rise in the
same precincts, so the two become correlated in a way that honest variation does not usually
produce. In the two-dimensional histogram this shows as a tail reaching toward the corner where
both are 100 per cent.

What is implemented here is the DEPENDENCE, measured three ways: a rank correlation, the
winner's share in the top turnout decile against the rest, and the share of the winner's total
vote that sits in high-turnout units. What is NOT implemented is the estimator that converts
that dependence into a count of anomalous votes, because that construction has not been checked
against its source and inventing one would put an unearned number into a serious claim. Issue #9
tracks it.

A large positive dependence is not by itself evidence of anything. Turnout and partisanship are
genuinely correlated in most countries: rural areas often have both higher turnout and stronger
support for incumbents. The confounds list says so and the report prints it.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from psephos._types import Finding, Flag

CONFOUNDS = [
    "Genuine geographic correlation between turnout and partisanship. Rural and small-town "
    "areas frequently have both higher turnout and different politics, which produces this "
    "dependence with nobody adding a single vote.",
    "Differential mobilisation: a well-organised campaign raises both its own turnout and its "
    "own share in the same places, which is what campaigns are for.",
    "Compositional effects, where units differ in the age or ethnic composition that drives "
    "both quantities.",
]


def turnout_share_dependence(
    turnout_pct: np.ndarray,
    share_pct: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    slice_name: str = "all",
    winner: str = "the leading contestant",
) -> Finding:
    """Measure how strongly the winner's share rises with turnout."""
    t = np.asarray(turnout_pct, dtype=float)
    s = np.asarray(share_pct, dtype=float)
    if t.shape != s.shape:
        raise ValueError(f"turnout and share differ in shape: {t.shape} vs {s.shape}")
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)

    usable = np.isfinite(t) & np.isfinite(s) & np.isfinite(w) & (t >= 0) & (t <= 100)
    n_used = int(usable.sum())
    n_excluded = int(usable.size - n_used)

    if n_used < 100:
        return Finding(
            check="turnout_share_dependence",
            title=f"Only {n_used} usable units, too few to measure this dependence.",
            flag=Flag.UNDERPOWERED,
            n_used=n_used,
            n_excluded=n_excluded,
            slice_name=slice_name,
        )

    tt, ss, ww = t[usable], s[usable], w[usable]
    rho, pvalue = stats.spearmanr(tt, ss)
    rho = float(rho)
    pvalue = float(pvalue)

    cut = float(np.quantile(tt, 0.9))
    top = tt >= cut
    if top.sum() == 0 or (~top).sum() == 0:
        gap = float("nan")
    else:
        # Vote-weighted, so a handful of tiny precincts cannot drive the number.
        gap = float(np.average(ss[top], weights=ww[top]) - np.average(ss[~top], weights=ww[~top]))

    winner_votes = ss / 100.0 * ww
    total_winner = float(winner_votes.sum())
    share_in_top = (
        float(winner_votes[top].sum() / total_winner) if total_winner > 0 else float("nan")
    )

    if pvalue <= 0.001 and abs(rho) >= 0.3:
        flag = Flag.STRONG
    elif pvalue <= 0.05 and abs(rho) >= 0.1:
        flag = Flag.NOTABLE
    else:
        flag = Flag.OK

    title = (
        f"Share for {winner} rises with turnout: Spearman rho {rho:+.3f}. "
        f"In the top turnout decile (turnout at or above {cut:.1f} per cent) the share is "
        f"{gap:+.1f} points {'higher' if gap >= 0 else 'lower'} than elsewhere, and "
        f"{100 * share_in_top:.1f} per cent of that contestant's votes are cast there."
    )

    return Finding(
        check="turnout_share_dependence",
        title=title,
        flag=flag,
        statistic=rho,
        pvalue=pvalue,
        effect=gap,
        n_used=n_used,
        n_excluded=n_excluded,
        slice_name=slice_name,
        confounds=CONFOUNDS if flag in (Flag.NOTABLE, Flag.STRONG) else [],
        details={
            "spearman_rho": rho,
            "top_decile_cut_pct": cut,
            "share_gap_points": gap,
            "winner_votes_in_top_decile": share_in_top,
            "winner": winner,
            "no_anomalous_vote_estimate": (
                "psephos deliberately does not convert this dependence into a count of "
                "anomalous votes. See issue #9."
            ),
        },
    )


def turnout_roundness(
    turnout_pct: np.ndarray,
    *,
    slice_name: str = "all",
) -> Finding:
    """Descriptive: how turnout is distributed across the range, and where it clusters.

    Reports the deciles and the share of units in the top few percentage points. It is a
    description rather than a test, because a second mode near complete turnout has honest
    causes as often as not, and testing for one without the joint distribution overstates what
    a single histogram can show. Issue #10 covers the joint test.
    """
    t = np.asarray(turnout_pct, dtype=float)
    usable = np.isfinite(t) & (t >= 0) & (t <= 100)
    n_used = int(usable.sum())
    if n_used < 50:
        return Finding(
            check="turnout_distribution",
            title=f"Only {n_used} usable turnout values.",
            flag=Flag.UNDERPOWERED,
            n_used=n_used,
            n_excluded=int(usable.size - n_used),
            slice_name=slice_name,
        )
    tt = t[usable]
    deciles = np.quantile(tt, np.arange(0, 11) / 10.0)
    near_full = float(np.mean(tt >= 95.0))
    return Finding(
        check="turnout_distribution",
        title=(
            f"Turnout spans {tt.min():.1f} to {tt.max():.1f} per cent, median {np.median(tt):.1f}; "
            f"{100 * near_full:.2f} per cent of units report 95 per cent or higher."
        ),
        flag=Flag.OK,
        statistic=float(np.median(tt)),
        effect=near_full,
        n_used=n_used,
        n_excluded=int(usable.size - n_used),
        slice_name=slice_name,
        details={
            "deciles": deciles.tolist(),
            "share_at_or_above_95pct": near_full,
            "descriptive_only": True,
        },
    )
