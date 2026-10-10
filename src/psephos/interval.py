"""Confidence intervals for effect sizes, by resampling units.

A point effect with no interval cannot be read. The same 0.4-point excess means one thing
measured on 300 precincts and another on 30,000, and psephos reported both the same way until
this module existed. With 95,000 precincts almost any departure from a null is significant, so
the effect size and its interval are the only part of the output that says whether the departure
matters, and the report leads with them.

What is resampled
-----------------
**Units, not rows within units, and stratified by precinct size.** A precinct is the unit of
observation for every check here, so the interval has to reflect uncertainty about which
precincts were drawn. Resampling within a precinct would answer a different question — how its
voters might have voted — which is not a question psephos asks.

The stratification matters more than it looks. Every percentage effect in psephos depends on
precinct size: a whole-number percentage is common by arithmetic in a 200-voter station and rare
in a 3,000-voter one. An unstratified resample draws size mixes the actual election did not
have, so the interval widens with variation in a quantity that is fixed and known. Drawing each
size band's own count back, with replacement, holds the size mix at the one the estimate came
from and leaves the interval measuring what it should.

What the interval is
--------------------
The percentile bootstrap, shifted by the bootstrap estimate of its own bias: resample,
recompute, take the empirical quantiles, and subtract ``mean(resamples) - estimate`` from both
ends.

The bias term is not a refinement. Two of the three effects here are means, where it is
negligible and the correction does nothing. The third, the last digits' total variation
distance from uniform, is a sum of absolute deviations, and resampling adds noise that can only
push a sum of absolute deviations up. Measured on this repository's own clean generator, the
bootstrap distribution of that distance sits 26 to 118 per cent of the estimate above it at
sample sizes between 500 and 32,000, and the bias does not shrink away. The plain percentile
interval therefore sits above the point estimate and sometimes excludes it outright, which is
not a defect worth shipping: an interval that does not contain the number it is an interval for
tells a reader something false in the most direct way available. If the shift cannot make it
contain the estimate, no interval is reported and the finding says why.

Two further consequences a reader has to know, and which the findings state in their own
details:

- **An interval that excludes zero is not a test.** The p-value in the same finding is the test,
  computed against a stated null. The interval says how precisely the effect is measured, and
  those are different questions. For a statistic that cannot be negative — a total variation
  distance, for instance — the interval never contains zero even on perfectly honest data, so
  reading it as a test would flag every election ever held.
- **The interval is about sampling the precincts, not about the null.** It does not widen to
  cover the possibility that the null model is wrong, which is the dominant uncertainty in this
  whole tool, and no interval can.

Reference: the percentile bootstrap of Efron, B. (1979), "Bootstrap methods: another look at
the jackknife", Annals of Statistics 7(1), 1-26. The stratified version is the standard
extension, resampling within strata at their observed sizes. The bias shift is the plain
first-order correction, not the accelerated interval of Efron (1987): BCa would also handle the
skewness and is the better instrument, and it is not used here because its acceleration term
needs a jackknife over units, which is a second pass whose cost and whose behaviour on a
boundary statistic are worth measuring before adopting rather than assuming.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np

#: Resamples by default, chosen by measuring how much the interval ends move when only the
#: resampling seed changes, as a share of the interval's own width: 7.0 per cent at 200
#: resamples, 4.9 at 500, 3.6 at 1,000 and 3.1 at 2,000. A thousand is where that stops paying,
#: and it keeps the reported ends good to roughly their second significant digit, which is all
#: the precision the effects carry. ``--bootstrap 500`` halves the cost for a point of noise.
DEFAULT_N_BOOT = 1000

#: Below this many usable units, no interval is reported. A percentile interval from a handful of
#: units is a number with a spurious air of precision, and the underpowered flag is the honest
#: answer at that size.
MIN_UNITS_FOR_INTERVAL = 100


def stratum_codes(sizes: np.ndarray | None, n_units: int, edges: Sequence[float]) -> np.ndarray:
    """One integer per unit saying which size band it sits in.

    Units with a missing or non-positive size, and every unit when ``sizes`` is None, land in a
    single band, which makes the resample unstratified. That is the honest fallback for a check
    whose inputs do not carry a size, and the caller records it.
    """
    if sizes is None:
        return np.zeros(n_units, dtype=np.intp)
    s = np.asarray(sizes, dtype=float)
    if s.shape != (n_units,):
        raise ValueError(f"sizes has shape {s.shape}, expected ({n_units},)")
    codes = np.digitize(s, np.asarray(edges, dtype=float)[1:-1], right=False)
    codes = np.where(np.isfinite(s) & (s > 0), codes, -1)
    return codes.astype(np.intp)


def _resample_indices(
    codes: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Iterator[np.ndarray]:
    """Yield ``n_boot`` index arrays, each a stratified resample of the units with replacement.

    Built per stratum and concatenated, so each band contributes exactly as many units as it
    actually has. The order within a resample is therefore grouped by stratum, which no
    statistic here depends on.

    A generator rather than a list on purpose. Holding all the resamples at once would be
    ``n_boot * n_units * 8`` bytes, which is 32 MB on a 4,000-precinct election and 760 MB on a
    national one at the default thousand resamples — for index arrays that are each used once
    and thrown away.
    """
    groups = [g for g in (np.flatnonzero(codes == c) for c in np.unique(codes)) if g.size]
    if len(groups) == 1:
        # The unstratified case, which is also every check without precinct sizes: draw the
        # offsets directly and skip the concatenate.
        only = groups[0]
        for _ in range(n_boot):
            yield only[rng.integers(0, only.size, only.size)]
        return
    for _ in range(n_boot):
        yield np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_units: int,
    *,
    estimate: float,
    sizes: np.ndarray | None = None,
    edges: Sequence[float] = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, np.inf),
    n_boot: int = DEFAULT_N_BOOT,
    level: float = 0.95,
    floor: float | None = None,
    seed: int | None = None,
) -> tuple[float | None, float | None, dict[str, object]]:
    """A percentile interval for ``statistic``, resampling units within size bands.

    Parameters
    ----------
    statistic : callable
        Takes an index array of unit positions and returns the effect computed on those units.
        Called once per resample, so it should be a recount rather than anything expensive.
    n_units : int
        How many units the estimate was computed on. Indices passed to ``statistic`` are in
        ``range(n_units)``.
    estimate : float
        The effect as computed on the real units. Used for the bias shift, and the interval must
        end up containing it or none is returned.
    sizes : array-like, optional
        Precinct size per unit, for the strata. None resamples unstratified and says so in the
        returned details.
    n_boot : int
        Resamples. ``0`` disables the interval, which is how a caller turns it off.
    level : float
        Coverage, so 0.95 gives the 2.5th and 97.5th percentiles.
    floor : float, optional
        Lower bound the statistic cannot go below, such as ``0.0`` for a distance. The shifted
        interval is clamped to it and the clamp is recorded, because an interval reaching below
        an impossible value invites it to be read as a test against that value.

    Returns
    -------
    (ci_low, ci_high, details)
        ``(None, None, details)`` when no interval was computed, with ``details["ci_note"]``
        saying why. ``details`` is meant to be merged into ``Finding.details``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    if n_boot <= 0:
        return None, None, {"ci_note": "intervals disabled (n_boot=0)"}
    if n_units < MIN_UNITS_FOR_INTERVAL:
        return (
            None,
            None,
            {
                "ci_note": (
                    f"no interval: {n_units} units is below the {MIN_UNITS_FOR_INTERVAL} this "
                    "would need to mean anything"
                )
            },
        )

    codes = stratum_codes(sizes, n_units, edges)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i, idx in enumerate(_resample_indices(codes, rng, n_boot)):
        values[i] = statistic(idx)

    finite = values[np.isfinite(values)]
    if finite.size < n_boot // 2:
        return (
            None,
            None,
            {
                "ci_note": (
                    f"no interval: only {finite.size} of {n_boot} resamples produced a finite "
                    "statistic, so the quantiles would be drawn from a biased subset"
                )
            },
        )

    alpha = 1.0 - level
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Shift by the bootstrap's estimate of its own bias. A no-op for a mean; the whole point for
    # a distance, where resampling can only inflate a sum of absolute deviations.
    bias = float(finite.mean()) - float(estimate)
    lo, hi = float(lo) - bias, float(hi) - bias
    clamped = False
    if floor is not None:
        lo, hi = max(lo, floor), max(hi, floor)
        clamped = lo == floor

    if not lo <= estimate <= hi:
        return (
            None,
            None,
            {
                "ci_note": (
                    f"no interval: the bias-shifted interval [{lo:.6g}, {hi:.6g}] does not "
                    f"contain the estimate {estimate:.6g}, so the resampling distribution is "
                    "too skewed for this method and reporting it would state something false"
                )
            },
        )

    n_strata = int(np.unique(codes[codes >= 0]).size)
    details: dict[str, object] = {
        "ci_level": level,
        "ci_method": "bias-shifted percentile bootstrap over units",
        "ci_bias_shift": bias,
        "ci_floored_at": floor if clamped else None,
        "ci_n_boot": n_boot,
        "ci_n_strata": n_strata,
        "ci_stratified_by": "precinct size" if sizes is not None else None,
        "ci_resamples_used": int(finite.size),
        "ci_seed": seed,
        "ci_note": (
            "Resampled units within size bands, so the interval holds the precinct-size mix "
            "fixed at the one observed, then shifted by the bootstrap's own bias. It measures "
            "how precisely the effect is estimated and is NOT a test: the p-value beside it is "
            "the test, and an interval that excludes zero does not add to it."
        ),
    }
    if sizes is None:
        details["ci_note"] += (
            " This check's inputs carry no precinct size, so the resample is unstratified and "
            "the interval may be wider than the design warrants."
        )
    return lo, hi, details


def expected_tvd_under_uniform(
    n_used: int, n_cells: int = 10, *, n_sim: int = 400, seed: int = 0
) -> float:
    """Total variation distance from uniform that honest data of this size produces anyway.

    A total variation distance is non-negative, so its interval never contains zero and
    excluding zero means nothing. What a reader needs instead is the value the statistic takes
    when nothing is wrong, which is not zero and which shrinks as ``1 / sqrt(n)``. Simulated
    rather than taken from the normal approximation, because at the cell counts and sample
    sizes here the approximation is worth less than 400 multinomial draws cost.
    """
    if n_used <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_used, np.full(n_cells, 1.0 / n_cells), size=n_sim)
    props = draws / n_used
    return float(np.mean(0.5 * np.abs(props - 1.0 / n_cells).sum(axis=1)))
