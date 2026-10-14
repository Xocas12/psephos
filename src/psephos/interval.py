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
For the effects that are means and shares, :func:`bootstrap_ci`: the percentile bootstrap,
shifted by the bootstrap estimate of its own bias. Resample, recompute, take the empirical
quantiles, and subtract ``mean(resamples) - estimate`` from both ends. For a mean the shift is
negligible and does nothing; it stays as a guard, and if it cannot make the interval contain the
estimate, no interval is reported and the finding says why.

The last digits' total variation distance from uniform does not go through it. A distance is a
sum of absolute deviations, and counting noise can only push one up: the distance counted on
perfectly uniform digits is about ``1.2 / sqrt(n)``, not zero. The plain percentile interval sat
above its own estimate and sometimes excluded it, which is why the shift was added; but the
shift only re-centres the interval on the estimate, and the estimate is the biased number. The
shifted interval covered the value the counted distance takes at that sample size, not the
distance, and covered a true distance of zero in 0 to 3 per cent of draws (issue #29).
:func:`distance_interval` reports the distance less its noise, with an interval that bounds the
true distance from each side by what counting noise can do to the counted one.

Two further consequences a reader has to know, and which the findings state in their own
details:

- **An interval that excludes zero is not a test.** The p-value in the same finding is the test,
  computed against a stated null. The interval says how precisely the effect is measured, and
  those are different questions.
- **The interval is about sampling the precincts, not about the null.** It does not widen to
  cover the possibility that the null model is wrong, which is the dominant uncertainty in this
  whole tool, and no interval can.

Reference: the percentile bootstrap of Efron, B. (1979), "Bootstrap methods: another look at
the jackknife", Annals of Statistics 7(1), 1-26. The stratified version is the standard
extension, resampling within strata at their observed sizes. The bias shift is the plain
first-order correction, not the accelerated interval of Efron (1987): BCa would also handle the
skewness and is the better instrument, and it is not used here because its acceleration term
needs a jackknife over units, which is a second pass whose cost and whose behaviour on a
boundary statistic are worth measuring before adopting rather than assuming. The distance
interval needs no reference beyond the triangle inequality and the standard deviation of a
binomial share; :func:`distance_interval` has the argument.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from statistics import NormalDist

import numpy as np
from scipy import stats

from psephos.size import DEFAULT_EDGES

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
    edges: Sequence[float] = DEFAULT_EDGES,
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
    # Only claim stratification when it actually happened. A per-size-band finding passes the
    # sizes of one band, so every unit lands in the same stratum and the resample is plain:
    # saying "stratified by precinct size" there would describe the argument, not the method.
    stratified = sizes is not None and n_strata > 1
    details: dict[str, object] = {
        "ci_level": level,
        "ci_method": "bias-shifted percentile bootstrap over units",
        "ci_bias_shift": bias,
        "ci_floored_at": floor if clamped else None,
        "ci_n_boot": n_boot,
        "ci_n_strata": n_strata,
        "ci_stratified_by": "precinct size" if stratified else None,
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
    elif not stratified:
        details["ci_note"] += (
            f" Every unit here falls in one size band, so the resample is unstratified in "
            f"effect: {n_strata} band, nothing to hold fixed."
        )
    if finite.size < n_boot:
        # Above the half-way threshold the quantiles are still taken, so say what they were
        # taken from. Silence here would let a partly-undefined statistic read as a clean one.
        details["ci_note"] += (
            f" {n_boot - finite.size} of {n_boot} resamples produced no finite value and were "
            "dropped, so the interval comes from the remainder and is that much less certain "
            "than its width suggests."
        )
    return lo, hi, details


def tvd_noise_bias(observed: np.ndarray, reference: np.ndarray) -> float:
    """How much counting noise adds, on average, to a total variation distance at these shares.

    The expected distance from ``reference`` of a histogram of the same size drawn at the
    observed shares, less the observed distance: the bootstrap's estimate of the bias, computed
    exactly instead of by resampling. Each cell's count in such a draw is binomial, and the
    distance is a sum over cells, so the expectation is a sum of binomial mean absolute
    deviations, using ``|X - c| = (X - c) + 2 * max(c - X, 0)``.

    Exact and seedless on purpose. The effect is this distance less this bias, and an effect that
    moved with the resampling seed, or vanished with ``--no-intervals``, would be a worse number
    than the biased one it replaces.
    """
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(reference, dtype=float)
    n = float(obs.sum())
    if n <= 0:
        return float("nan")
    expected_abs = 0.0
    for count, share in zip(obs, ref, strict=True):
        c = share * n
        below = np.arange(int(np.ceil(c)))
        tail = float(((c - below) * stats.binom.pmf(below, int(n), count / n)).sum())
        expected_abs += (count - c) + 2.0 * tail
    observed_tvd = float(0.5 * np.abs(obs / n - ref).sum())
    return 0.5 * expected_abs / n - observed_tvd


def distance_interval(
    cells: np.ndarray,
    reference: np.ndarray,
    *,
    sizes: np.ndarray | None = None,
    edges: Sequence[float] = DEFAULT_EDGES,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float | None, float | None, dict[str, object]]:
    """A total variation distance from ``reference``, less its noise, with an interval that
    covers the true distance.

    ``cells`` holds one cell index per unit (a last digit, say) and ``reference`` the share each
    cell should get. Returns ``(effect, ci_low, ci_high, details)``: the counted distance less
    :func:`tvd_noise_bias`, floored at zero, and an interval for the distance between the shares
    the units were drawn from and ``reference``. ``(effect, None, None, details)`` when no
    interval was computed, with ``details["ci_note"]`` saying why.

    Why not :func:`bootstrap_ci`. The counted distance ``T`` sits above the true distance
    ``theta`` by about what counting noise adds, which at ``theta = 0`` is 0.038 at 1,000 units.
    A percentile bootstrap shifted by its own bias is centred on ``T``, so it covers the value
    ``T`` takes on average at this sample size, not ``theta``: on last digits with a known
    distance, at 1,000 and 4,000 units, it covered ``theta`` in 0 to 3 per cent of draws at
    ``theta = 0`` and in 10 to 33 per cent with the departure on two digits (issue #29). The
    basic bootstrap, centred on ``2 T - mean(T*)``, did better and still covered only 58 to 60
    per cent at ``theta = 0`` and 70 to 92 per cent from 0.01 to 0.04, because the bootstrap
    estimates the bias at the observed shares, where no cell sits exactly on its reference, and
    the bias is largest where cells do.

    The construction. ``theta`` is the largest ``p(B) - r(B)`` over sets of cells ``B``, reached
    at ``A``, the cells above their reference; ``T`` is the same with the observed shares.

    - Above. ``T`` is at least ``phat(A) - r(A)``, so ``theta - T`` is at most
      ``p(A) - phat(A)``: one share's sampling error, whose standard deviation is at most
      ``1 / (2 sqrt(n))`` whatever ``A`` is. The upper end is ``T`` plus the normal quantile
      times that.
    - Below. ``T - theta`` is at most the distance between the observed and the true shares,
      which is exactly ``T`` when ``theta`` is zero. Its quantile is taken from resamples of the
      units within size bands, as the distance between each resample's shares and the observed
      ones. The lower end is ``T`` less that quantile, floored at zero.

    Each end misses with probability at most ``(1 - level) / 2``, so the interval covers
    ``theta`` with at least ``level`` whatever the shares are, and it reaches zero whenever the
    data are consistent with ``reference``. It is wider than the shifted interval, 0.069 against
    0.046 on uniform digits at 1,000 units, because the lower end has to allow for the case
    where most of the counted distance is noise. Measured coverage was 96 to 100 per cent from
    ``theta = 0`` to 0.08 at 1,000, 4,000 and 20,000 units, with the departure spread or on two
    digits, 200 draws a cell.

    Resamples are drawn as one multinomial per size band, at the band's own count and shares,
    which is the same distribution as resampling the band's units with replacement and recounting
    their cells, without materialising ``n_boot * n_units`` indices.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    ref = np.asarray(reference, dtype=float)
    k = ref.size
    cells = np.asarray(cells, dtype=np.intp)
    n = int(cells.size)
    if n == 0:
        raise ValueError("no units to measure a distance on")
    observed = np.bincount(cells, minlength=k).astype(float)
    if observed.size != k:
        raise ValueError(f"cells must index the {k} reference shares; got index {cells.max()}")
    props = observed / n
    tvd = float(0.5 * np.abs(props - ref).sum())
    bias = tvd_noise_bias(observed, ref)
    effect = max(0.0, tvd - bias)
    details: dict[str, object] = {"tvd_as_counted": tvd, "tvd_noise_bias": bias}

    if n_boot <= 0:
        details["ci_note"] = "intervals disabled (n_boot=0)"
        return effect, None, None, details
    if n < MIN_UNITS_FOR_INTERVAL:
        details["ci_note"] = (
            f"no interval: {n} units is below the {MIN_UNITS_FOR_INTERVAL} this would need to "
            "mean anything"
        )
        return effect, None, None, details

    codes = stratum_codes(sizes, n, edges)
    rng = np.random.default_rng(seed)
    resampled = np.zeros((n_boot, k))
    for code in np.unique(codes):
        members = cells[codes == code]
        band = np.bincount(members, minlength=k)
        resampled += rng.multinomial(members.size, band / members.size, size=n_boot)
    noise = 0.5 * np.abs(resampled / n - props).sum(axis=1)

    alpha = 1.0 - level
    below = float(np.quantile(noise, 1.0 - alpha / 2.0))
    above = NormalDist().inv_cdf(1.0 - alpha / 2.0) * 0.5 / np.sqrt(n)
    lo, hi = max(0.0, tvd - below), tvd + above
    # The effect sits inside by construction: the bias is at most the mean of the same noise
    # whose upper quantile sets the lower end. Only a size banding that leaves almost no noise
    # in the resamples could break that, and a lower end moved down is still a valid one.
    lo = min(lo, effect)

    n_strata = int(np.unique(codes[codes >= 0]).size)
    stratified = sizes is not None and n_strata > 1
    details.update(
        {
            "ci_level": level,
            "ci_method": "noise-bounded interval for a total variation distance",
            "ci_noise_quantile": below,
            "ci_upper_margin": float(above),
            "ci_reaches_zero": lo == 0.0,
            "ci_n_boot": n_boot,
            "ci_n_strata": n_strata,
            "ci_stratified_by": "precinct size" if stratified else None,
            "ci_resamples_used": n_boot,
            "ci_seed": seed,
            "ci_note": (
                "An interval for the distance itself, not for the value the counted distance "
                "takes at this sample size, so it reaches zero when the data are consistent "
                "with the reference shares. Its lower end allows for as much counting noise as "
                "resampling the units within size bands produces, its upper end for one share's "
                "sampling error. It is NOT a test: the p-value beside it is the test."
            ),
        }
    )
    if sizes is None:
        details["ci_note"] += (
            " This check's inputs carry no precinct size, so the resample is unstratified."
        )
    elif not stratified:
        details["ci_note"] += (
            f" Every unit here falls in one size band, so the resample is unstratified in "
            f"effect: {n_strata} band, nothing to hold fixed."
        )
    return effect, lo, hi, details


def expected_tvd_under_uniform(
    n_used: int, n_cells: int = 10, *, n_sim: int = 400, seed: int = 0
) -> float:
    """Total variation distance from uniform that honest data of this size produces anyway.

    Counting noise alone keeps the counted distance above zero on perfectly uniform digits, at
    about this value, which shrinks as ``1 / sqrt(n)``. Printed beside the counted distance so a
    reader can see how much of it noise accounts for. Simulated rather than taken from the
    normal approximation, because at the cell counts and sample sizes here the approximation is
    worth less than 400 multinomial draws cost.
    """
    if n_used <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_used, np.full(n_cells, 1.0 / n_cells), size=n_sim)
    props = draws / n_used
    return float(np.mean(0.5 * np.abs(props - 1.0 / n_cells).sum(axis=1)))
