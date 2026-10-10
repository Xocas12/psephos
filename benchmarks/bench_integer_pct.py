"""What the integer-percentage null costs, and whether vectorising the replicate loop helps.

Issue #17 asks for the numbers before anything is changed, so this is a script rather than a
test: timings on a shared runner are a flake, and the point is a measurement a reader can
reproduce and disagree with.

    uv run python benchmarks/bench_integer_pct.py                     # cost of one check
    uv run python benchmarks/bench_integer_pct.py --breakdown         # where the time goes
    uv run python benchmarks/bench_integer_pct.py --compare-blocked   # the vectorisation A/B
    uv run python benchmarks/bench_integer_pct.py --units 20000 --n-mc 200

``--breakdown`` is the one to run first, and it is the only measurement here that does not
depend on how busy the machine is. It splits one check into the binomial sampling, the counting
step and the Python loop overhead. The loop is what issue #17 proposed to vectorise, and the
breakdown bounds what that could ever save.

``--compare-blocked`` tries the vectorisation anyway, interleaving the variants so that machine
drift does not decide the answer, and asserting that every variant produced the identical null.
What the numbers say is recorded in docs/performance.md.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from psephos.methods.integer_pct import integer_excess, integer_excess_by_threshold

DEFAULT_THRESHOLDS = (100, 250, 500, 1000)


def synthetic_election(n_units: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Registered voters and ballots cast for an honest election of ``n_units`` precincts.

    Sizes are lognormal, roughly the shape of a real precinct-size distribution: a long tail of
    large urban stations over a mass of small ones. Turnout is a beta around 60 per cent. No
    rounding signal is injected, because the benchmark measures cost, not detection.
    """
    rng = np.random.default_rng(seed)
    registered = np.clip(np.rint(rng.lognormal(mean=6.7, sigma=0.75, size=n_units)), 20, None)
    rates = rng.beta(6.0, 4.0, size=n_units)
    cast = rng.binomial(registered.astype(np.int64), rates).astype(float)
    return registered, cast


def _loop_null(d: np.ndarray, p: np.ndarray, tol: float, n_mc: int, seed: int) -> np.ndarray:
    """The null as psephos draws it: one replicate at a time."""
    rng = np.random.default_rng(seed)
    d_int = np.rint(d).astype(np.int64)
    null = np.empty(n_mc, dtype=float)
    for i in range(n_mc):
        draw = rng.binomial(d_int, p)
        pct = 100.0 * draw / d
        null[i] = np.count_nonzero(np.abs(pct - np.round(pct)) <= tol)
    return null


def _blocked_null(
    d: np.ndarray, p: np.ndarray, tol: float, n_mc: int, seed: int, rows: int
) -> np.ndarray:
    """The same null, drawn ``rows`` replicates at a time as one two-dimensional array.

    NumPy fills a ``(rows, n_units)`` draw in row-major order from the same stream, so this is
    the same replicates in the same order for the same seed, whatever ``rows`` is. The A/B
    below asserts that, so a timing difference is never a difference in the answer.
    """
    rng = np.random.default_rng(seed)
    d_int = np.rint(d).astype(np.int64)
    n = d_int.size
    null = np.empty(n_mc, dtype=float)
    done = 0
    while done < n_mc:
        take = min(rows, n_mc - done)
        draw = rng.binomial(d_int, p, size=(take, n))
        pct = 100.0 * draw / d
        null[done : done + take] = np.count_nonzero(np.abs(pct - np.round(pct)) <= tol, axis=1)
        done += take
    return null


def _usable(cast: np.ndarray, registered: np.ndarray, min_den: int):
    usable = registered >= min_den
    k, d = cast[usable], registered[usable]
    return d, np.clip(k / d, 0.0, 1.0)


def _compare_blocked(args: argparse.Namespace) -> None:
    registered, cast = synthetic_election(args.units)
    d, p = _usable(cast, registered, 250)
    n = d.size
    print(f"{n} usable units, n_mc={args.n_mc}, {args.repeats} interleaved repeats")
    print("A block of R replicates holds R x n x 16 bytes of working memory.\n")

    variants: list[tuple[str, int | None]] = [("loop (one replicate)", None)]
    variants += [(f"blocked, {r} replicates", r) for r in args.rows]

    reference = _loop_null(d, p, 0.05, args.n_mc, 0)
    times: dict[str, list[float]] = {name: [] for name, _ in variants}
    for _ in range(args.repeats):
        for name, rows in variants:
            t0 = time.perf_counter()
            null = (
                _loop_null(d, p, 0.05, args.n_mc, 0)
                if rows is None
                else _blocked_null(d, p, 0.05, args.n_mc, 0, rows)
            )
            times[name].append(time.perf_counter() - t0)
            # The variants must be the same replicates, or the timing comparison is meaningless.
            assert np.array_equal(null, reference), f"{name} changed the null"

    print(f"{'variant':26} {'peak MB':>8} {'best':>8} {'median':>8}  s/replicate")
    for name, rows in variants:
        ts = sorted(times[name])
        peak = (rows or 1) * n * 16 / 1e6
        print(
            f"{name:26} {peak:8.1f} {ts[0]:7.2f}s {ts[len(ts) // 2]:7.2f}s  {ts[0] / args.n_mc:.4f}"
        )
    print("\nEvery variant produced the identical null, so the only difference is cost.")


def _breakdown(args: argparse.Namespace) -> None:
    """Split one check into sampling, counting and loop overhead.

    Timed with a clock either side of each step rather than with a profiler, because a profiler
    adds per-call overhead to exactly the quantity in question. The three shares are measured
    within one run, so a machine that is twice as slow today moves all three together and the
    shares stay comparable.
    """
    registered, cast = synthetic_election(args.units)
    d, p = _usable(cast, registered, 250)
    d_int = np.rint(d).astype(np.int64)
    rng = np.random.default_rng(0)

    t_draw = 0.0
    t_count = 0.0
    start = time.perf_counter()
    for _ in range(args.n_mc):
        a = time.perf_counter()
        draw = rng.binomial(d_int, p)
        b = time.perf_counter()
        pct = 100.0 * draw / d
        np.count_nonzero(np.abs(pct - np.round(pct)) <= 0.05)
        t_draw += b - a
        t_count += time.perf_counter() - b
    total = time.perf_counter() - start
    overhead = total - t_draw - t_count
    n = d_int.size

    print(f"{n} usable units, n_mc={args.n_mc}, {n * args.n_mc:,} binomial draws")
    print(f"  total            {total:8.3f}s")
    print(f"  rng.binomial     {t_draw:8.3f}s  {100 * t_draw / total:5.1f}%")
    print(f"  counting step    {t_count:8.3f}s  {100 * t_count / total:5.1f}%")
    print(f"  loop overhead    {overhead:8.3f}s  {100 * overhead / total:5.2f}%")
    print(f"  per binomial draw {1e9 * t_draw / (n * args.n_mc):.0f} ns")
    print(
        "\nThe loop overhead line is the ceiling on what vectorising the replicate loop can "
        "save.\nThe sampling line is where the time is, and it is spent inside NumPy either way."
    )


def _cost(args: argparse.Namespace) -> None:
    registered, cast = synthetic_election(args.units)
    print(f"synthetic election: {args.units} units, n_mc={args.n_mc}, repeats={args.repeats}")
    print(f"  registered: median {np.median(registered):.0f}, max {registered.max():.0f}")

    single = integer_excess(cast, registered, n_mc=args.n_mc, seed=0)
    print(
        f"  check used {single.n_used} units, excluded {single.n_excluded}, "
        f"flag {single.flag.value}\n"
    )

    def timed(fn, repeats: int) -> tuple[float, float]:
        ts = [0.0] * repeats
        for i in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts[i] = time.perf_counter() - t0
        return min(ts), float(np.mean(ts))

    best, mean = timed(
        lambda: integer_excess(cast, registered, n_mc=args.n_mc, seed=0), args.repeats
    )
    draws = single.n_used * args.n_mc
    print(f"one check          best {best:7.2f}s  mean {mean:7.2f}s  ({draws:,} binomial draws)")
    print(f"                   {1e9 * best / draws:.0f} ns per binomial draw")

    best_s, mean_s = timed(
        lambda: integer_excess_by_threshold(
            cast, registered, thresholds=DEFAULT_THRESHOLDS, n_mc=args.n_mc, seed=0
        ),
        max(1, args.repeats - 2),
    )
    print(f"threshold sweep x4 best {best_s:7.2f}s  mean {mean_s:7.2f}s")
    print(
        "\nThe audit runs the sweep twice (turnout and the winner's share) and then a "
        "stratified pass,\nso a full audit is several times the sweep figure."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", type=int, default=100_000)
    ap.add_argument("--n-mc", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--breakdown",
        action="store_true",
        help="split one check into sampling, counting and Python loop overhead",
    )
    ap.add_argument(
        "--compare-blocked",
        action="store_true",
        help="A/B the replicate loop against blocked two-dimensional draws",
    )
    ap.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[4, 16, 64],
        help="block sizes to compare, in replicates (with --compare-blocked)",
    )
    args = ap.parse_args()
    if args.breakdown:
        _breakdown(args)
    elif args.compare_blocked:
        _compare_blocked(args)
    else:
        _cost(args)


if __name__ == "__main__":
    main()
