"""Tests for the progress callback, and for the claim docs/performance.md rests on.

Issue #17 asks for speed without trading correctness. These tests pin behaviour, not timing: a
timing assertion on a shared runner is a flake, so the measurement lives in
benchmarks/bench_integer_pct.py and its result in docs/performance.md.

What is worth pinning is the premise of that document. It reports that drawing the null in
blocks is not faster, and it can only say "not faster" rather than "different" because a
blocked draw is the same replicates as the loop. If NumPy ever changed that fill order, the
document's comparison would be between two different nulls and its conclusion would be void.
"""

from __future__ import annotations

import numpy as np
import pytest

from psephos.methods.integer_pct import integer_excess


@pytest.fixture
def election() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    registered = np.clip(np.rint(rng.lognormal(6.6, 0.7, 1200)), 50, None)
    cast = rng.binomial(registered.astype(np.int64), rng.beta(6.0, 4.0, 1200)).astype(float)
    return cast, registered


@pytest.mark.parametrize("rows", [2, 7, 50])
def test_a_blocked_binomial_draw_is_the_same_stream_as_the_loop(rows):
    """The premise of the benchmark in docs/performance.md.

    NumPy fills a ``(rows, n)`` draw in row-major order from the same stream, so blocking is a
    cost change and never an answer change. If this fails, the benchmark is comparing two
    different nulls and the document's conclusion does not follow from it.
    """
    n, n_mc = 300, 50
    d = np.random.default_rng(1).integers(200, 3000, n).astype(np.int64)
    p = np.random.default_rng(2).uniform(0.2, 0.9, n)

    rng = np.random.default_rng(99)
    loop = np.stack([rng.binomial(d, p) for _ in range(n_mc)])

    rng = np.random.default_rng(99)
    blocks = []
    done = 0
    while done < n_mc:
        take = min(rows, n_mc - done)
        blocks.append(rng.binomial(d, p, size=(take, n)))
        done += take

    assert np.array_equal(np.concatenate(blocks), loop)


def test_progress_is_called_and_reaches_the_total(election):
    cast, registered = election
    seen: list[tuple[int, int]] = []
    integer_excess(
        cast, registered, n_mc=200, seed=0, progress=lambda done, total: seen.append((done, total))
    )
    assert len(seen) > 1, "a 200-replicate run should report more than once"
    assert seen[-1] == (200, 200), "the last report must say the work finished"
    assert all(total == 200 for _, total in seen)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert all(1 <= d <= 200 for d, _ in seen)


def test_progress_reaches_the_total_even_when_n_mc_is_tiny(election):
    """A final report is what clears the progress line, so it cannot depend on the interval."""
    cast, registered = election
    seen: list[tuple[int, int]] = []
    integer_excess(cast, registered, n_mc=3, seed=0, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (3, 3)


def test_progress_does_not_change_the_finding(election):
    """It is a callback on the way past, not part of the statistic."""
    cast, registered = election
    with_cb = integer_excess(cast, registered, n_mc=120, seed=5, progress=lambda d, t: None)
    without = integer_excess(cast, registered, n_mc=120, seed=5)
    assert with_cb.pvalue == without.pvalue
    assert with_cb.statistic == without.statistic
    assert with_cb.flag is without.flag
    assert with_cb.details["null_mean"] == without.details["null_mean"]
