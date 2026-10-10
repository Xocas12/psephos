# What the integer-percentage null costs, and why the obvious optimisation does not help

Issue #17 observed that the integer-percentage null is a Python loop over Monte Carlo
replicates, each drawing a binomial per unit, and that at 500 replicates and 95,000 units that
is 47.5 million draws per check, run eight or more times by one audit. It asked for the numbers
first and then for the loop to be vectorised into a single two-dimensional draw.

The numbers are below. The first half of the observation is right — the check is expensive, and
the arithmetic in the issue is accurate. The proposed fix is not: the Python loop accounts for
0.01 per cent of the time, so removing it cannot save anything. This document records the
measurement, because an optimisation that was tried and did not work is worth more written
down than repeated.

Everything here is reproducible:

```
uv run python benchmarks/bench_integer_pct.py --breakdown
uv run python benchmarks/bench_integer_pct.py --compare-blocked
uv run python benchmarks/bench_integer_pct.py
```

## What it costs

A synthetic election of 100,000 precincts, lognormal sizes, turnout beta around 60 per cent,
`min_denominator=250`, which leaves 94,232 usable units. One check at `n_mc=500` is 47,116,000
binomial draws.

| | |
| --- | --- |
| one check, `n_mc=500` | 5–23 s |
| the four-threshold sweep | 16–71 s |
| a full audit (two sweeps and a stratified pass) | several times the sweep |

The ranges are wide because they are wide. The same command on the same machine measured 5.07 s
and 23.12 s on the same afternoon, a factor of four, depending on what else the machine was
doing. **Absolute timings from this benchmark are not comparable across runs**, which is why the
comparison below interleaves its variants and why the breakdown measures shares within a single
run.

## Where the time goes

This is the measurement that decides the issue, and it is the one that does not care how busy
the machine is: the three shares are measured inside one run, so a machine twice as slow today
moves all three together.

```
94232 usable units, n_mc=200, 18,846,400 binomial draws
  total               2.376s
  rng.binomial        2.250s   94.7%
  counting step       0.126s    5.3%
  loop overhead       0.000s   0.01%
  per binomial draw 119 ns
```

Ninety-five per cent of the check is inside `numpy.random.Generator.binomial`. The counting
step — compute the percentages, round, compare against the tolerance — is five per cent. The
Python loop that was to be vectorised away is one part in ten thousand.

So the ceiling on vectorising the replicate loop is 0.01 per cent. That is not a measurement
that needs repeating on a quieter machine or a better profiler; it is arithmetic. The draws are
not slow because a Python loop asks for them one row at a time. They are slow because there are
47 million of them and each costs roughly 100 nanoseconds inside NumPy, whether they are asked
for one row at a time or all at once.

## The vectorisation, measured anyway

Drawing `rows` replicates at a time as one `(rows, n_units)` array, interleaved with the loop
over nine repeats, taking the best and the median of each:

```
94232 usable units, n_mc=150, 9 interleaved repeats

variant                     peak MB     best   median  s/replicate
loop (one replicate)            1.5    1.68s    1.93s  0.0112
blocked, 4 replicates           6.0    1.51s    2.00s  0.0101
blocked, 16 replicates         24.1    1.32s    1.70s  0.0088
blocked, 64 replicates         96.5    1.49s    2.11s  0.0099
```

Every variant produced the identical null. That is not a coincidence and it is not an
approximation: NumPy fills a `(rows, n)` broadcast draw in row-major order from the same
stream, so a blocked draw is the same replicates in the same order for the same seed, whatever
the block size. `tests/test_performance.py` pins that property, because the comparison above is
only a comparison of cost while it holds.

The spread within one variant is larger than the spread between variants. An earlier run of the
same comparison put the loop ahead of every block size; this one puts 16 replicates ahead by
about a fifth. Neither difference survives the run-to-run noise, and a fifth is not what the
issue was after.

What is not within the noise is the memory. A block of 64 replicates holds 96 MB where the loop
holds 1.5 MB, and a national dataset at `n_mc=500` would hold 760 MB as a single array, which
is the kind of allocation that gets a process killed on a laptop. Paying 64 times the working
memory for a difference that does not reproduce is a bad trade, so the loop stays.

## What would actually make it fast

Not implemented here, because each one changes what the check reports and that is a decision
for the repository, not a performance patch. Recorded so the next person does not start from
the loop again.

- **Compute the null exactly instead of sampling it.** The null count is a sum of independent
  indicators: unit *i* contributes 1 with probability *q_i* = P(its percentage lands within
  tolerance of a whole number), which is a sum of binomial PMF terms and can be computed
  directly. The count is then Poisson-binomial over the *q_i*, giving the mean and standard
  deviation in closed form and the p-value from a Poisson-binomial tail. This removes the Monte
  Carlo entirely: no replicates, no seed, no `1 / (n_mc + 1)` p-value floor. It is also the
  largest change, because the strong flag is currently defined against that floor
  (`mc_p <= 2 * p_floor and z >= 5`), so the flag thresholds would have to be redefined.
- **Share one null across the threshold sweep.** The usable sets nest, so one draw at the most
  permissive threshold could be re-counted at the stricter ones, for something close to the
  sweep's full cost. The sweep exists to show whether a signal survives a stricter threshold,
  and the multiple-testing correction already collapses the four thresholds to one hypothesis
  through the smallest raw p among them. Drawing those four p-values from shared replicates
  correlates them, and a minimum over correlated p-values is not the selection a minimum over
  independent ones is. Tractable, and it needs stating before it is relied on.
- **A faster exact binomial sampler.** At roughly 100 ns a draw this is where the time is. Any
  sampler that is correct gives a valid null, but a different one gives different replicates,
  so the same seed would stop reproducing a published number.

## What this change did do

Since the loop is not the problem, what it got instead is the thing that makes a slow check
tolerable rather than faster:

- `benchmarks/bench_integer_pct.py`, so the numbers above can be re-measured and argued with.
- A `progress` callback on `integer_excess`, reported on stderr by the audit and the CLI
  (`--progress`, `--no-progress`; on by default when stderr is a terminal). This is not
  cosmetic. `n_mc` sets the Monte Carlo p-value floor at `1 / (n_mc + 1)`, and the strong flag
  is defined against it, so someone who kills a silent multi-minute run and retries with fewer
  replicates does not get the same answer sooner — they get a different answer. A run that
  says what it is doing is a run that gets left alone to finish.
