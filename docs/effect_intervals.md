# Intervals on effect sizes, and the one that needed correcting

Issue #16: a finding reported a point effect and no uncertainty, so a reader could not tell an
effect measured on 300 precincts from the same effect measured on 30,000. This document records
what the interval is, what it is not, and the one place where the obvious construction states
something false.

## What is resampled, and why it is stratified

**Units, not rows within units.** A precinct is the unit of observation for every check in
psephos, so the interval has to express uncertainty about which precincts were drawn.
Resampling voters inside a precinct would answer a different question, one psephos does not ask.

**Within precinct-size bands**, at each band's observed count. This is not a refinement. Every
percentage effect here depends on size: a whole-number turnout percentage is common by
arithmetic in a 200-voter station and rare in a 3,000-voter one, which is the whole reason
`size.py` exists. An unstratified resample draws size mixes the election did not have, so the
interval widens with variation in a quantity that is fixed and known. The bands are the ones
`size.py` already uses.

Checks whose inputs do not carry a precinct size — `last_digit_uniformity` called on counts
alone — resample unstratified and say so in `details["ci_stratified_by"]`. The audit passes the
registered-voters column wherever it has one, so in an audit they are stratified too.

## The method, and the bias that forced it

A percentile bootstrap, shifted by the bootstrap's own estimate of its bias:

```
bias = mean(resample statistics) - estimate
interval = [q(alpha/2) - bias, q(1 - alpha/2) - bias]
```

For the two effects that are means — the integer-percentage excess per unit, and the top-decile
share gap — the shift is negligible and does nothing. It is in the code for the third.

The last-digit effect is the total variation distance from uniform, a sum of absolute
deviations. Resampling adds noise to a histogram, and added noise can only push a sum of
absolute deviations **up**. Measured across twenty clean synthetic elections and two contestant
columns each:

- The bootstrap distribution sat above the estimate in **40 of 40 cases**.
- The gap was **13 to 56 per cent of the interval's own width**, median 33 per cent.
- It does not shrink away: at 500, 2,000, 8,000 and 32,000 units the bias was 118, 26, 63 and
  40 per cent of the estimate.
- The plain percentile interval therefore **excluded the estimate outright in 5 of those 40
  cases**, on honest data with nothing injected.

An interval that does not contain the number it is an interval for misleads a reader as
directly as a wrong number would, so the plain construction is not shipped.
`tests/test_interval.py` pins both halves: that the inflation is one-directional across all 40
cases, and one specific case where the uncorrected method fails and the shifted one does not.

Two further guards. `Finding.__post_init__` refuses an interval that does not contain its own
effect, so this class of error cannot reach a report from any check. And if the shift is not
enough to make the interval contain the estimate — a resampling distribution too skewed for
this method — no interval is reported at all and `details["ci_note"]` says why.

Not BCa. Efron's accelerated interval would handle the skewness as well as the bias and is the
better instrument. Its acceleration term needs a jackknife over units, which is a second pass
whose cost and whose behaviour on a boundary statistic are worth measuring before adopting
rather than assuming.

## A distance has no interval that contains zero

This follows from the above and is worth stating on its own, because it is the misreading that
would do the most damage. A total variation distance is non-negative. Its interval therefore
never contains zero, however honest the data, and "the interval excludes zero" means nothing at
all for it.

What a reader needs instead is the value the statistic takes **when nothing is wrong**, which
is not zero and which falls as `1 / sqrt(n)`. Every `last_digit` finding carries it as
`details["null_expected_tvd"]` and prints it in the title:

```
Last digit of PartyA across 3490 units: most common 1 (11.0 per cent), least common 7
(9.3 per cent); uniform would be 10.0 per cent. Distance from uniform 0.0146
[0.0038, 0.0269], against 0.0203 for genuinely uniform digits at this sample size.
```

0.0146 against an expected 0.0203 is a dataset slightly *closer* to uniform than chance
produces. A reader given only the first number and an interval excluding zero could have
concluded the opposite.

## Why the interval comes first

The report prints `effect=... [low, high]` before `stat=` and `p=`. With 95,000 precincts almost
any departure from a null is significant, so the p-value is the least informative number on the
line, and a line that opens with it invites a trivial effect to be read as an important one.
The ordering is the finding of issue #16, not a formatting preference.

## What the interval does not cover

It covers uncertainty about which precincts were drawn. It does not widen for the possibility
that the **null model is wrong**, which is the dominant uncertainty in this entire tool and
which no interval can express. The binomial null in `integer_pct.py` assumes each precinct's
count is noise around its own observed rate; if that is the wrong model for a given electoral
administration, the effect and its interval are both measuring the wrong thing precisely. The
report says so, and the interpretation section says it again.

## Cost

One recount per resample over an index array, never a second Monte Carlo. The
integer-percentage check makes this possible by accumulating how often each unit individually
landed on a whole number while it draws the null it already draws, so the effect becomes a mean
of per-unit contributions — `sum(hit_i - q_i) / n`, which is the same number as
`(observed - null_mean) / n` — and a resample is a mean over an index array. Deriving `q_i`
afterwards would have meant re-running the Monte Carlo for every resample.

Measured on a 4,000-precinct audit, the intervals add about a second to three, or roughly a
third. In the test suite the share is larger, because the tests run many audits at a small
`n_mc` where the Monte Carlo is cheap and the bootstrap is not: `tests/test_audit.py` and
`tests/test_methods.py` together take about 1.9 times as long as before, measured over three
interleaved pairs. Two things keep that from being worse:

- The resample indices are a generator, not a list. Holding a thousand of them at once would be
  32 MB on a 4,000-precinct election and 760 MB on a national one, for arrays each used once.
- `turnout._quantile` replaces the sort inside `np.quantile` with an O(n) selection, since the
  bootstrap recomputes the decile cut once per resample. That alone took the dependence check
  from 0.41 s to 0.15 s at a thousand resamples. It agrees with `np.quantile` to within 2 units
  in the last place and is exact where the quantile falls on an order statistic, which cannot
  move a unit across the cut.

### The number of resamples

Chosen by measurement rather than convention. Varying only the resampling seed and recording how
far the interval ends move, as a share of the interval's own width:

| resamples | movement in the ends |
| --- | --- |
| 200 | 7.0 per cent of the width |
| 500 | 4.9 per cent |
| 1,000 | 3.6 per cent |
| 2,000 | 3.1 per cent |

A thousand is where it stops paying, and it leaves the reported ends good to about their second
significant digit, which is all the precision the effects carry.

`--bootstrap N` sets the resamples, `--ci-level P` the coverage, `--no-intervals` turns them
off.
