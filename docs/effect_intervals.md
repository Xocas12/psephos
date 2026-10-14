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

A per-size-band finding is the case worth naming: it is handed the sizes of one band, so every
unit lands in the same stratum and there is nothing to hold fixed. Those findings report
`ci_stratified_by: null` and `ci_n_strata: 1`, because claiming stratification there would
describe the argument rather than the method — four such findings appear in a default audit.

When some resamples produce no finite value, how many goes in `ci_note`. Above the halfway
threshold an interval is still returned, and one computed from 70 of 100 resamples looks exactly
like one computed from all 100 unless it says otherwise.

The bands are imported from `size.py` rather than restated, so "the ones `size.py` uses" is true
by construction instead of by two lists that happen to agree today.

## The method, and the bias that forced it

A percentile bootstrap, shifted by the bootstrap's own estimate of its bias:

```
bias = mean(resample statistics) - estimate
interval = [q(alpha/2) - bias, q(1 - alpha/2) - bias]
```

For the effects that are means and shares -- the integer-percentage excess per unit, the
top-decile share gap, and the descriptive shares -- the shift is negligible and does nothing.
It was put in the code for the last-digit distance, and there it turned out not to be enough;
the next section has why, and what the distance uses instead.

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

## The distance needed more than a shift

Issue #29. The shift made the interval contain its estimate; it did not make it cover the
distance. The counted distance is itself biased upward by the noise, by about `1.2 / sqrt(n)` on
perfectly uniform digits: 0.038 at 1,000 units, 0.019 at 4,000. An interval re-centred on the
counted value covers what the counted value averages at that sample size, which is not the
distance a reader takes it for.

Measured with `last_digit_uniformity` on digits drawn at a known distance theta from uniform,
200 draws per cell, the same draws before and after. The departure is either spread, +theta/5 on
digits 0-4 and -theta/5 on 5-9, or put on two digits, one over and one under, with the other eight
untouched. Coverage of theta, shifted interval -> the one now shipped:

| units | departure | theta = 0 | 0.01 | 0.02 | 0.04 |
| --- | --- | --- | --- | --- | --- |
| 1,000 | spread | 0.01 -> 0.97 | 0.18 -> 0.99 | 0.64 -> 1.00 | 0.91 -> 1.00 |
| 1,000 | two digits | 0.00 -> 0.96 | 0.10 -> 0.99 | 0.20 -> 0.97 | 0.28 -> 0.99 |
| 4,000 | spread | 0.01 -> 0.96 | 0.55 -> 1.00 | 0.90 -> 1.00 | 0.93 -> 0.99 |
| 4,000 | two digits | 0.03 -> 0.99 | 0.33 -> 0.99 | 0.26 -> 0.99 | 0.22 -> 0.98 |

At 20,000 units and at theta 0.005 and 0.08 the shipped interval covered 96 to 100 per cent as
well. It also settles the comparison the interval exists for. With a true distance of 0.01,
100 draws at 1,000 units against 100 at 32,000, the shifted intervals were disjoint in 57 per
cent of the 10,000 pairs, and the shipped ones in none.

What `psephos.interval.distance_interval` does:

- **The effect** is the counted distance less the bootstrap's estimate of its bias,
  `T - (E*[T*] - T)`, computed exactly rather than by resampling: each digit's count in a
  resample is binomial, so the expected distance of a resample is a sum of binomial mean
  absolute deviations. It is seedless and does not change with `--no-intervals`. It removes
  nearly all the bias once the departure is large and about 40 per cent of it at zero, 0.023
  against 0.038 at 1,000 units. No estimate of a distance is unbiased near zero, which is why
  the interval, not the point, is the thing to read there.
- **The interval** bounds the true distance from each side by what counting noise can do to the
  counted one. The distance theta is the largest `p(B) - |B| / 10` over sets of digits B,
  reached at A, the digits with more than a tenth.
  - Above. T is at least the observed share of A less `|A| / 10`, so `theta - T` is at most the
    true share of A less the observed one: one share's sampling error, with a standard
    deviation of at most `1 / (2 sqrt(n))` whatever A is. The upper end is
    `T + 1.96 / (2 sqrt(n))`.
  - Below. `T - theta` is at most the distance between the observed shares and the true ones,
    which is all of T when theta is zero. Its 97.5th percentile comes from resampling the
    precincts within size bands, as the distance between each resample's shares and the
    observed ones. The lower end is T less that, floored at zero.

  Each end misses at most 2.5 per cent of the time whatever the true shares are, so the
  interval covers the distance at least 95 per cent of the time.

The cost is width. The lower end has to allow for the case where most of the counted distance
is noise, so on uniform digits at 1,000 units the interval is 0.069 wide on average against
0.046 for the shifted one, and at 4,000 units 0.035 against 0.023. The shifted interval was
narrower by being about a different quantity.

Two alternatives were measured and not shipped. The basic bootstrap,
`[2T - q(1 - alpha/2), 2T - q(alpha/2)]`, is centred on a bias-corrected estimate. Simulated in
the cells of the table above, it covered 70 to 92 per cent with theta above zero and 58 to 60
per cent at zero. It estimates the bias at the observed shares, where no digit sits exactly on a tenth,
and the bias is largest where digits do, as the eight untouched digits of the two-digit pattern
do. Cohen's w, with a noncentral chi-square interval, has an unbiased square and an interval
that can include zero, but it is a different number from the distance the title prints, and a
given w allows any distance from `w / sqrt(20)` to `w / 2`.

The STRONG flag's size gate, a distance of 0.02, now reads the effect. On uniform digits at
4,000 units the counted distance reached 0.02 in 373 of 1,000 draws, the effect in 101; the
p-value at or below 0.001 that the flag also needs is what keeps honest data from it.

## The distance's interval reaches zero

The earlier version of this document said a distance's interval could not contain zero. That was
wrong twice: the floor clamp did return zero as the lower end now and then, and nothing stops a
valid interval for a distance from reaching zero. A distance cannot be negative, but zero is a
value it takes, and an interval for it should include zero whenever the data are consistent with
no departure at all. The one now reported does: on uniform digits its lower end was zero in 96 to
99 per cent of draws.

The counted distance is still printed, beside what genuinely uniform digits count at the same
sample size, `details["null_expected_tvd"]`:

```
Last digit of party_second across 2864 units: most common 4 (10.8 per cent), least common 5
(9.2 per cent); uniform would be 10.0 per cent. Distance from uniform, less what counting noise
adds: 0.0084 (95 per cent interval [0.0000, 0.0370]). As counted it is 0.0186, against about
0.0225 for genuinely uniform digits at this sample size.
```

That is the second party of `clean_election()` from `tests/conftest.py`, with the registered
voters as sizes, as an audit runs it. The same digits used to print `0.0186 [0.0060, 0.0325]`:
an interval clear of zero, on honest digits slightly closer to uniform than chance puts them.

## Every check that reports an effect

All of them, not only the three the issue named. `last_two_digit_pairs` and `turnout_roundness`
are labelled descriptive but still report an effect, and `_template.py` is the canonical example
of the contribution bar, so all three gained one too. A default audit now has an interval on
every one of its 17 findings that carries an effect, which is what lets the README say so.

The coverage in a finding's title comes from `ci_level`, not from the string "95 per cent":
`--ci-level 0.8` puts "80 per cent interval" in the title. A hardcoded figure would have
contradicted the report's own EFFECT INTERVALS section two sections further down.

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
