# What counts as one family, and why

An audit runs about eighteen checks across several slices. Under a true null, some of them
flag. That is not a defect; it is what a significance level means. But a reader handed a list
of flagged findings with no statement of how many tests produced them has been given a
misleading picture, and the misleading direction is always the same one.

This document argues the family definition. The implementation is `src/psephos/correction.py`.

## The measured baseline

Over twenty independent clean elections from `tests/conftest.py`, the audit produced **28
flags out of 360 findings**: about **7.8 per cent** against a 5 per cent nominal, or **1.4
flags per audit**.

That number was a surprise and is worth dwelling on. An earlier version of this repository
reported "0 flagged of 18" on clean data and offered it as evidence of good calibration. It
was one fortunate seed. `tests/test_audit.py` now pins the rate rather than requiring zero, and
`docs/why_no_benford.md` records the related lesson: a single-seed "must not flag" assertion
requires a correct check to get lucky.

So the correction is not defensive tidying. On this tool's own clean data, roughly three flags
in forty are noise.

## The problem with counting tests naively

The obvious family is "every finding that carries a p-value". It is wrong in both directions.

**It over-counts.** The integer-percentage check runs at four `min_denominator` thresholds:
100, 250, 500 and 1000. These are not four tests. They are one claim, *are turnout percentages
piling up on whole numbers*, re-asked of four nested subsamples of the same data. The 1000
subsample is contained in the 100 subsample. Their p-values are strongly positively correlated
by construction, and treating them as four independent tests would inflate the family by a
factor of four and destroy the power of the tool's single most important check.

The size strata re-ask the same turnout claim, but they are not nested. `stratify()` cuts the
precincts into disjoint bands of registered voters (0-99, 100-249, 250-499, 500-999, 1000-1999,
2000 and up), the audit tests every band with at least 200 precincts, and no precinct is in
two bands; `tests/test_audit.py` asserts that. They belong to the turnout hypothesis because
they ask where in the size range the same signal lives, not a different question. But their
p-values are close to independent of each other, the opposite of the thresholds, and that
decides how the group may enter the correction; see below.

**It under-counts nothing, but it does mislead.** The per-contestant last-digit tests are a
different case. Each contestant's counts are a separate claim about a separate set of numbers.
They are not independent of each other, since the counts sum within a precinct, but they are
separate hypotheses in the sense that matters: a flag on one is not a re-statement of a flag on
another.

## The rule

> **One hypothesis is one substantive claim. Re-tests of the same claim, on nested subsamples
> or on disjoint size bands, enter the correction once, through the Simes combination of their
> p-values.**

Concretely, `hypothesis_key` groups:

| Findings | Hypothesis | Why |
|---|---|---|
| integer percentage on turnout, at all four thresholds and all size strata | `integer_pct:turnout` | one claim, re-asked of four nested subsamples and of disjoint size bands |
| integer percentage on the winner's share, at all four thresholds | `integer_pct:<winner> share` | one claim, re-asked of four nested subsamples |
| last digit, per contestant | one per contestant | separate numbers, separate claim |
| last two digits | its own | separate claim |
| turnout and share dependence | its own | separate claim |

The effect is that the family is the number of *hypotheses*, not the number of *findings*, and
it is materially smaller. A test asserts that inequality holds rather than trusting it.

## One p per hypothesis, and why it is not the smallest

Counting a group once only helps if the p it enters with is a p-value, and the smallest of K
valid p-values is not one. When the K are independent it lands at or below 0.05 in `1 - 0.95^K`
of null datasets: about 19 per cent for four looks, 34 per cent for eight. Nested thresholds
limit the damage because they mostly agree. The size bands are the bad case: they are close to
independent, so a minimum over them is a minimum over separate draws. An earlier version of this
correction entered every group through its smallest p.

Measured on clean elections from `clean_election(n=4000)`, one election per seed and every
hypothesis a true null, with `n_boot=0`: the share of audits in which a hypothesis entered the
correction at or below 0.05.

| | turnout (4 thresholds, 4 bands) | winner's share (4 thresholds) |
|---|---|---|
| each slice on its own, `n_mc=500` | 3.6 to 4.3 per cent | 3.2 to 4.1 per cent |
| smallest p, `n_mc=500`, 440 audits | **16.6 per cent** (73) | **7.7 per cent** (34) |
| Simes, `n_mc=500`, 440 audits | 3.2 per cent (14) | 2.3 per cent (10) |
| smallest p, `n_mc=100`, 330 audits | **17.3 per cent** (57) | **9.4 per cent** (31) |
| Simes, `n_mc=100`, 330 audits | 0.9 per cent (3) | 2.4 per cent (8) |

The smallest p entered turnout at more than three times the nominal rate, most of it from the
four bands. Both smallest-p rows fail a one-sided binomial test against 5 per cent at the 0.01
level, and every Simes row passes it. `tests/test_correction_calibration.py` repeats the
`n_mc=100` measurement on 200 audits and fails if either hypothesis enters at or below 0.05 more
often than a valid p-value can.

The combination is Simes (1986): with the group's K raw p-values sorted, it enters with
`min_i(K * p_(i) / i)`. Three properties made it the choice.

- It is a valid p-value, `P(p <= a) <= a` under the null, when the K values are independent or
  positively dependent, which are the two cases these groups contain (next section).
- It costs nothing when the slices agree. Thresholds 100 and 250 select exactly the same
  precincts on the synthetic elections and return the same p, and K identical values combine to
  that value. A real rounding signal shows at every threshold and pays little.
- It charges a lone slice the full factor K, as Bonferroni would. A signal that appears in one
  band and nowhere else has to be K times stronger to enter where it would have entered alone,
  which is the price of having looked in K places. The sweep exists to stop a threshold being
  tuned until the answer is interesting; entering through the smallest p brought that selection
  back inside the correction.

It is conservative, and the table shows by how much. Each slice's own Monte Carlo p-value
already lands at or below 0.05 a little under 5 per cent of the time (the first row), and Simes
gives back more on strongly correlated nested slices. At `n_mc=100` with eight slices the
turnout hypothesis almost never enters below 0.05, because the Monte Carlo floor of 1/101 is
above the 0.05/8 a lone slice needs. Run the default `n_mc=500` or more when the turnout
hypothesis's adjusted p matters.

Considered and not used:

- **Bonferroni or Holm within the group** (K times the smallest p). Valid under any dependence,
  but it charges K even when the slices agree, which on nested thresholds is most of the time,
  and with eight slices at `n_mc=100` it cannot reach 0.05 at all.
- **A joint Monte Carlo null** (a Westfall-Young min-p), drawing each precinct's replicates once
  and recounting every threshold and band from the same draws. Exact under the real dependence
  and the most powerful option, but it needs the slices to share replicates, which changes what
  each slice reports and is recorded as a decision of its own in `docs/performance.md`.
- **One pre-registered slice as the entering p**, the rest descriptive. Exact, but it makes the
  choice of slice a parameter again.

## What the correction assumes about dependence

Both steps of the correction rest on one assumption: **the p-values being combined are
independent, or positively dependent in the PRDS sense** (positive regression dependence on
each one from a subset, Benjamini and Yekutieli 2001). Under it the Simes p of a group is a
valid p-value (Simes 1986 for independence; Sarkar 1998 and Benjamini and Yekutieli 2001 for
positive dependence), and Benjamini-Hochberg bounds the false discovery rate at `q` (Benjamini
and Yekutieli 2001, Theorem 1.2).

**Within a hypothesis** the assumption can be argued, and measured. The correlations below are
Spearman rank correlations between slice p-values over the 440 clean audits at `n_mc=500`.

- *Size bands against each other.* They share no precinct, and under the null each precinct's
  count is its own draw, so the bands' observed counts are independent. The one shared
  ingredient is that each band's Monte Carlo null starts from the same seed. Measured: -0.04 to
  +0.01.
- *Thresholds against each other, and a threshold against a band inside it.* A precinct that
  lands on a whole number raises the count of every slice that contains it, so the counts are
  sums over overlapping sets of the same independent indicators and rise together. Measured:
  +0.63 to +1.00 between turnout thresholds, +0.36 to +0.93 between share thresholds, and +0.28
  to +0.90 between a threshold and a band it contains.
- *A threshold against a band outside it*, such as the 500 threshold and the 250-499 band: no
  shared precinct, independent again. Measured: -0.04 to -0.02.

That this positive dependence meets PRDS exactly is argued, not proven. The entering-p
measurement above is the check that it holds closely enough, and a test pins it.

**Across hypotheses** it is an assumption, not an argument. The hypotheses read different numbers
from the same precincts: turnout percentages, the winner's share percentages, the last digits of
each contestant's count, the relation of turnout to share. Nothing in how they are computed makes
one p small when another is large, which is the negative dependence that breaks
Benjamini-Hochberg, but PRDS has not been shown for them either.

The variant of Benjamini and Yekutieli (2001) controls the rate under *any* dependence by running
the same procedure at `q / c(m)`, with `c(m) = 1 + 1/2 + ... + 1/m` for m hypotheses. It is not
used. It would multiply every adjusted p by `c(m)`, 2.45 for the six hypotheses that carry a
p-value in a default audit of the synthetic elections, to insure against a kind of dependence the
family has no mechanism for, in a tool whose failure mode on dirty data is already being too
quiet. A reader who needs the arbitrary-dependence bound can multiply the adjusted p by `c(m)`
for the number of hypotheses the report states.

What the assumption buys can be measured. On clean elections every hypothesis is a true null,
so the false discovery rate is the share of audits with any adjusted p at or below `q`. Over the
440 audits at `n_mc=500` above, that share is 6.8 per cent (30 audits), down from 11.1 per cent
(49) when groups entered through their smallest p. 23 of the 33 adjusted p-values at or below
`q` in those audits belong to the three last-digit hypotheses, whose own excess on minor-party
counts is filed as [#25](https://github.com/Xocas12/psephos/issues/25); Benjamini-Hochberg run
on the family without them rejects in 4.3 per cent (19).

## Why a false-discovery-rate procedure, not a family-wise one

Benjamini-Hochberg at `q = 0.05`, not Bonferroni or Sidak.

The question this tool answers is "which of these findings deserve investigation?", and the
cost structure is asymmetric in a specific way. A false positive here costs someone's time and,
if published carelessly, a false accusation. A false negative costs a missed anomaly that other
evidence may still catch. Controlling the expected *proportion* of false discoveries among the
flagged set matches that question better than controlling the probability of *any* false
positive, which at eighteen tests would make the tool nearly blind.

The counter-argument deserves recording: if a single flag from psephos is going to be quoted in
public as though it stood alone, family-wise control is the honest choice, because then one
false positive is the whole cost. The answer is not to switch procedures but to refuse that use
of the output, which is what `INTERPRETATION.md` does at length.

## The correction reports, it does not decide

**No flag is ever raised or withdrawn by the correction.** The adjusted p is printed beside the
raw one and nothing else changes.

This is deliberate. Letting the correction move flags would make the flag depend on which other
checks happened to run in the same audit, so adding an unrelated check could silence a real
finding. A reader who wants to apply a threshold to the adjusted value can, and the report tells
them how many hypotheses were tested so the number means something.

## The one counter-intuitive consequence

The adjusted value belongs to the **hypothesis**, not to the individual finding. So a finding
can carry an adjusted p *below* its own raw p, and this is correct.

The sweep entry at `min_denominator=1000` might have a raw p of 0.21 while the hypothesis,
combined over all its slices, entered the correction at 0.02. Every member of the group then
reports the hypothesis's adjusted value, 0.03 say, which is below 0.21.

The naive invariant `adjusted >= raw` is therefore false here by design, and a test asserting it
was removed for that reason. What does hold, and is tested:

- `the p that entered >= the smallest raw p` among the hypothesis's re-tests
- `adjusted >= the p that entered` for the hypothesis
- `adjusted <= 1`
- flags are byte-identical before and after
- all members of one hypothesis share one adjusted value

If that grouping is ever abandoned, the naive invariant returns and the tests should say so.

## What this does not fix

A correction cannot rescue a badly specified test, and it is not a substitute for a baseline.
The largest limitation of this tool remains the absence of a reference corpus
([#4](https://github.com/Xocas12/psephos/issues/4)): knowing that a finding survives correction
within one audit says nothing about whether its value is unusual for an election.

Nor does it address the slice dimension properly. When `--by region` lands
([#5](https://github.com/Xocas12/psephos/issues/5)), eighty-five regions times several checks
will produce flagged regions from nothing, and the grouping above does not yet describe how a
region's tests relate to the national ones. That is a live question, not a settled one.
