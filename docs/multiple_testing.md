# What counts as one family, and why

An audit runs about eighteen checks across several slices. Under a true null, some of them
flag. That is not a defect; it is what a significance level means. But a reader handed a list
of flagged findings with no statement of how many tests produced them has been given a
misleading picture, and the misleading direction is always the same one.

This document argues the family definition. The implementation is `src/psephos/correction.py`.

## The measured baseline

Over twenty independent clean elections from `tests/conftest.py` (seeds 400 to 419, `n_mc=200`),
the audit produced **27 flags out of 360 findings**: about **7.5 per cent** against a 5 per cent
nominal, or **1.35 flags per audit**. Before the last-digit check allowed for its own
`min_count` cut ([#25](https://github.com/Xocas12/psephos/issues/25)) the same elections gave
28, 7.8 per cent.

That number was a surprise and is worth dwelling on. An earlier version of this repository
reported "0 flagged of 18" on clean data and offered it as evidence of good calibration. It
was one fortunate seed. `tests/test_audit.py` now pins the rate rather than requiring zero, and
`docs/why_no_benford.md` records the related lesson: a single-seed "must not flag" assertion
requires a correct check to get lucky.

Two things in that number are different in kind. The flags per audit are what multiplicity
produces: eighteen findings at 5 per cent each give about 0.9 flags per audit with nothing
wrong, and more checks give more. The rate per finding is not. A check calibrated at 5 per cent
flags 5 per cent of its findings however many other checks run beside it, so a rate above 5
per cent is a check firing too often on honest data, and no correction for multiplicity
repairs that. Allowing for the last-digit cut made a net one fewer: two `party_third` flags
gone, one `party_incumbent` flag added. Of the 27 left, 23 are integer-percentage findings,
from 9 of the 20 elections and 7 of them from one: the sweep re-asks one question of nested
subsets, so an election that trips it trips it several times, and twenty elections measure
that rate only loosely.

So the correction is not defensive tidying. On this tool's own clean data, roughly three flags
in forty are noise.

## The problem with counting tests naively

The obvious family is "every finding that carries a p-value". It is wrong in both directions.

**It over-counts.** The integer-percentage check runs at four `min_denominator` thresholds:
100, 250, 500 and 1000. These are not four tests. They are one claim, *are turnout percentages
piling up on whole numbers*, re-asked of four nested subsamples of the same data. The 1000
subsample is contained in the 100 subsample. Their p-values are strongly positively correlated
by construction, and treating them as four independent tests would inflate the family by a
factor of four and destroy the power of the tool's single most important check. The size strata
have the same structure.

**It under-counts nothing, but it does mislead.** The per-contestant last-digit tests are a
different case. Each contestant's counts are a separate claim about a separate set of numbers.
They are not independent of each other, since the counts sum within a precinct, but they are
separate hypotheses in the sense that matters: a flag on one is not a re-statement of a flag on
another.

## The rule

> **One hypothesis is one substantive claim. Re-tests of the same claim on nested or
> overlapping data enter the correction once, through the smallest p among them.**

Concretely, `hypothesis_key` groups:

| Findings | Hypothesis | Why |
|---|---|---|
| integer percentage on turnout, at all four thresholds and all size strata | `integer_pct:turnout` | one claim, re-asked of nested subsamples |
| integer percentage on the winner's share, ditto | `integer_pct:<winner> share` | same |
| last digit, per contestant | one per contestant | separate numbers, separate claim |
| last two digits | its own | separate claim |
| turnout and share dependence | its own | separate claim |

The effect is that the family is the number of *hypotheses*, not the number of *findings*, and
it is materially smaller. A test asserts that inequality holds rather than trusting it.

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

The sweep entry at `min_denominator=1000` might have a raw p of 0.21 while the hypothesis
entered the correction at 0.02 through the `min_denominator=100` entry. Every member of the
group then reports the hypothesis's adjusted value, 0.03 say, which is below 0.21.

The naive invariant `adjusted >= raw` is therefore false here by design, and a test asserting it
was removed for that reason. What does hold, and is tested:

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
