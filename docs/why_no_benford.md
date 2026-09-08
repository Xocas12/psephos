# Why there is no first-digit Benford test

Sooner or later someone asks for Benford's law. It is a standard item in election forensics
toolkits, and among digit tests it is the cheapest to compute, which is part of its appeal.
This document is the answer psephos gives instead of the test.

The short version: **on honest precinct data the first-digit test rejects almost always, and
when it rejects, the p-value looks like proof.** The demonstration below runs on data this
repository generated, with no fraud anywhere in it.

## The argument

Benford's law says the first significant digit d of a number appears with probability
log10(1 + 1/d): about 30.1 per cent of values start with 1 and 4.6 per cent with 9 (Benford
1938). Hill (1995) derives the law for samples drawn from random mixtures of distributions,
which is why it shows up in tables that combine very different scales. The forensic
accounting literature turns that into a practical checklist for when the test applies: the
data should span at least one to two orders of magnitude, the more the better, and should
have no built-in minimum or maximum (Nigrini 2012; Durtschi, Hillison and Pacini 2004).

Precinct vote counts inside a single election meet none of that.

- **The span is not there.** Precinct sizes are an administrative choice, and within one
  election they sit inside a narrow band: in the honest election generated below, the counts
  run from 80 to 2340 ballots and the middle half of them lies within a factor of 2.3.
  The law is not about the extremes; it is about the shape across the span, and there is
  almost no span to have a shape across.
- **There are built-in bounds.** Jurisdictions split precincts that grow too large and
  consolidate those that shrink too small, so the size distribution is truncated by design at
  both ends. A hard minimum or maximum is one of the stated disqualifiers for the test.
- **With the span missing, the first-digit distribution is set by the precinct size
  distribution.** Where the sizes put their mass decides how many counts begin with 1.
  Fabrication can barely move it, and honesty does not hold it near Benford.

Because an election has thousands of precincts, a chi-square test has enormous power against
that size-distribution signal. The honest-data rejection does not come back at p = 0.03; it
comes back at p = 4.23e-79. That is the worst possible output shape for a forensic tool: a
number that reads as proof, produced by honest data.

## The demonstration

Run this from the repository root. `clean_election()` is the generator from
`tests/conftest.py`: lognormal precinct sizes, smoothly varying turnout and support, and
every count a binomial draw. No rounding, no stuffing, no invented digit. It is the data
every check in psephos is required to stay quiet on, and every check in psephos does.

```console
$ uv run python - <<'PY'
import sys
sys.path.insert(0, "tests")

import numpy as np
from scipy import stats

from conftest import clean_election

counts = clean_election()["ballots_cast"].to_numpy(float)
counts = counts[counts > 0]  # a count of zero has no first digit

# Chi-square of the first digits of these honest counts against Benford's law,
# P(d) = log10(1 + 1/d), on 8 degrees of freedom.
n = np.rint(counts).astype(np.int64)
first = n // 10 ** np.floor(np.log10(n)).astype(np.int64)
observed = np.bincount(first, minlength=10)[1:10]
benford = np.log10(1.0 + 1.0 / np.arange(1, 10))
expected = benford * observed.sum()
chi2 = float(((observed - expected) ** 2 / expected).sum())
pvalue = float(stats.chi2.sf(chi2, df=8))
tvd = float(0.5 * np.abs(observed / observed.sum() - benford).sum())

print(f"{len(counts)} precincts, ballots cast from {int(counts.min())} to {int(counts.max())}")
q1, q3 = np.percentile(counts, [25, 75])
print(f"middle half: {int(q1)} to {int(q3)}, a factor of {q3 / q1:.1f}")
for d in range(1, 10):
    share = observed[d - 1] / observed.sum()
    print(
        f"digit {d}: observed {observed[d - 1]:4d} ({share:.3f})"
        f"  benford {expected[d - 1]:7.1f} ({benford[d - 1]:.3f})"
    )
print(f"chi2={chi2:.1f} df=8 p={pvalue:.2e} tvd={tvd:.4f}")
PY
```

```text
4000 precincts, ballots cast from 80 to 2340
middle half: 324 to 748, a factor of 2.3
digit 1: observed  792 (0.198)  benford  1204.1 (0.301)
digit 2: observed  572 (0.143)  benford   704.4 (0.176)
digit 3: observed  617 (0.154)  benford   499.8 (0.125)
digit 4: observed  545 (0.136)  benford   387.6 (0.097)
digit 5: observed  478 (0.119)  benford   316.7 (0.079)
digit 6: observed  347 (0.087)  benford   267.8 (0.067)
digit 7: observed  294 (0.073)  benford   232.0 (0.058)
digit 8: observed  213 (0.053)  benford   204.6 (0.051)
digit 9: observed  142 (0.035)  benford   183.0 (0.046)
chi2=681.4 df=8 p=7.20e-142 tvd=0.1864
```

Benford predicts 30.1 per cent of counts to start with 1; 15.4 per cent of these honest
counts do. Chi-square 389.0 on 8 degrees of freedom, p = 4.23e-79, total variation distance
from Benford 0.146. In the reporting conventions of the README's sample output that is
`stat=681.4  p=7.20e-142  effect=0.1864`: a STRONG flag, on data that contains no anomaly of
any kind.

One election could be luck. Run the same test over a hundred independent honest elections
from the same generator:

```console
$ uv run python - <<'PY'
import sys
sys.path.insert(0, "tests")

import numpy as np
from scipy import stats

from conftest import clean_election

benford = np.log10(1.0 + 1.0 / np.arange(1, 10))
rejections = 0
for seed in range(100):
    counts = clean_election(seed=seed)["ballots_cast"].to_numpy(float)
    n = np.rint(counts).astype(np.int64)
    first = n // 10 ** np.floor(np.log10(n)).astype(np.int64)
    observed = np.bincount(first, minlength=10)[1:10].astype(float)
    expected = benford * observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    if stats.chi2.sf(chi2, df=8) <= 0.05:
        rejections += 1
print(f"the first-digit test rejected {rejections} of 100 honest elections at the 0.05 level")
PY
```

```text
the first-digit test rejected 100 of 100 honest elections at the 0.05 level
```

A test that fires on every honest election carries no information on this data. The
informative contrast is with a digit test psephos does ship, run on exactly the same counts:

```console
$ uv run python - <<'PY'
import sys
sys.path.insert(0, "tests")

from conftest import clean_election

from psephos.methods.digits import last_digit_uniformity

counts = clean_election()["ballots_cast"].to_numpy(float)
finding = last_digit_uniformity(counts)
print(f"flag={finding.flag.value} p={finding.pvalue:.3f} effect={finding.effect:.4f}")
PY
```

```text
flag=ok p=0.749 effect=0.0171
```

Same data, same family, same chi-square construction, opposite verdict. The difference is not
the machinery: the last digit of a count of several hundred ballots is free, while the first
digit is pinned by how large the precinct is allowed to be. If that diagnosis is right, the
span is what decides it, and nothing else. Widen the precinct sizes in the same generator to
cover several orders of magnitude, leave the counts exactly as honest as they were, and the
first-digit test stops rejecting:

```console
$ uv run python - <<'PY'
import sys
sys.path.insert(0, "tests")

import numpy as np
from scipy import stats

# The same honest process as clean_election(), with precinct sizes spanning several
# orders of magnitude instead of one: lognormal sigma 1.8 rather than 0.6.
rng = np.random.default_rng(11)
size = np.clip(rng.lognormal(mean=7.5, sigma=1.8, size=4000), 10, 10_000_000)
turnout = np.clip(rng.normal(0.55, 0.10, size=4000), 0.05, 0.98)
counts = rng.binomial(np.rint(size).astype(int), turnout).astype(float)
counts = counts[counts > 0]  # a count of zero has no first digit

n = np.rint(counts).astype(np.int64)
first = n // 10 ** np.floor(np.log10(n)).astype(np.int64)
observed = np.bincount(first, minlength=10)[1:10].astype(float)
benford = np.log10(1.0 + 1.0 / np.arange(1, 10))
expected = benford * observed.sum()
chi2 = float(((observed - expected) ** 2 / expected).sum())
pvalue = float(stats.chi2.sf(chi2, df=8))
print(f"counts from {int(counts.min())} to {int(counts.max())}")
print(f"chi2={chi2:.1f} df=8 p={pvalue:.3f}")
PY
```

```text
counts from 1 to 857145
chi2=6.7 df=8 p=0.566
```

## What happens anyway

The test is applied to election data constantly, and it produces exactly the output shape
above: a tiny p-value and a confident headline. Deckert, Myagkov and Ordeshook (2011) put
Benford's-law digit tests against elections simulated with and without fraud and found false
positives on the honest elections and false negatives on the fraudulent ones; their verdict
is that the method has "significant, even fatal limitations". Mebane, the leading proponent
of the second-digit variant, replied in the same issue (Mebane 2011).

psephos prints an innocent explanation next to every finding it flags. For the first-digit
test the honest explanation is "this is what honest precinct data looks like", which would
make every finding content-free. The alternative is to ship the check with a warning, and
warnings do not survive contact with a p-value that small. So the check is absent, and this
document is why.

There is also this repository's own merge bar: CONTRIBUTING.md requires every new check to
stay quiet on `clean_election()`. The demonstration shows a first-digit Benford test cannot
pass that requirement, so it is excluded before implementation is ever discussed.

## The second digit is a different question

Nothing here argues against the second-digit test (2BL; Mebane 2008), which is the form of
Benford's law actually used in the election forensics literature. Its expected distribution
is far flatter than the first-digit law's, 11.4 per cent for digit 0 down to 8.4 per cent for
digit 9 where the first-digit law runs from 30.1 to 4.6, and it is applied to counts that sit
inside one order of magnitude, so the argument above does not decide that case and this
document must not be cited as though it did. If the second-digit test is wanted in psephos,
it needs its own issue with its own argument and its own quiet test on honest data. psephos
does not implement that one either, yet.

## Sources

- Beber, B. and Scacco, A. (2012). "What the Numbers Say: A Digit-Based Test for Election
  Fraud". Political Analysis 20(2).
- Benford, F. (1938). "The Law of Anomalous Numbers". Proceedings of the American
  Philosophical Society 78(4), 551-572. doi:10.2307/984802
- Deckert, J., Myagkov, M. and Ordeshook, P. C. (2011). "Benford's Law and the Detection of
  Election Fraud". Political Analysis 19(3), 245-268. doi:10.1093/pan/mpr014
- Durtschi, C., Hillison, W. and Pacini, C. (2004). "The Effective Use of Benford's Law to
  Assist in Detecting Fraud in Accounting Data". Journal of Forensic Accounting 5(1), 17-34.
- Hill, T. P. (1995). "A Statistical Derivation of the Significant-Digit Law". Statistical
  Science 10(4), 354-363. doi:10.1214/ss/1177009869
- Mebane, W. R. Jr. (2008). "Election Forensics: The Second-Digit Benford's Law Test and
  Recent American Presidential Elections". In Alvarez, R. M., Hall, T. E. and Hyde, S. D.
  (eds), "Election Fraud: Detecting and Deterring Electoral Manipulation". Brookings
  Institution Press, 162-181.
- Mebane, W. R. Jr. (2011). "Comment on 'Benford's Law and the Detection of Election
  Fraud'". Political Analysis 19(3), 269-272.
- Nigrini, M. J. (2012). "Benford's Law: Applications for Forensic Accounting, Auditing, and
  Fraud Detection". Wiley.
