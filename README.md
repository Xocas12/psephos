# psephos

Statistical anomaly detection for precinct-level election results.

**It flags patterns that warrant an explanation. It does not detect fraud, and no statistic
can.** Read [INTERPRETATION.md](INTERPRETATION.md) before you use any output in public.

```console
$ psephos audit results.csv --registered registered --votes "party_a,party_b,party_c"

==============================================================================
psephos audit: results.csv
4000 units
leading contestant: party_a
==============================================================================

INTEGRITY
  [  ok  ] duplicate_units: No duplicate rows on ['region', 'precinct'].
  [  ok  ] negative_counts: No negative counts.
  [  ok  ] turnout_over_100: No unit reports more ballots than registered voters.

FLAGGED FINDINGS (2 of 19)
  [STRONG ] integer_percentage  (turnout, min_denominator=250)
      614 of 3820 units land on a whole-number turnout percentage; binomial noise
      predicts about 189. Excess +425 units (+11.13 per cent of those tested).
      stat=28.4  p=0.001996  effect=0.1113  n=3820  excluded=180
      - could also be: Small precincts, where whole-number percentages are common by
        arithmetic alone...
```

## Why this exists

Election forensics has a real literature and a real failure mode. The methods are published and
reproducible; the failure mode is that they get applied to one election, with no comparison and
no false-positive rate, by someone who already knows the answer they want. The output is a
p-value that reads like proof.

psephos is built so that is hard to do:

- **Nothing gets flagged without an innocent explanation printed next to it.** This is enforced
  in code. Constructing a flagged finding with no confounds raises an exception.
- **Precinct size is handled everywhere.** A station with 100 registered voters lands on a whole
  percentage by arithmetic. Percentage checks exclude small units, say how many they excluded,
  and run across a sweep of thresholds, because a signal that appears only at the most permissive
  threshold is a signal about small precincts.
- **There is no fraud score.** No single number, no traffic light, nothing a headline can quote.
  Findings carry a flag meaning "worth explaining", not "guilty".
- **The interpretation section cannot be suppressed.** It is in the text report and in the JSON.
- **Underpowered is a distinct outcome.** When too few units survive the filters, the check says
  so instead of returning a reassuring null.
- **Checks that would overclaim are deliberately absent.** No first-digit Benford test (precinct
  counts do not span enough orders of magnitude for it to mean anything), and no conversion of
  turnout-share dependence into a count of "stolen votes" until that estimator is validated
  against its source.

## Install

```console
git clone https://github.com/Xocas12/psephos
cd psephos
uv sync            # or: pip install -e ".[parquet]"
uv run psephos --help
```

Python 3.10 or newer. Dependencies are numpy, scipy and pandas.

## Use

Start by looking at what your file has:

```console
psephos columns results.csv
```

Then audit it. Auto-detection is a convenience; name the columns explicitly for anything you
intend to rely on:

```console
psephos audit results.csv \
  --registered "Registered voters" \
  --ballots-cast "Ballots in boxes" \
  --votes "Candidate A,Candidate B" \
  --precinct-id "Station" --region "Region" \
  --json report.json
```

For anything with more than a handful of contestants, put the mapping in a file instead. The
point is not brevity: a map file records decisions that change the answer, most obviously which
of several denominators counts as turnout, and a file can be committed, diffed and argued with
where a command line buried in shell history cannot.

```console
psephos columns results.csv --write-map draft.yaml   # a starting point, with every guess noted
$EDITOR draft.yaml                                   # correct the guesses
psephos audit results.csv --map draft.yaml
```

Map files need the `yaml` extra (`pip install 'psephos[yaml]'`); the core install stays on
numpy, scipy and pandas.

As a library:

```python
from psephos import ColumnMap, ElectionData, audit, to_text
import pandas as pd

data = ElectionData(
    frame=pd.read_csv("results.csv"),
    columns=ColumnMap(
        registered="registered",
        ballots_cast="ballots_cast",
        votes={"A": "party_a", "B": "party_b"},
    ),
)
report = audit(data)
print(to_text(report))
for f in report.flagged:
    print(f.check, f.pvalue, f.confounds)
```

## What it checks

**Integrity**, first, because a structural problem explains most surprising statistics:
duplicate units, negative counts, missing values, turnout above 100 per cent, and votes
exceeding ballots counted.

**Integer percentages.** Whether turnout and vote-share percentages pile up on whole numbers
more than counting noise allows. The null redraws each unit's count as binomial around its own
observed rate, which holds precinct size fixed. This is the strongest check here. Following
Kobak, Shpilkin and Pshenichnikov (2016).

**Last digits.** Whether the last digit of vote counts is uniform. Numbers invented by people
are not. Following Beber and Scacco (2012), with the caveat that psephos reports the direction
of the last-two-digit signature descriptively rather than as a test, pending validation.

**Turnout and vote share.** How strongly the leading contestant's share rises with turnout,
measured three ways. Reported as dependence, not as a vote count.

**By precinct size.** The percentage checks again, within size bands, so you can see whether a
signal lives in the small units.

## What it does not do

- Tell you whether an election was fraudulent.
- Produce a single score.
- Estimate a number of anomalous votes.
- Run a first-digit Benford test, on purpose. Precinct counts do not span the orders of
  magnitude the first-digit law needs, so the test rejects honest data and the p-value looks
  like proof. The argument, with a demonstration on honest data this repository generated, is
  in [docs/why_no_benford.md](docs/why_no_benford.md).
- Compare your election against others. **This is the biggest gap**: a statistic from one
  election means little without a baseline. See issue #4.
- Handle multi-round, multi-member or preferential systems properly yet. See issue #11.

## Honest status

Version 0.1. The methods are implemented from published descriptions and tested against
synthetic elections where the answer is known by construction: the false-positive rate on clean data must sit near its nominal level,
injected rounding must be found, and the measured effect must rise with the injected amount.

**They have not yet been validated against a real election with a known, published result.**
That is issue #6 and it is the most important open item in the repository. Until it is done,
treat psephos as a tool for generating questions, not answers.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: it states the bar a new check has to clear,
and `src/psephos/methods/_template.py` is a working example that clears it.

Issues are organised by milestone. Good places to start are labelled `good first issue`.

If you are adding a check, it needs: a null you can state in one sentence, a list of confounds,
a synthetic test showing it stays quiet on clean data, and a synthetic test showing it finds the
thing it claims to find. A check without the quiet test will not be merged.

## Licence

MIT. See [LICENSE](LICENSE).

## Related repositories

Five repositories, one programme. All public.

| Repository | What it is | State |
|---|---|---|
| [psephos](https://github.com/Xocas12/psephos) | standalone election anomaly-detection tool, CLI and library | **works**; suite green |
| [forensics-core](https://github.com/Xocas12/forensics-core) | the shared research method library | scaffold; 605 tests, no analysis run |
| [forensic-elections](https://github.com/Xocas12/forensic-elections) | Russian federal elections, the calibration project | scaffold; 736 tests, no analysis run |
| [forensic-economy](https://github.com/Xocas12/forensic-economy) | accounting enforcement, Chinese provincial statistics, Soviet statistics | scaffold; 995 tests, no analysis run |
| [gosplan-env](https://github.com/Xocas12/gosplan-env) | multi-agent environment where reporting pathologies emerge from incentives | skeleton; nothing run |

The four research repositories work under a discipline that forbids running anything before its
gate, and none of them contains a result. psephos is deliberately the opposite: it is meant to
be run today.
