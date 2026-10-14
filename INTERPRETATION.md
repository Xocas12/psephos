# How to read psephos output

Read this before you say anything public on the basis of a psephos report.

## The one-sentence version

psephos tells you that a dataset departs from a statistical null model. It does not tell you
why, and "somebody cheated" is one explanation among several for every pattern it can find.

## What a flag means

| Flag | Meaning |
|---|---|
| `ok` | The statistic is where the null says it should be, or the check could not distinguish anything. |
| `notable` | Departs from the null more than chance comfortably explains. |
| `strong` | Departs far enough that a confound has to be argued for explicitly rather than waved at. |
| `underpowered` | Too few usable units. **This is not a clean bill of health.** |
| `not_applicable` | The data lacks the columns the check needs. |

`strong` does not mean fraud. It means the null model does not fit, and the null model is always
a simplification of a real electoral process.

## The four ways to be wrong with this tool

### 1. Treating an anomaly as evidence of fraud

Every check here has innocent explanations, and psephos prints them with each finding. They are
not boilerplate. Turnout genuinely correlates with partisanship in most countries. Vote counts
genuinely get rounded during aggregation in some administrations. Some figures genuinely are
estimates, or were back-calculated from a rounded percentage somewhere between the polling
station and your file.

The question a finding raises is "what produced this?", and fraud is one candidate answer that
has to compete with the others on evidence about how that election was actually administered.

### 2. Ignoring precinct size

This is the most common way to get election forensics wrong, and it produces confident,
publishable, entirely spurious results.

A polling station with 100 registered voters can only report turnout in whole percentage
points. Every attainable turnout is a round number. A station with 2,500 voters almost never
lands on one by chance. So any test for "too many round percentages" that pools all stations
together is partly measuring how many small stations a country has, which is a fact about its
geography and its rules on institutional voting, not about its honesty.

psephos handles this in the null rather than in the reading. Its integer-percentage check
compares each station against binomial noise at that station's own size, so a 100-voter station
is expected to land on a whole number every time and a 200-voter station half the time, and a
country full of small stations does not flag for having them. Small stations do not push the
check up. Where the null fits them badly, at turnout close to 100 per cent as in some
institutional precincts, it expects more whole numbers than honest stations produce, so they
push it down.

psephos also excludes the smallest units, reports how many it excluded, and runs the check
across several thresholds. The exclusion is about power, not false alarms: below about 250
voters, rounding barely changes how often a station lands on a whole number, so those stations
mostly dilute the test. **Look at the sweep, and read it as a map.** It shows where in the size
range the excess sits. It is not what separates rounding from arithmetic; the null at every
threshold already does that, by holding station size fixed. An excess that fades as the
threshold rises is an excess in the smaller stations the higher thresholds drop, which is where
rounding is often alleged, and it needs explaining like any other. One that holds at the
strictest threshold sits in the large stations.

### 3. Analysing one election with no baseline

A p-value of 0.001 in one election tells you the null does not fit. It does not tell you whether
that is unusual, because you have not looked at any other election.

The comparisons that would make a finding meaningful:

- the same country's previous elections, ideally including ones nobody disputes;
- other countries with similar electoral administration, ballot design and precinct sizes;
- within the same election, regions that differ in administration.

psephos does not yet ship a reference corpus, which is its most important limitation. Until it
does, you have to build the comparison yourself, and a report without one is a report about a
null model rather than about an election.

### 4. Running many checks and reporting the one that fired

A psephos audit runs roughly twenty checks across several slices. If nothing were wrong, some of
them would still flag. That is what a significance level means.

Report the whole audit, not the flagged findings alone. The text report shows the flagged count
against the total for exactly this reason, and `--verbose` prints all of them.

## What psephos deliberately refuses to do

**No single score.** There is no "fraud index", no percentage, no traffic light. Composite
scores get quoted without their caveats, and their weights are always arbitrary.

**No estimate of anomalous votes.** Converting a turnout-share relationship into "N stolen
votes" requires an estimator whose construction psephos has not validated against its source.
Inventing one would put an unearned number into a serious accusation.

**No first-digit Benford test.** Benford's law needs data spanning several orders of magnitude.
Precinct vote counts do not. The test is widely applied to election data anyway and produces
confident nonsense, so it is left out on purpose.

**No error exit on a flagged finding.** `psephos audit` exits 0 whether or not anything is
flagged, so that a script cannot quietly turn "anomaly" into "failure".

## If you are going to publish

1. Show the threshold sweep, not one threshold.
2. State how many checks you ran and how many flagged.
3. Give a baseline: another election, another country, or an earlier round.
4. State the confounds you considered and why you set them aside. If you cannot set them aside,
   say that instead.
5. Have someone who knows that country's electoral administration read it.

The cost of a false accusation is not symmetric with the cost of a missed one. A wrong claim of
fraud damages people who ran an honest election and corrodes trust that is hard to rebuild.

## Sources

The methods implemented here are described in:

- Kobak, Shpilkin and Pshenichnikov, "Integer percentages as electoral falsification
  fingerprints", *Annals of Applied Statistics* 10(1), 2016.
- Beber and Scacco, "What the Numbers Say: A Digit-Based Test for Election Fraud",
  *Political Analysis* 20(2), 2012.

psephos implements these from their published descriptions. It has not been line-by-line
validated against the authors' own code or reproduced their published figures on their data.
That work is tracked in issue #6, and until it is done every number here is provisional.
