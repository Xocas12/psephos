"""Each check the audit runs flags clean data near its nominal rate, column by column.

Quietness is a rate, not a single draw. At alpha 0.05 an honest check flags about one clean
election in twenty, so a test that demands OK on one seed asserts that a correct check got
lucky, and a test that runs a few seeds and accepts NOTABLE on all of them asserts nothing about
the rate at all. Each test here runs one check on the 400 independent clean elections of
``CLEAN_RATE_SEEDS`` and allows at most 30 flags (``assert_clean_rate`` in conftest.py, where
the bound is argued): a check at the nominal 5 per cent passes with probability 0.989, one at
10 per cent fails with probability 0.948.

The tests are per column, not pooled. An audit-wide rate lets one bad column disappear among
the rest: the last-digit check flags honest minor-party counts at up to 2.7 times nominal (issue
#25), and the audit-wide test in test_audit.py never noticed. The same cut miscalibrates the
last-two-digit check where the winner's counts are dense across it. Those cases are marked as
expected failures until #25 is fixed. Where the excess is far past the bound (``party_third``,
45 to 54 flags of 400, and the straddling winner, 61 to 64) the marker is strict, so it fails
the suite the moment the case is calibrated and cannot outlive the fix. ``party_second`` sits
at the bound, 32 flags of 400 under numpy 2.5 and 27 under 2.2 and 2.4, whose binomial streams
differ, so its marker cannot be strict without failing half the CI matrix.

What is counted is what the audit runs, with two economies that do not touch the flag. Intervals
are off (``n_boot=0``): no flag depends on an interval. And the integer-percentage null draws
39 replicates rather than the audit's 500. Its add-one p-value then reaches 0.05 exactly, at
(1 + 1) / 40, so a calibrated null flags at most 5 per cent of clean elections at this n_mc as
at any other; a smaller n_mc makes the p-value coarser, not the test looser. That check runs at
the lowest and the highest threshold of the audit's sweep: the thresholds between select nested
sets of the same units under the same null, and on this generator, whose precincts start at 300
registered, 100 and 250 select exactly the same units.

``test_every_hypothesis_the_audit_tests_has_a_rate_test`` keeps the list complete: a check or a
column the audit gains without a rate test here fails it.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from conftest import (
    CLEAN_RATE_ALPHA,
    CLEAN_RATE_MAX_FLAGGED,
    CLEAN_RATE_MAX_STRONG,
    CLEAN_RATE_SEEDS,
    assert_clean_rate,
    clean_election,
)
from scipy import stats

from psephos import audit
from psephos.audit import DEFAULT_THRESHOLDS
from psephos.methods.digits import last_digit_uniformity, last_two_digit_pairs
from psephos.methods.integer_pct import integer_excess
from psephos.methods.turnout import turnout_share_dependence
from psephos.schema import ColumnMap, ElectionData

COLUMNS = ColumnMap(
    votes={
        "party_incumbent": "party_incumbent",
        "party_second": "party_second",
        "party_third": "party_third",
    },
    registered="registered",
    ballots_cast="ballots_cast",
    precinct_id="precinct",
    region="region",
)

#: The integer-percentage null's replicates here; see the module docstring.
N_MC = 39

#: The lowest and highest size threshold of the audit's sweep.
THRESHOLDS = (min(DEFAULT_THRESHOLDS), max(DEFAULT_THRESHOLDS))

#: The clean generator's precincts are too small for the last-two-digit check: it needs 500
#: counts of at least 1000, and the winner reaches 1000 in 4 to 18 of the 4,000 precincts, so
#: the audit's finding there is always UNDERPOWERED. Larger precincts and a larger winner give
#: it enough, in two shapes, because where the counts sit relative to that cut of 1000 decides
#: whether this check is calibrated (issue #25 again). From 2,500 registered the winner's
#: counts mostly lie well above the cut, about 3,400 usable per election; from 1,500 they are
#: dense across it, about 1,450 usable. Still clean_election: every count is a binomial draw.
_LARGER = dict(max_size=12000, turnout_mean=0.75, winner_mean=0.6)
LARGE_PRECINCTS = {
    "counts_above_the_cut": dict(min_size=2500, **_LARGER),
    "counts_across_the_cut": dict(min_size=1500, **_LARGER),
}

_ISSUE_25 = (
    "issue #25: the digit checks keep counts from min_count up and test them against a flat "
    "uniform, and an honest count distribution cut where it is dense has uneven endings"
)
KNOWN_ISSUE_25 = pytest.mark.xfail(strict=True, reason=_ISSUE_25)
#: Over the bound under one numpy's random streams and under it under another; see above.
KNOWN_ISSUE_25_AT_THE_BOUND = pytest.mark.xfail(strict=False, reason=_ISSUE_25)


@pytest.fixture(scope="module")
def clean_runs() -> dict[str, list]:
    """Each check's finding on each clean election, keyed by the hypothesis the audit files it
    under. Computed once for the module: every test below reads its own key."""
    runs: dict[str, list] = defaultdict(list)
    for seed in CLEAN_RATE_SEEDS:
        data = ElectionData(frame=clean_election(seed=seed), columns=COLUMNS, source="clean")
        winner = data.winner_label()
        reg = data.registered()
        cast = data.total_votes()
        winner_votes = data.frame[data.columns.votes[winner]].to_numpy(dtype=float)
        valid = data.votes_frame().sum(axis=1).to_numpy(dtype=float)

        for label, col in data.columns.votes.items():
            runs[f"last_digit:{label}"].append(
                last_digit_uniformity(data.frame[col].to_numpy(dtype=float), n_boot=0)
            )
        runs["turnout_share_dependence"].append(
            turnout_share_dependence(
                data.turnout(), data.share(winner, of="valid"), weights=valid, n_boot=0
            )
        )
        for t in THRESHOLDS:
            runs[f"integer_pct:turnout@{t}"].append(
                integer_excess(cast, reg, min_denominator=t, n_mc=N_MC, n_boot=0, seed=seed)
            )
            runs[f"integer_pct:{winner} share@{t}"].append(
                integer_excess(
                    winner_votes, valid, min_denominator=t, n_mc=N_MC, n_boot=0, seed=seed
                )
            )
    return runs


# ---------------------------------------------------------------- the bound itself


def test_the_bound_tells_nominal_from_twice_nominal():
    """The bound is only worth something if it separates 5 per cent from 10 per cent. Asserted
    on the binomial, so changing the number of elections or the bound cannot quietly turn it
    back into a test that a check at twice nominal passes."""
    n = len(CLEAN_RATE_SEEDS)
    k = CLEAN_RATE_MAX_FLAGGED
    assert stats.binom.cdf(k, n, CLEAN_RATE_ALPHA) >= 0.98, "a calibrated check fails too often"
    assert stats.binom.sf(k, n, 0.10) >= 0.90, "a check at twice nominal passes too often"
    assert stats.binom.sf(CLEAN_RATE_MAX_STRONG, n, 0.001) <= 0.01


# ---------------------------------------------------------------- one test per check and column


@pytest.mark.parametrize(
    "column",
    [
        "party_incumbent",
        pytest.param("party_second", marks=KNOWN_ISSUE_25_AT_THE_BOUND),
        pytest.param("party_third", marks=KNOWN_ISSUE_25),
    ],
)
def test_last_digit_clean_rate(clean_runs, column):
    assert_clean_rate(clean_runs[f"last_digit:{column}"], f"last digit of {column}")


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("column", ["turnout", "party_incumbent share"])
def test_integer_percentage_clean_rate(clean_runs, column, threshold):
    assert_clean_rate(
        clean_runs[f"integer_pct:{column}@{threshold}"],
        f"integer percentage of {column}, min_denominator={threshold}",
    )


def test_turnout_share_dependence_clean_rate(clean_runs):
    """This check also needs a rank correlation of at least 0.1 to flag, which on 4,000 clean
    precincts it essentially never reaches, so its flag count is near zero by construction. The
    p-value count is the one that tests it: the multiple-testing correction reads the p-value,
    not the flag."""
    assert_clean_rate(clean_runs["turnout_share_dependence"], "turnout and share dependence")


@pytest.mark.parametrize(
    "shape",
    [
        "counts_above_the_cut",
        pytest.param("counts_across_the_cut", marks=KNOWN_ISSUE_25),
    ],
)
def test_last_two_digits_clean_rate(shape):
    findings = []
    for seed in CLEAN_RATE_SEEDS:
        df = clean_election(seed=seed, **LARGE_PRECINCTS[shape])
        findings.append(last_two_digit_pairs(df["party_incumbent"].to_numpy(dtype=float), n_boot=0))
    assert_clean_rate(findings, f"last two digits of party_incumbent, {shape.replace('_', ' ')}")


# ---------------------------------------------------------------- coverage


#: Every hypothesis the audit tests, as ``psephos.correction`` groups them, with the rate test
#: above that covers it.
RATE_TESTED = {
    "integer_pct:turnout": test_integer_percentage_clean_rate,
    "integer_pct:party_incumbent share": test_integer_percentage_clean_rate,
    "turnout_share_dependence": test_turnout_share_dependence_clean_rate,
    "last_digit:party_incumbent": test_last_digit_clean_rate,
    "last_digit:party_second": test_last_digit_clean_rate,
    "last_digit:party_third": test_last_digit_clean_rate,
    "last_two_digits:party_incumbent": test_last_two_digits_clean_rate,
}


def test_every_hypothesis_the_audit_tests_has_a_rate_test():
    """Every finding of a real audit that carries a p-value, or that could (the last-two-digit
    check is underpowered on this generator but is still filed under its hypothesis), must
    belong to a hypothesis with a rate test above. A new check, or a new column for an old one,
    fails here until it has one."""
    data = ElectionData(frame=clean_election(seed=CLEAN_RATE_SEEDS[0]), columns=COLUMNS)
    rep = audit(data, n_mc=20, n_boot=0, seed=0)
    hypotheses = {f.hypothesis for f in rep.findings if f.hypothesis is not None}
    untagged = [f.check for f in rep.findings if f.pvalue is not None and f.hypothesis is None]
    assert not untagged, f"findings with a p-value but no hypothesis: {untagged}"
    assert not [f.check for f in rep.integrity if f.pvalue is not None], (
        "an integrity check now carries a p-value and needs a rate test"
    )
    missing = sorted(hypotheses - set(RATE_TESTED))
    assert not missing, f"tested by the audit, with no clean-data rate test: {missing}"
    stale = sorted(set(RATE_TESTED) - hypotheses)
    assert not stale, f"rate tests for hypotheses the audit no longer tests: {stale}"
