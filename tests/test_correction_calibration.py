"""The p a re-tested hypothesis enters the correction with must be a p-value.

The audit re-tests two hypotheses on several slices: turnout at four size thresholds and in
every disjoint size band with at least 200 precincts (four on these elections), and the
winner's share at four thresholds. Each enters
Benjamini-Hochberg once, through one combined p. For that number to mean anything it has to
behave like a p-value on clean data: at or below 0.05 in no more than about 5 per cent of
audits. The smallest raw p among the re-tests, which is what the correction used to enter
with, did not: on the 200 elections below it was at or below 0.05 in 40 audits for turnout and
23 for the winner's share, because a hypothesis re-tested on K slices got K chances to look
significant and was charged for one. The Simes combination it enters with now was at or below
0.05 in 1 and 5.

This is a measurement over many audits, so it is slow by the standards of the suite and kept in
its own file. docs/multiple_testing.md records the same measurement at the default n_mc.
"""

from __future__ import annotations

import importlib

from conftest import clean_election
from scipy.stats import binomtest

from psephos.audit import audit
from psephos.schema import ElectionData

#: Clean audits measured. The bar is a binomial test of the share at or below 0.05 against 5
#: per cent, and at 200 audits it sees the old entering p's excess for both hypotheses, not
#: only for turnout: 40 of 200 is p = 6e-14, and 23 of 200 is p = 2e-4.
N_AUDITS = 200

#: Monte Carlo replicates per check. The Monte Carlo p-value cannot go below 1 / (n_mc + 1), so
#: too few replicates would leave no p small enough to reach 0.05 after combination and the test
#: would pass whatever the correction did. 100 puts that floor near 0.01, and it is the
#: setting the defect was first measured at.
N_MC = 100


def test_a_re_tested_hypothesis_enters_the_correction_at_no_more_than_its_nominal_rate(
    monkeypatch, column_map
):
    """Over 200 clean elections, the p with which integer_pct:turnout and the winner's
    integer_pct share enter Benjamini-Hochberg is at or below 0.05 in a share of audits that a
    binomial test cannot tell from 5 per cent, or below it, at the 0.01 level.

    The test is one-sided on purpose. What the correction needs is a valid p-value, P(p <= a)
    <= a under the null, and a conservative one passes that. A two-sided test would also fail
    valid but conservative combinations such as Bonferroni, and the slices' own Monte Carlo
    p-values are already a little conservative: 3.2 to 4.3 per cent at or below 0.05 over 440
    clean audits at the default n_mc, in docs/multiple_testing.md.
    """
    # Integrity findings carry no p-value and never join the family, and the row-by-row
    # total-row scan is most of an audit's time on a 4,000-precinct file. psephos.audit is
    # imported as a module because the package exports a function of the same name.
    monkeypatch.setattr(importlib.import_module("psephos.audit"), "run_integrity", lambda d: [])

    hits: dict[str, int] = {}
    members: dict[str, set[int]] = {}
    for s in range(N_AUDITS):
        data = ElectionData(
            frame=clean_election(n=4000, seed=s), columns=column_map, source="clean"
        )
        rep = audit(data, n_mc=N_MC, n_boot=0, seed=s)
        winner = rep.meta["winner"]
        records = {h["hypothesis"]: h for h in rep.meta["multiple_testing"]["hypotheses"]}
        for key in ("integer_pct:turnout", f"integer_pct:{winner} share"):
            hits[key] = hits.get(key, 0) + (records[key]["p"] <= 0.05)
            members.setdefault(key, set()).add(records[key]["n_findings"])

    # The measurement is only about the defect if the re-tests are there: four thresholds and
    # four size bands for turnout, four thresholds for the share.
    assert members["integer_pct:turnout"] == {8}, members
    assert members["integer_pct:party_incumbent share"] == {4}, members

    too_often = []
    for key, k in hits.items():
        p = binomtest(k, N_AUDITS, 0.05, alternative="greater").pvalue
        if p <= 0.01:
            too_often.append(
                f"{key} entered at or below 0.05 in {k} of {N_AUDITS} clean audits "
                f"({k / N_AUDITS:.1%}, binomial test p = {p:.2g})"
            )
    assert not too_often, "; ".join(too_often) + ". A valid p-value does that in at most 5%."
