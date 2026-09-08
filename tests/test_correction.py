"""The multiple-testing correction: one family, counted honestly, reported beside the raw p.

The correction is a report, not a verdict, so the tests pin the two decisions that carry the
substance: re-tests of one hypothesis, such as the size-threshold sweep, are counted once
through their smallest raw p, and each contestant's digit test stands on its own. They also
pin that no flag ever moves. docs/multiple_testing.md carries the argument; these tests keep
the code agreeing with it.
"""

from __future__ import annotations

import json

import pytest
from conftest import with_rounded_turnout

from psephos._types import AuditReport, Finding, Flag
from psephos.audit import audit
from psephos.correction import apply_correction, benjamini_hochberg, hypothesis_key
from psephos.report import to_json, to_text
from psephos.schema import ColumnMap, ElectionData


def _finding(
    p: float | None,
    *,
    flag: Flag = Flag.OK,
    hypothesis: str | None = None,
    check: str = "x",
    slice_name: str = "all",
) -> Finding:
    return Finding(
        check=check,
        title="t",
        flag=flag,
        pvalue=p,
        slice_name=slice_name,
        hypothesis=hypothesis,
        confounds=["c"] if flag in (Flag.NOTABLE, Flag.STRONG) else [],
    )


# ---------------------------------------------------------------- the arithmetic


def test_bh_adjusted_p_hand_computed():
    """Values small enough to check by hand. With m = 4 and all p-values at or below their BH
    line, the envelope flattens every adjusted value onto the largest; with a spread-out
    family, each keeps its own envelope step."""
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert benjamini_hochberg([0.01, 0.5, 0.9]) == pytest.approx([0.03, 0.75, 0.9])
    assert benjamini_hochberg([0.6, 0.8, 1.0]) == pytest.approx([1.0, 1.0, 1.0])
    assert benjamini_hochberg([0.3]) == pytest.approx([0.3])
    assert benjamini_hochberg([]) == []


def test_bh_adjusted_values_land_in_input_order():
    """The envelope is computed on the sorted p-values but must come back where they came
    from, or a report would attach one hypothesis's correction to another's finding."""
    adj = benjamini_hochberg([0.5, 0.01, 0.9, 0.2])
    assert adj == pytest.approx([4 * 0.5 / 3, 4 * 0.01, 0.9, 4 * 0.2 / 2])


def test_bh_is_monotone_in_the_raw_p_and_never_below_it():
    ps = [0.9, 0.04, 0.5, 0.04, 0.75, 0.2, 0.33, 0.61]
    adj = benjamini_hochberg(ps)
    for a, p in zip(adj, ps, strict=True):
        assert p <= a <= 1.0
    ranked = sorted(zip(ps, adj, strict=True))
    assert [a for _, a in ranked] == sorted(a for _, a in ranked)


# ---------------------------------------------------------------- the family


def test_a_hypothesis_gets_one_adjusted_value_from_its_smallest_raw_p():
    rep = AuditReport(
        source="s",
        n_units=1,
        findings=[
            _finding(0.2, hypothesis="h"),
            _finding(0.004, hypothesis="h"),
            _finding(0.4),
        ],
    )
    apply_correction(rep)
    # Two hypotheses: the re-tests of h enter through 0.004, the untagged one through 0.4.
    assert rep.findings[0].adjusted_pvalue == pytest.approx(2 * 0.004)
    assert rep.findings[1].adjusted_pvalue == pytest.approx(2 * 0.004)
    assert rep.findings[2].adjusted_pvalue == pytest.approx(0.4)


def test_findings_without_a_pvalue_made_no_test_and_join_no_family():
    """Underpowered and not-applicable findings read like results and are nothing of the kind;
    counting them would shrink the correction for tests that were actually made."""
    rep = AuditReport(
        source="s",
        n_units=1,
        findings=[
            _finding(None, flag=Flag.UNDERPOWERED),
            _finding(None, flag=Flag.NOT_APPLICABLE),
            _finding(0.1),
        ],
    )
    apply_correction(rep)
    mt = rep.meta["multiple_testing"]
    assert mt["n_hypotheses"] == 1
    assert mt["n_findings_with_pvalue"] == 1
    assert rep.findings[0].adjusted_pvalue is None
    assert rep.findings[2].adjusted_pvalue == pytest.approx(0.1), "m = 1: adjusted equals raw"


def test_untagged_findings_fall_back_to_one_hypothesis_each():
    """Findings from outside the audit get no assumption: per finding, never merged per check,
    because merging distinct hypotheses would understate the correction."""
    rep = AuditReport(
        source="s",
        n_units=1,
        findings=[
            _finding(0.01, check="c", slice_name="s1"),
            _finding(0.9, check="c", slice_name="s2"),
        ],
    )
    apply_correction(rep)
    assert rep.meta["multiple_testing"]["n_hypotheses"] == 2
    assert hypothesis_key(rep.findings[0]) == "c: s1"


def test_a_report_where_every_check_was_underpowered_has_nothing_to_correct():
    rep = AuditReport(source="s", n_units=1, findings=[_finding(None, flag=Flag.UNDERPOWERED)])
    apply_correction(rep)
    assert rep.meta["multiple_testing"]["n_hypotheses"] == 0
    assert "No statistical test ran" in to_text(rep)


# ---------------------------------------------------------------- the report


def test_the_correction_never_raises_or_withdraws_a_flag():
    """The contract the docs argue for: the adjusted p travels beside the raw one, and the
    flags are exactly what the checks set, supported or not."""
    rep = AuditReport(
        source="s",
        n_units=1,
        findings=[
            _finding(0.9, flag=Flag.NOTABLE, check="c"),
            _finding(0.0001, flag=Flag.STRONG, check="d"),
        ],
    )
    before = [f.flag for f in rep.findings]
    apply_correction(rep)
    assert [f.flag for f in rep.findings] == before
    # Benjamini-Hochberg leaves the LARGEST p unchanged: its rank factor is n/n = 1. An
    # earlier version of this test expected 1.0, which is Bonferroni-like behaviour and not
    # what a false-discovery-rate procedure does.
    assert rep.findings[0].adjusted_pvalue == pytest.approx(0.9)
    assert rep.findings[1].adjusted_pvalue == pytest.approx(0.0002)


def test_the_threshold_sweep_is_one_hypothesis_and_the_digit_tests_are_not(clean_data):
    """The two family decisions the docs argue. The sweep is four correlated re-tests of one
    claim and must not multiply the family; each contestant's digits are their own claim."""
    rep = audit(clean_data, thresholds=(100, 250), by_size=False, n_mc=60)
    mt = rep.meta["multiple_testing"]

    turnout = [f for f in rep.findings if f.hypothesis == "integer_pct:turnout"]
    assert len(turnout) == 2, "both thresholds should be tagged as re-tests of one hypothesis"
    assert len({f.adjusted_pvalue for f in turnout}) == 1, "re-tests share one adjusted value"

    share = [f for f in rep.findings if f.hypothesis == "integer_pct:party_incumbent share"]
    assert len(share) == 2

    digits = {f.hypothesis for f in rep.findings if f.check == "last_digit"}
    assert len(digits) == len(clean_data.columns.votes), "per-contestant tests stand alone"

    assert mt["n_hypotheses"] == len(mt["hypotheses"])
    assert mt["n_hypotheses"] < mt["n_findings_with_pvalue"], "re-tests did not multiply the family"


def test_every_p_value_keeps_an_adjusted_one_beside_it(clean_data):
    rep = audit(clean_data, n_mc=80, seed=0)
    # The adjusted value belongs to the HYPOTHESIS, not the individual finding: re-tests of
    # one hypothesis enter the correction once, through the smallest p among them. So
    # `f.pvalue <= f.adjusted_pvalue` is FALSE by design for any member that is not the
    # smallest, because a sweep entry at min_denominator=1000 carries the adjusted value of a
    # sweep that entered at min_denominator=100. The invariant that does hold is against the
    # value that actually entered.
    entered = {h["hypothesis"]: h["p"] for h in rep.meta["multiple_testing"]["hypotheses"]}
    for f in rep.findings:
        if f.pvalue is None:
            assert f.adjusted_pvalue is None
            continue
        assert f.adjusted_pvalue is not None
        assert entered[hypothesis_key(f)] <= f.adjusted_pvalue <= 1.0


def test_the_report_states_how_many_tests_ran(clean_data):
    rep = audit(clean_data, by_size=False, n_mc=60)
    text = to_text(rep)
    mt = rep.meta["multiple_testing"]
    assert "MULTIPLE TESTING" in text
    assert f"tested {mt['n_hypotheses']} hypotheses in {mt['n_findings']} findings" in text
    assert "docs/multiple_testing.md" in text
    assert "no flag is raised or withdrawn by it" in text


def test_the_text_report_shows_the_adjusted_p_beside_the_raw_one(clean_df, column_map):
    dirty = ElectionData(
        frame=with_rounded_turnout(clean_df, fraction=0.15),
        columns=column_map,
        source="synthetic-rounded",
    )
    rep = audit(dirty, n_mc=200, seed=0)
    flagged = rep.flagged
    assert flagged, "this fixture is expected to flag; the test needs a flagged finding"
    text = to_text(rep)
    for f in flagged:
        assert f"adj_p={f.adjusted_pvalue:.4g}" in text, "the adjusted p must be printable"
    q = rep.meta["multiple_testing"]["q"]
    survive = sum(1 for f in flagged if f.adjusted_pvalue is not None and f.adjusted_pvalue <= q)
    assert f"{survive} of the {len(flagged)} flags above keep an adjusted p" in text


def test_the_json_carries_the_correction_and_the_family_record(clean_data):
    payload = json.loads(to_json(audit(clean_data, by_size=False, n_mc=60)))
    mt = payload["meta"]["multiple_testing"]
    assert mt["method"] == "Benjamini-Hochberg"
    assert mt["q"] == 0.05
    assert mt["doc"] == "docs/multiple_testing.md"
    assert len(mt["hypotheses"]) == mt["n_hypotheses"]
    for h in mt["hypotheses"]:
        assert 0.0 <= h["p"] <= h["adjusted_pvalue"] <= 1.0
        assert h["n_findings"] >= 1
    p_bearing = [f for f in payload["findings"] if f["pvalue"] is not None]
    assert p_bearing and all(f["adjusted_pvalue"] is not None for f in p_bearing)
    assert all("hypothesis" in f for f in payload["findings"])


def test_an_audit_with_no_mapped_contestants_has_nothing_to_correct(clean_df):
    rep = audit(ElectionData(frame=clean_df, columns=ColumnMap(votes={}), source="s"), n_mc=30)
    assert rep.meta["multiple_testing"]["n_hypotheses"] == 0
    assert "No statistical test ran" in to_text(rep)
