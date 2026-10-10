"""The report leads with the effect and its interval, and explains what the interval is not.

Issue #16: "Report the interval, not the p-value, as the headline." With 95,000 precincts almost
any departure from a null is significant, so a line that opens with the p-value is a line that
makes a trivial effect look important.
"""

from __future__ import annotations

import json

from psephos._types import AuditReport, Finding, Flag
from psephos.audit import audit
from psephos.report import to_json, to_text


def _finding(**kw) -> Finding:
    base = dict(
        check="integer_percentage",
        title="t",
        flag=Flag.OK,
        effect=0.0123,
        ci_low=0.0045,
        ci_high=0.0201,
        pvalue=0.004,
        statistic=3.2,
        n_used=4000,
    )
    base.update(kw)
    return Finding(**base)


def test_the_effect_and_its_interval_are_printed_before_the_pvalue():
    report = AuditReport(source="s", n_units=4000)
    report.meta["intervals"] = {"level": 0.95, "n_boot": 1000, "n_findings_with_ci": 1}
    report.add(_finding(flag=Flag.NOTABLE, confounds=["something innocent"]))
    text = to_text(report)

    line = next(ln for ln in text.splitlines() if "effect=" in ln)
    assert "[0.0045, 0.0201]" in line
    assert line.index("effect=") < line.index("p="), "the effect must come first on the line"


def test_a_finding_without_an_interval_still_prints_its_effect():
    report = AuditReport(source="s", n_units=50)
    report.add(_finding(flag=Flag.NOTABLE, confounds=["c"], ci_low=None, ci_high=None))
    line = next(ln for ln in to_text(report).splitlines() if "effect=" in ln)
    assert "effect=0.0123" in line
    assert "[" not in line.split("effect=")[1].split()[0]


def test_the_interval_section_says_it_is_not_a_test():
    report = AuditReport(source="s", n_units=4000)
    report.meta["intervals"] = {"level": 0.95, "n_boot": 1000, "n_findings_with_ci": 2}
    report.add(_finding())
    text = to_text(report)

    assert "EFFECT INTERVALS" in text
    assert "NOT a second test" in text
    assert "1000 resamples" in text
    assert "95 per cent" in text
    # The two misreadings the section exists to block.
    assert "never contains zero" in text
    assert "null model is wrong" in text


def test_the_interval_section_is_absent_when_nothing_carried_an_interval():
    report = AuditReport(source="s", n_units=4000)
    report.meta["intervals"] = {"level": 0.95, "n_boot": 0, "n_findings_with_ci": 0}
    report.add(_finding(ci_low=None, ci_high=None))
    assert "EFFECT INTERVALS" not in to_text(report)


def test_an_audit_puts_intervals_on_its_findings_and_in_its_json(clean_data):
    report = audit(clean_data, n_mc=60, seed=0, n_boot=120, by_size=False)
    with_ci = [f for f in report.findings if f.ci_low is not None]
    assert with_ci, "the audit should produce intervals"
    assert report.meta["intervals"]["n_findings_with_ci"] == len(with_ci)
    assert report.meta["settings"]["n_boot"] == 120

    payload = json.loads(to_json(report))
    serialised = [f for f in payload["findings"] if f["ci_low"] is not None]
    assert len(serialised) == len(with_ci)
    for f in serialised:
        assert f["ci_low"] <= f["effect"] <= f["ci_high"]
        assert f["details"]["ci_level"] == 0.95


def test_no_bootstrap_leaves_an_audit_without_intervals(clean_data):
    report = audit(clean_data, n_mc=60, seed=0, n_boot=0, by_size=False)
    assert all(f.ci_low is None for f in report.findings)
    assert "EFFECT INTERVALS" not in to_text(report)


def test_the_audit_stratifies_its_intervals_by_precinct_size(clean_data):
    """Not a detail: every percentage effect here depends on precinct size, so a resample that
    changed the size mix would widen the interval with a quantity that is fixed and known."""
    report = audit(clean_data, n_mc=60, seed=0, n_boot=120, by_size=False)
    stratified = [
        f
        for f in report.findings
        if f.ci_low is not None and f.details.get("ci_stratified_by") == "precinct size"
    ]
    assert stratified, "the percentage and digit checks should all be stratified"
    assert all(f.details["ci_n_strata"] >= 2 for f in stratified)
