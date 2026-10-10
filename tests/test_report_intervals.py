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


# --------------------------------------------------------------------- the stated coverage


def test_the_title_states_the_coverage_that_was_actually_used():
    """--ci-level is plumbed through, so a title hardcoding "95 per cent" would contradict the
    report's own EFFECT INTERVALS section two sections below it."""
    import sys

    sys.path.insert(0, "tests")
    from conftest import clean_election

    from psephos.methods.integer_pct import integer_excess
    from psephos.methods.turnout import turnout_share_dependence

    df = clean_election(2000)
    reg = df["registered"].to_numpy(float)
    cast = df["ballots_cast"].to_numpy(float)

    at_95 = integer_excess(cast, reg, n_mc=40, seed=0, n_boot=120, ci_level=0.95)
    at_80 = integer_excess(cast, reg, n_mc=40, seed=0, n_boot=120, ci_level=0.80)
    assert "95 per cent interval" in at_95.title
    assert "80 per cent interval" in at_80.title
    assert "95 per cent" not in at_80.title
    # And the narrower coverage really is a narrower interval.
    assert (at_80.ci_high - at_80.ci_low) < (at_95.ci_high - at_95.ci_low)

    dep_80 = turnout_share_dependence(
        100.0 * cast / reg,
        100.0 * df["party_incumbent"].to_numpy(float) / cast,
        weights=cast,
        sizes=reg,
        n_boot=120,
        ci_level=0.80,
    )
    assert "80 per cent interval" in dep_80.title
    assert "95 per cent" not in dep_80.title


def test_every_effect_in_a_default_audit_carries_an_interval(clean_data):
    """README says "an interval on every effect". That has to be true of shipped output, not
    only of the three checks the issue named."""
    report = audit(clean_data, n_mc=40, seed=0, n_boot=120)
    with_effect = [f for f in report.findings if f.effect is not None]
    missing = [(f.check, f.slice_name) for f in with_effect if f.ci_low is None]
    assert with_effect, "the audit should report effects at all"
    assert missing == [], f"effects with no interval: {missing}"


def test_turnout_distribution_got_an_interval(clean_data):
    """It reports an effect - the share of units at or above 95 per cent turnout - so it falls
    under the same bar, even though it is labelled descriptive."""
    report = audit(clean_data, n_mc=40, seed=0, n_boot=120)
    found = [
        f for f in report.findings if f.check == "turnout_distribution" and f.effect is not None
    ]
    assert found, "turnout_distribution not in the audit"
    for f in found:
        assert f.ci_low is not None
        assert f.ci_low <= f.effect <= f.ci_high
        assert f.ci_low >= 0.0, "a share cannot be negative"


def test_last_two_digit_pairs_got_an_interval():
    """Driven directly: on the clean generator this check is underpowered, because it needs 500
    units with counts of at least 1000, so an audit of that fixture never gives it an effect."""
    import numpy as np

    from psephos.methods.digits import last_two_digit_pairs

    rng = np.random.default_rng(0)
    counts = rng.integers(1000, 90_000, 1500).astype(float)
    f = last_two_digit_pairs(counts, n_boot=120)
    assert f.effect is not None, "this fixture should be powered enough to report an effect"
    assert f.ci_low is not None
    assert f.ci_low <= f.effect <= f.ci_high
    # The effect is signed (repeated-pair share minus the uniform 0.10), so unlike a distance
    # its interval can and here should straddle zero on uniform digits.
    assert f.ci_low < 0.0 < f.ci_high


def test_the_template_check_clears_the_bar_it_documents():
    """_template.py is the canonical example of CONTRIBUTING.md's seven items, one of which is
    now an interval. An example that violates the bar it illustrates teaches the wrong thing."""
    import numpy as np

    from psephos.methods._template import zero_five_ending_share

    rng = np.random.default_rng(0)
    counts = rng.integers(200, 5000, 1200).astype(float)
    f = zero_five_ending_share(counts, n_boot=120)
    assert f.effect is not None
    assert f.ci_low is not None and f.ci_low <= f.effect <= f.ci_high


def test_the_html_report_orders_its_bits_the_way_it_says_it_does(clean_data):
    """html_report.py claims "same order as the text report". It emitted stat= first."""
    from psephos.html_report import to_html

    report = audit(clean_data, n_mc=40, seed=0, n_boot=120)
    html = to_html(report, clean_data)
    for f in report.findings:
        if f.effect is None or f.statistic is None:
            continue
        needle = f"effect={f.effect:.4g}"
        if needle not in html:
            continue
        chunk = html[html.index(needle) - 200 : html.index(needle)]
        assert "stat=" not in chunk, "stat= must not precede effect= in the HTML report"
        break
