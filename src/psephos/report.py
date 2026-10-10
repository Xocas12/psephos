"""Rendering an audit as text or JSON.

The text report always ends with the interpretation section. That is not decoration: a list of
flagged statistics with no statement of what they do not mean is the single most likely way for
this tool to be misused, so the section is not optional and there is no flag to suppress it.
"""

from __future__ import annotations

import json
from typing import Any

from psephos._types import AuditReport, Finding, Flag

MARK = {
    Flag.OK: "  ok  ",
    Flag.NOTABLE: "NOTABLE",
    Flag.STRONG: "STRONG ",
    Flag.UNDERPOWERED: " weak ",
    Flag.NOT_APPLICABLE: "  n/a ",
}

INTERPRETATION = """\
HOW TO READ THIS

psephos detects statistical anomalies. It does not detect fraud, and no statistic can. Every
pattern it flags has innocent explanations, which are printed with each finding, and a flagged
result means one thing only: this dataset departs from a stated null in a way that is worth
explaining.

Four things that will make you wrong if you skip them.

1. Anomalies are not evidence of fraud on their own. They are evidence that a null model does
   not fit. The null is always a simplification of a real electoral process, and the process
   is what you have to argue about.

2. Precinct size drives most false alarms. Small precincts produce whole-number percentages by
   arithmetic. psephos excludes them and shows the result across several thresholds; a signal
   that only appears at the most permissive threshold is a signal about small precincts.

3. Comparison is what makes a number mean anything. A statistic from one election, with no
   comparison to other elections in the same country, to previous elections, or to countries
   with similar administration, cannot tell you whether the value is unusual. psephos gives you
   the statistic; the comparison is your job.

4. The direction of an effect matters and is often ambiguous. Turnout and partisanship are
   genuinely correlated in most countries. High turnout with a strong incumbent share is the
   normal condition of rural politics as well as a signature of ballot stuffing.

If you are going to say something public about an election on the basis of this output, get the
analysis reviewed by someone who knows that country's electoral administration. The cost of a
false accusation is not symmetric with the cost of a missed one.
"""


def _fmt_finding(f: Finding, indent: str = "  ") -> list[str]:
    lines = [f"{indent}[{MARK[f.flag]}] {f.check}  ({f.slice_name})"]
    lines.append(f"{indent}    {f.title}")
    bits = []
    # The effect and its interval come first on purpose. With 95,000 precincts almost any
    # departure from a null is significant, so the p-value is the least informative number on
    # the line and leading with it is how a trivial effect gets read as an important one.
    if f.effect is not None:
        if f.ci_low is not None and f.ci_high is not None:
            bits.append(f"effect={f.effect:.4g} [{f.ci_low:.4g}, {f.ci_high:.4g}]")
        else:
            bits.append(f"effect={f.effect:.4g}")
    if f.statistic is not None:
        bits.append(f"stat={f.statistic:.4g}")
    if f.pvalue is not None:
        bits.append(f"p={f.pvalue:.4g}")
    if f.adjusted_pvalue is not None:
        bits.append(f"adj_p={f.adjusted_pvalue:.4g}")
    bits.append(f"n={f.n_used}")
    if f.n_excluded:
        bits.append(f"excluded={f.n_excluded}")
    lines.append(f"{indent}    " + "  ".join(bits))
    for c in f.confounds:
        lines.append(f"{indent}    - could also be: {c}")
    return lines


def _interval_lines(report: AuditReport) -> list[str]:
    """What the bracketed numbers beside each effect are, and what they are not.

    Not optional and with no flag to suppress it, like the sections it sits beside. An interval
    is the most misreadable number in the output: it looks like a test, it is not one, and for a
    statistic that cannot be negative it excludes zero on perfectly honest data.
    """
    ci = report.meta.get("intervals")
    if not ci or not ci.get("n_findings_with_ci"):
        return []
    out = ["", "EFFECT INTERVALS"]
    out.append(
        f"  The bracketed range beside each effect is a {100 * ci['level']:.0f} per cent "
        f"percentile bootstrap interval, from {ci['n_boot']} resamples of the precincts within "
        "size bands. It holds the precinct-size mix at the one observed, because a whole-number "
        "percentage is common by arithmetic in a small station and rare in a large one."
    )
    out.append(
        "  It says how precisely the effect is measured, which is why it is printed first: the "
        "same effect on 300 precincts and on 30,000 is not the same finding. Read the effect "
        "and its width before the p-value."
    )
    out.append(
        "  It is NOT a second test. The p-value beside it is the test, against a stated null. "
        "An interval that excludes zero adds nothing to it, and an effect that cannot be "
        "negative -- a distance from uniform, for instance -- has an interval that never "
        "contains zero however honest the data; compare those against the expected value under "
        "the null, which the finding carries."
    )
    out.append(
        "  It covers uncertainty about which precincts were drawn, and nothing else. It does "
        "not widen for the possibility that the null model is wrong, which is the larger "
        "uncertainty in everything above."
    )
    return out


def _multiple_testing_lines(report: AuditReport) -> list[str]:
    """The section that says how many tests ran and what the correction did to them.

    A reader cannot judge a count of flags without knowing how many tests produced them, so
    this section is not optional and there is no flag to suppress it, like the interpretation
    section it precedes.
    """
    mt = report.meta.get("multiple_testing")
    if not mt:
        return []
    out = ["", "MULTIPLE TESTING"]
    if mt["n_hypotheses"] == 0:
        out.append("  No statistical test ran, so there is nothing to correct.")
        return out
    out.append(
        f"  The audit tested {mt['n_hypotheses']} hypotheses in {mt['n_findings']} findings, "
        f"{mt['n_findings_with_pvalue']} of which carried a p-value. Re-tests of one "
        "hypothesis, such as the four size thresholds or the size strata, are counted once, "
        "through the smallest raw p among them."
    )
    out.append(
        f"  Corrected with the {mt['method']} false-discovery-rate procedure at q = {mt['q']}. "
        "The adjusted p is printed beside the raw one and is a report, not a verdict: no flag "
        "is raised or withdrawn by it."
    )
    out.append(f"  What counts as one family, and why, is argued in {mt['doc']}.")
    flagged = report.flagged
    if flagged:
        survive = sum(
            1 for f in flagged if f.adjusted_pvalue is not None and f.adjusted_pvalue <= mt["q"]
        )
        out.append(
            f"  {survive} of the {len(flagged)} flags above keep an adjusted p at or below "
            f"q = {mt['q']}."
        )
    return out


def to_text(report: AuditReport, *, verbose: bool = False) -> str:
    """Human-readable report. Always includes the interpretation section."""
    out: list[str] = []
    out.append("=" * 78)
    out.append(f"psephos audit: {report.source}")
    out.append(f"{report.n_units} units")
    if report.meta.get("winner"):
        out.append(f"leading contestant: {report.meta['winner']}")
    out.append("=" * 78)

    notes = report.meta.get("columns", {}).get("notes") or []
    if notes:
        out.append("")
        out.append("COLUMN MAPPING")
        for n in notes:
            out.append(f"  - {n}")

    load_notes = report.meta.get("load_notes") or []
    if load_notes:
        out.append("")
        out.append("HOW THE FILE WAS READ")
        out.append(
            "  Each line is an assumption underneath every number below. A loader that repaired "
            "this file silently would have hidden them."
        )
        for n in load_notes:
            out.append(f"  - {n}")

    if report.meta.get("size_note"):
        out.append("")
        out.append("PRECINCT SIZE")
        out.append(f"  {report.meta['size_note']}")

    out.append("")
    out.append("INTEGRITY")
    for f in report.integrity:
        if verbose or f.flag is not Flag.OK:
            out.extend(_fmt_finding(f))
        else:
            out.append(f"  [{MARK[f.flag]}] {f.check}: {f.title}")

    flagged = report.flagged
    out.append("")
    out.append(f"FLAGGED FINDINGS ({len(flagged)} of {len(report.findings)})")
    if not flagged:
        out.append("  Nothing departed from its null enough to flag.")
    for f in flagged:
        out.extend(_fmt_finding(f))

    out.extend(_interval_lines(report))
    out.extend(_multiple_testing_lines(report))

    if verbose:
        out.append("")
        out.append("ALL FINDINGS")
        for f in report.findings:
            if f not in flagged:
                out.extend(_fmt_finding(f))

    out.append("")
    out.append("-" * 78)
    out.append(INTERPRETATION)
    return "\n".join(out)


def to_json(report: AuditReport, *, indent: int = 2) -> str:
    """Machine-readable report. Carries the interpretation text so it cannot be stripped by
    reading the JSON instead of the text."""
    payload: dict[str, Any] = report.to_dict()
    payload["interpretation"] = INTERPRETATION
    payload["disclaimer"] = (
        "psephos detects statistical anomalies, not fraud. A flagged finding means a dataset "
        "departs from a stated null model, and every finding lists innocent explanations."
    )
    return json.dumps(payload, indent=indent, ensure_ascii=False)
