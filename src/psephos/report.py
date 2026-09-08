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
    if f.statistic is not None:
        bits.append(f"stat={f.statistic:.4g}")
    if f.pvalue is not None:
        bits.append(f"p={f.pvalue:.4g}")
    if f.adjusted_pvalue is not None:
        bits.append(f"adj_p={f.adjusted_pvalue:.4g}")
    if f.effect is not None:
        bits.append(f"effect={f.effect:.4g}")
    bits.append(f"n={f.n_used}")
    if f.n_excluded:
        bits.append(f"excluded={f.n_excluded}")
    lines.append(f"{indent}    " + "  ".join(bits))
    for c in f.confounds:
        lines.append(f"{indent}    - could also be: {c}")
    return lines


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
