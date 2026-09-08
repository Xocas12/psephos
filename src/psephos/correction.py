"""Multiple-testing correction across one audit.

An audit is a battery, not a test: run enough checks and some of them flag under a true null.
Measured over twenty independent clean elections from tests/conftest.py, one audit produced
1.4 flags on average against the 5 per cent nominal level of each individual check; the
measurement is pinned in tests/test_audit.py. The correction here adjusts for that, and it is
deliberately an adjustment rather than a verdict: the corrected value is reported beside the
raw p-value, never instead of it, and no flag is raised or withdrawn by it. What counts as one
family, and why the threshold sweep is one member rather than four, is argued in
docs/multiple_testing.md.

Method
------
The family is the set of distinct hypotheses the audit tests, not the set of findings. Each
finding names the hypothesis it re-tests in ``Finding.hypothesis``; findings that re-test one
hypothesis, such as the four size thresholds of one sweep, contribute the smallest raw p-value
among them and count once. Across hypotheses the correction is Benjamini-Hochberg, and the
adjusted p-value reported beside each raw one is the step-up envelope of the BH critical
values, so that a hypothesis whose adjusted value is at most ``q`` is exactly one the BH
procedure would reject at ``q``.

False discovery rate rather than family-wise error is the right contract for a screening tool
whose flags mean "worth explaining", not "guilty": it bounds the expected share of flagged
hypotheses that are false. A Bonferroni bound on the probability of even one false flag, over
a family of this size, would leave the tool near-blind and feed the reassuring-on-dirty-data
failure mode the ROADMAP warns about. The dependence caveats are argued in the document rather
than hidden here.

Reference: Benjamini, Y. and Hochberg, Y. (1995). "Controlling the false discovery rate: a
practical and powerful approach to multiple testing", Journal of the Royal Statistical Society,
Series B (Methodological) 57(1), 289-300. The variant of Benjamini and Yekutieli (2001), which
controls the rate under arbitrary dependence at the cost of a heavier correction, is
deliberately not used; docs/multiple_testing.md says why.
"""

from __future__ import annotations

import math
from dataclasses import replace

from psephos._types import AuditReport, Finding

#: The false-discovery-rate level the correction reports at. Not a fraud threshold and not an
#: error rate on the audit: it bounds the expected share of flagged hypotheses that are false
#: under the assumptions stated in docs/multiple_testing.md.
FDR_Q = 0.05

#: Where the family definition is argued. The report cites this path so the reasoning travels
#: with the output instead of living only in the code.
FAMILY_DOC = "docs/multiple_testing.md"

#: The method name as it appears in the report text and the JSON meta.
METHOD = "Benjamini-Hochberg"


def hypothesis_key(finding: Finding) -> str:
    """The family member a finding belongs to.

    The audit tags findings that re-test one hypothesis with an explicit slug. A finding
    without a tag, for example in a report assembled by hand, is treated as a hypothesis of
    its own, keyed by check and slice. Merging distinct hypotheses would understate the
    correction, so the fallback is per finding rather than per check.
    """
    return finding.hypothesis or f"{finding.check}: {finding.slice_name}"


def _usable_p(finding: Finding) -> float | None:
    """The finding's p-value if it is one a correction can use, else None.

    A finding without a usable p-value made no test: not applicable, failed, descriptive, or
    underpowered. Underpowered is the important one. It can read like a null result and is
    nothing of the kind, so it must not join the family and shrink the correction for tests
    that were actually made.
    """
    if finding.pvalue is None or not math.isfinite(finding.pvalue):
        return None
    return finding.pvalue


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Adjusted p-values for the Benjamini-Hochberg step-up procedure.

    With the p-values sorted ascending, the adjusted value of the i-th smallest is the
    monotone envelope of ``m * p_(i) / i`` taken from the largest down and capped at one.
    Each returned value is in the same position as the p-value it belongs to. The envelope is
    monotone in the sorted p-values, so it never ranks two hypotheses inconsistently with
    their raw ones.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, m * pvalues[idx] / rank, 1.0)
        adjusted[idx] = running
    return adjusted


def apply_correction(report: AuditReport, *, q: float = FDR_Q) -> AuditReport:
    """Fill in ``adjusted_pvalue`` on a finished report and record what was corrected.

    Findings are grouped by hypothesis; each hypothesis enters the correction through the
    smallest raw p among its re-tests, and every re-test of it is then reported with the
    hypothesis's adjusted value beside its own raw one. A finding with no p-value gets none:
    it made no test and has nothing to correct. The flags are left exactly as the checks set
    them.

    Returns the same report, so the audit can end with ``return apply_correction(report)``.
    """
    groups: dict[str, list[Finding]] = {}
    for f in report.findings:
        groups.setdefault(hypothesis_key(f), []).append(f)

    smallest: dict[str, float] = {}
    for key, members in groups.items():
        ps = [p for f in members if (p := _usable_p(f)) is not None]
        if ps:
            smallest[key] = min(ps)

    keys = sorted(smallest)
    adjusted = dict(zip(keys, benjamini_hochberg([smallest[k] for k in keys]), strict=True))

    report.findings = [
        replace(
            f,
            adjusted_pvalue=adjusted.get(hypothesis_key(f)) if _usable_p(f) is not None else None,
        )
        for f in report.findings
    ]
    report.meta["multiple_testing"] = {
        "method": METHOD,
        "q": q,
        "n_hypotheses": len(keys),
        "n_findings": len(report.findings),
        "n_findings_with_pvalue": sum(1 for f in report.findings if _usable_p(f) is not None),
        "doc": FAMILY_DOC,
        "hypotheses": [
            {
                "hypothesis": key,
                "p": smallest[key],
                "adjusted_pvalue": adjusted[key],
                "n_findings": len(groups[key]),
            }
            for key in keys
        ],
    }
    return report
