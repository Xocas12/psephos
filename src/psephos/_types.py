"""Result types. Every check in psephos returns Findings, never a verdict.

A Finding says: this statistic took this value, this is how unlikely that is under this null,
this many units went into it, and here is the most plausible innocent explanation. It never
says "fraud", because no statistic can.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Flag(str, Enum):
    """How much attention a finding deserves. Deliberately not a fraud scale."""

    OK = "ok"
    """Nothing unusual, or the test could not distinguish anything."""

    NOTABLE = "notable"
    """Departs from the null more than chance comfortably explains."""

    STRONG = "strong"
    """Departs far enough that a confound has to be argued for explicitly."""

    UNDERPOWERED = "underpowered"
    """Too few usable units for the test to say anything either way."""

    NOT_APPLICABLE = "not_applicable"
    """The data lacks the columns this check needs."""


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Flag):
        return v.value
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


@dataclass(frozen=True)
class Finding:
    """One statistic, computed on one slice of the data.

    Attributes
    ----------
    check : str
        Machine-readable check name, e.g. ``"integer_percentage"``.
    title : str
        One line a human can read without knowing the method.
    flag : Flag
        See :class:`Flag`. Never a fraud judgement.
    statistic : float or None
        The test statistic or point estimate.
    pvalue : float or None
        Under the stated null. ``None`` where the check has no null.
    effect : float or None
        Size of the departure in units the title explains, so that a large sample cannot make
        a trivial effect look important.
    ci_low, ci_high : float or None
        A confidence interval for ``effect``, from the stratified unit bootstrap of
        :mod:`psephos.interval`, with the coverage and the method in ``details``. ``None`` where
        the check has no interval or there were too few units to compute one.

        It is an interval on the precision of the effect, not a second test. An interval that
        excludes zero adds nothing to the p-value beside it, and for an effect that cannot be
        negative it never contains zero however honest the data.
    n_used : int
        Units that entered the statistic.
    n_excluded : int
        Units deliberately dropped, and ``details["excluded_because"]`` says why.
    slice_name : str
        Which subset this was computed on, e.g. ``"all"`` or ``"registered 250-999"``.
    hypothesis : str or None
        The hypothesis this finding re-tests, when the audit runs one hypothesis on several
        slices: the size-threshold sweep and the size strata each re-test one hypothesis.
        Findings sharing a hypothesis are counted once by the multiple-testing correction.
        Set by the audit, not by the check; ``None`` where the finding is a hypothesis of its
        own or tests nothing.
    confounds : list of str
        Innocent explanations that produce this same signal. Never empty for a flagged
        finding: if nobody can name one, the check is not ready to be trusted.
    details : dict
        Everything else worth keeping.
    adjusted_pvalue : float or None
        The p-value after the audit's multiple-testing correction, filled in by
        :func:`psephos.correction.apply_correction`. It is reported beside the raw p-value,
        never instead of it, and no flag is raised or withdrawn by it. ``None`` where the
        check produced no p-value.
    """

    check: str
    title: str
    flag: Flag
    statistic: float | None = None
    pvalue: float | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_used: int = 0
    n_excluded: int = 0
    slice_name: str = "all"
    hypothesis: str | None = None
    confounds: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    adjusted_pvalue: float | None = None

    def __post_init__(self) -> None:
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError(
                f"{self.check}: an interval needs both ends; got "
                f"ci_low={self.ci_low!r}, ci_high={self.ci_high!r}"
            )
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(
                f"{self.check}: interval ends are the wrong way round: "
                f"[{self.ci_low}, {self.ci_high}]"
            )
        if (
            self.ci_low is not None
            and self.effect is not None
            and not self.ci_low <= self.effect <= self.ci_high
        ):
            # Not a pedantic check. The naive percentile bootstrap on a distance statistic
            # produces exactly this, and an interval that does not contain the number it is an
            # interval for misleads a reader as directly as a wrong number would.
            raise ValueError(
                f"{self.check}: the interval [{self.ci_low}, {self.ci_high}] does not contain "
                f"the effect {self.effect} it is an interval for"
            )
        if self.flag in (Flag.NOTABLE, Flag.STRONG) and not self.confounds:
            raise ValueError(
                f"{self.check}: a flagged finding must name at least one confound. "
                "If none is known, the check is not ready to be reported."
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        p = "" if self.pvalue is None else f" p={self.pvalue:.3g}"
        ci = "" if self.ci_low is None else f" ci=[{self.ci_low:.3g}, {self.ci_high:.3g}]"
        return f"[{self.flag.value}] {self.check} ({self.slice_name}): {self.title}{ci}{p}"


@dataclass
class AuditReport:
    """Everything one audit produced."""

    source: str
    n_units: int
    findings: list[Finding] = field(default_factory=list)
    integrity: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def flagged(self) -> list[Finding]:
        return [f for f in self.findings if f.flag in (Flag.NOTABLE, Flag.STRONG)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_units": self.n_units,
            "meta": _jsonable(self.meta),
            "integrity": [f.to_dict() for f in self.integrity],
            "findings": [f.to_dict() for f in self.findings],
        }
