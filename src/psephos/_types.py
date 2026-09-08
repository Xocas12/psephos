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
    n_used : int
        Units that entered the statistic.
    n_excluded : int
        Units deliberately dropped, and ``details["excluded_because"]`` says why.
    slice_name : str
        Which subset this was computed on, e.g. ``"all"`` or ``"registered 250-999"``.
    confounds : list of str
        Innocent explanations that produce this same signal. Never empty for a flagged
        finding: if nobody can name one, the check is not ready to be trusted.
    details : dict
        Everything else worth keeping.
    """

    check: str
    title: str
    flag: Flag
    statistic: float | None = None
    pvalue: float | None = None
    effect: float | None = None
    n_used: int = 0
    n_excluded: int = 0
    slice_name: str = "all"
    confounds: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.flag in (Flag.NOTABLE, Flag.STRONG) and not self.confounds:
            raise ValueError(
                f"{self.check}: a flagged finding must name at least one confound. "
                "If none is known, the check is not ready to be reported."
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        p = "" if self.pvalue is None else f" p={self.pvalue:.3g}"
        return f"[{self.flag.value}] {self.check} ({self.slice_name}): {self.title}{p}"


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
