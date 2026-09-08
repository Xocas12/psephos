"""The statistical checks.

Each returns a Finding rather than a bare number, so that a p-value never travels without the
sample size it came from and the confounds that could produce it.
"""

from psephos.methods import digits, integer_pct, turnout

__all__ = ["digits", "integer_pct", "turnout"]
