"""psephos: statistical anomaly detection for precinct-level election results.

It flags patterns that warrant an explanation. It does not detect fraud, and no statistic can.
Read INTERPRETATION.md before using any output in public.

    from psephos import load, audit, to_text

    data = load("results.csv", columns=ColumnMap(registered="reg", votes={"A": "a", "B": "b"}))
    print(to_text(audit(data)))
"""

from psephos._types import AuditReport, Finding, Flag
from psephos.audit import audit
from psephos.report import to_json, to_text
from psephos.schema import (
    ColumnMap,
    ElectionData,
    SchemaError,
    autodetect,
    load,
    load_column_map,
    write_column_map,
)

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "ColumnMap",
    "ElectionData",
    "Finding",
    "Flag",
    "SchemaError",
    "__version__",
    "audit",
    "autodetect",
    "load",
    "load_column_map",
    "to_json",
    "to_text",
    "write_column_map",
]
