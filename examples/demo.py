"""Run psephos on two synthetic elections: one clean, one with injected rounding.

Everything here is fabricated. The point is to show what the two reports look like side by
side, and in particular that the clean one is not flagged. A tool that fires on both would be
useless, and you should not trust psephos on real data without first checking that it stays
quiet on data you generated yourself.

    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import clean_election, with_rounded_turnout

from psephos import ColumnMap, ElectionData, audit, to_text

COLUMNS = ColumnMap(
    votes={
        "party_incumbent": "party_incumbent",
        "party_second": "party_second",
        "party_third": "party_third",
    },
    registered="registered",
    ballots_cast="ballots_cast",
    precinct_id="precinct",
    region="region",
)


def run(name: str, frame: pd.DataFrame) -> None:
    data = ElectionData(frame=frame, columns=COLUMNS, source=name)
    report = audit(data, n_mc=300, seed=0)
    print(to_text(report))
    print(f"\n>>> {name}: {len(report.flagged)} flagged of {len(report.findings)} findings\n")
    print("=" * 78, "\n")


def main() -> int:
    clean = clean_election(n=4000, seed=11)

    print("A clean synthetic election. Nothing should be flagged.\n")
    run("synthetic-clean", clean)

    print(
        "The same election with 15 per cent of precincts moved onto a whole-number turnout.\n"
        "The integer-percentage family should find it, and the signal should survive the\n"
        "strictest size threshold.\n"
    )
    run("synthetic-rounded", with_rounded_turnout(clean, fraction=0.15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
