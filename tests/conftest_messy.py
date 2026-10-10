"""Builders for the file shapes electoral commissions actually publish.

Each one injects exactly one awkwardness, named in the function, so a test that fails says which
shape broke rather than "the messy file". They write real files, because the point of issue #14
is the bytes on disk: an encoding, a header block, a merged cell and a thousands separator are
not things a DataFrame can be wrong about.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NBSP = "\u00a0"

#: A small, clean precinct table. Four columns, enough rows that a total row has something to be
#: the sum of, and sizes well above the thresholds so nothing is excluded for being small.
CLEAN_ROWS = [
    ("UIK 101", 1200, 800, 450),
    ("UIK 102", 1500, 900, 500),
    ("UIK 103", 980, 700, 390),
    ("UIK 104", 2100, 1400, 760),
    ("UIK 105", 1750, 1100, 620),
    ("UIK 106", 1320, 850, 470),
]

HEADERS = ("precinct", "registered", "ballots_cast", "party_a")


def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(CLEAN_ROWS, columns=list(HEADERS))


def write_cp1251_csv(path: Path) -> Path:
    """A CSV in Windows-1251 with Cyrillic column names, as Russian commissions publish."""
    rows = [
        "Участок,Внесенных в список,В стационарных ящиках,Кандидат А",
        *[f"УИК {i},{r[1]},{r[2]},{r[3]}" for i, r in enumerate(CLEAN_ROWS, start=101)],
    ]
    path.write_bytes(("\n".join(rows) + "\n").encode("cp1251"))
    return path


def write_header_block_csv(path: Path, lines: int = 3) -> Path:
    """A CSV with a title block above the table, which is the normal shape of a published file."""
    preamble = [
        "ELECTORAL COMMISSION OF SOMEWHERE",
        "Results by polling station",
        "Published 2026-03-02",
    ][:lines]
    rows = [
        *preamble,
        ",".join(HEADERS),
        *[",".join(str(x) for x in r) for r in CLEAN_ROWS],
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_separators_csv(
    path: Path, *, thousands: str = " ", decimal: str = ",", field_sep: str = ";"
) -> Path:
    """Counts written with a thousands separator, which is what makes a count column text.

    Semicolon-separated, because that is the only consistent way to publish decimal commas: a
    comma-separated file cannot also use commas for decimals without quoting every number, and
    continental European commissions use the semicolon. The turnout column carries the decimal
    mark, since a share beside the counts is what distinguishes `1.234` the thousand from
    `1.234` the number.
    """
    rows = [field_sep.join([*HEADERS, "turnout_pct"])]
    for name, reg, cast, party in CLEAN_ROWS:
        pct = f"{100 * cast / reg:.1f}".replace(".", decimal)
        counts = [f"{v:,}".replace(",", thousands) for v in (reg, cast, party)]
        rows.append(field_sep.join([name, *counts, pct]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_total_row_csv(path: Path, *, label: str | None = "TOTAL") -> Path:
    """The national total appended as if it were a precinct.

    ``label=None`` leaves it unlabelled, which is the case a word list cannot catch and the
    column-sum comparison has to.
    """
    reg = sum(r[1] for r in CLEAN_ROWS)
    cast = sum(r[2] for r in CLEAN_ROWS)
    party = sum(r[3] for r in CLEAN_ROWS)
    rows = [
        ",".join(HEADERS),
        *[",".join(str(x) for x in r) for r in CLEAN_ROWS],
        ",".join([label if label is not None else "UIK 999", str(reg), str(cast), str(party)]),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_merged_header_xlsx(path: Path) -> Path:
    """A spreadsheet with letterhead, a two-row header and merged cells over the vote columns.

    Written cell by cell rather than through pandas, because a merged cell is precisely what
    pandas would not produce: the heading sits in its first column and the cells after it are
    empty, which is what makes an un-filled two-row header give several identical names.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["ELECTORAL COMMISSION OF SOMEWHERE"])
    ws.append(["Results by polling station"])
    ws.append([])
    # Row 4: the merged top header. "Votes" spans the two contestant columns.
    ws.append(["Station", "Electorate", "Votes", None])
    # Row 5: the second header row, which is what distinguishes the two.
    ws.append([None, None, "Party A", "Party B"])
    for name, reg, cast, party in CLEAN_ROWS:
        ws.append([name, reg, party, cast - party])
    ws.merge_cells(start_row=4, start_column=3, end_row=4, end_column=4)
    wb.save(path)
    return path


def write_everything_csv(path: Path) -> Path:
    """One file with all of it: cp1251, a header block, NBSP separators and a labelled total."""
    rows = [
        "ИЗБИРАТЕЛЬНАЯ КОМИССИЯ",
        "Результаты по участкам",
        "Участок,Внесенных в список,В стационарных ящиках,Кандидат А",
    ]
    for i, (_, reg, cast, party) in enumerate(CLEAN_ROWS, start=101):
        counts = [f"{v:,}".replace(",", NBSP) for v in (reg, cast, party)]
        rows.append(",".join([f"УИК {i}", *counts]))
    totals = [sum(r[k] for r in CLEAN_ROWS) for k in (1, 2, 3)]
    rows.append(",".join(["ИТОГО", *[f"{v:,}".replace(",", NBSP) for v in totals]]))
    path.write_bytes(("\n".join(rows) + "\n").encode("cp1251"))
    return path
