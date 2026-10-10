"""Reading the files electoral commissions actually publish.

`pandas.read_csv` on a UTF-8 file with one header row and plain integers is the easy case, and
it is not the common one. Published precinct data arrives in Windows-1251, as a spreadsheet with
three lines of letterhead above the table, with a two-row header and merged cells, with counts
written `1 234` or `1.234`, and with a national total appended as if it were a precinct.

**Every repair this module makes is reported.** That is the whole design. A loader that silently
fixes a file is worse than one that fails, because the repair becomes an invisible assumption
underneath every number that follows, and the person reading the output has no way to find it.
So each step appends a sentence to a list of notes, the notes travel with the data into
`ElectionData.load_notes`, the audit puts them in its metadata and the text report prints them
in a section of their own before any statistic.

What is deliberately *not* done here is dropping rows. A suspected total row is reported, loudly,
and left in the table, because dropping it would be the loader guessing about the data. The
integrity check `total_row` is what makes it impossible to miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

#: Encodings tried in order when the file is not valid UTF-8. Windows-1251 comes first because
#: the largest openly published precinct datasets are Russian-language; 1252 and Latin-1 follow
#: for western European commissions.
FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1251", "cp1252", "latin-1")

#: Words that mark an aggregate row in the languages these datasets come in. Matched
#: case-insensitively against any text cell of a row.
TOTAL_ROW_WORDS: tuple[str, ...] = (
    "total",
    "totals",
    "итого",
    "всего",
    "сумма",
    "suma",
    "gesamt",
    "insgesamt",
    "totaal",
    "razem",
    "ogółem",
    "ogolem",
    "summa",
    "yhteensä",
)

#: Human names for the separator characters, because two of the five are whitespace and a note
#: reading "read as   thousands separator" tells a reader nothing at all.
SEPARATOR_NAMES = {
    ",": "comma",
    ".": "full stop",
    " ": "space",
    "\u00a0": "non-breaking space",
    "'": "apostrophe",
    ";": "semicolon",
    "\t": "tab",
    "|": "pipe",
    "": "no",
}

#: Field separators tried when reading a delimited text file, in order. Semicolon is not an
#: afterthought: a continental European file writes decimals with a comma, so it cannot also
#: separate fields with one, and semicolon-delimited CSV is the normal published form there.
FIELD_SEPARATORS: tuple[str, ...] = (",", ";", "\t", "|")

#: Separators tried when a column of digits did not parse as a number. Each pair is
#: (thousands, decimal).
_SEPARATOR_STYLES: tuple[tuple[str, str], ...] = (
    (",", "."),  # 1,234.5 — Anglophone
    (".", ","),  # 1.234,5 — most of continental Europe
    (" ", ","),  # 1 234,5 — Russian, French, Nordic
    (" ", "."),  # 1 234.5
    ("'", "."),  # 1'234.5 — Swiss
)


def _groups_like_thousands(values: pd.Series, thousands: str, decimal: str) -> bool:
    """Is every occurrence of ``thousands`` in every value a grouping break?

    A thousands separator groups in threes, always: `1 234 567`, never `1 23` or `1 2345`. So a
    candidate reading is rejected the moment one occurrence is followed by anything but exactly
    three digits.

    Checked on the integer part only, since `1.234,5` is a perfectly good thousands-grouped
    number with a decimal tail. Taking the part before the *last* decimal mark, because that is
    the one that can only be the decimal: an earlier occurrence of the same character would
    itself be a grouping break, and such a value is rejected anyway for having two decimal marks.

    This is what stops `66,7` being read as `667`. Trying comma-as-thousands first and accepting
    it because the result parses is exactly the silent repair issue #14 is about: no error, no
    note, and a turnout of 66.7 per cent recorded as 667.
    """
    integer_part = values.str.rsplit(decimal, n=1).str[0] if decimal else values
    pattern = r"^\d{1,3}(?:" + re.escape(thousands) + r"\d{3})+$"
    return bool(integer_part.str.fullmatch(pattern).all())


def _one_decimal_mark(values: pd.Series, sep: str) -> bool:
    """A decimal mark occurs at most once in a number."""
    return bool((values.str.count(re.escape(sep)) <= 1).all())


_DIGITS = re.compile(r"\d")

#: The non-breaking space a spreadsheet exports a thousands separator as. Invisible in every
#: error message it ever appears in, so it is named here and normalised once.
NBSP = "\u00a0"
#: Any letter, Latin or Cyrillic. A cell with one is a label, not a count.
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass
class ReadResult:
    """A table, and the record of what had to be done to read it."""

    frame: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    #: Row positions, in ``frame``, that look like aggregates rather than precincts.
    suspected_total_rows: list[int] = field(default_factory=list)


def detect_encoding(path: Path, sample_bytes: int = 1 << 20) -> tuple[str, str]:
    """Pick an encoding for a text file, and return it with a sentence about the choice.

    UTF-8 first, because a file that decodes as strict UTF-8 almost certainly is UTF-8: the
    multi-byte sequences are too constrained to arise by accident. After that the honesty
    matters more than the guess. **A single-byte codec cannot fail.** `cp1252` and `latin-1`
    decode any byte sequence whatsoever, so "it decoded" is not evidence that it decoded
    correctly, and the returned note says so rather than implying the encoding was determined.
    """
    raw = path.read_bytes()[:sample_bytes]
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        first_bad = exc.start
    else:
        return "utf-8", "read as UTF-8, which decoded cleanly"

    for enc in FALLBACK_ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        note = (
            f"not valid UTF-8 (first bad byte at offset {first_bad}), so it was read as {enc}. "
            "This is a GUESS."
        )
        if enc in ("cp1252", "latin-1"):
            note += (
                f" {enc} decodes any byte sequence, so the fact that it decoded is no evidence "
                "that it decoded correctly. Check any non-ASCII column name or region name "
                "below, and pass --encoding if they are wrong."
            )
        elif _looks_like_cyrillic(text):
            note += " The result contains Cyrillic text, which is consistent with the guess."
        else:
            note += (
                " The result contains no text in the script this encoding exists for, which is "
                "weak evidence against the guess. Pass --encoding if names look wrong."
            )
        return enc, note

    return "latin-1", (
        "no candidate encoding decoded the file, so it was read as latin-1, which cannot fail. "
        "Every non-ASCII character in the output is unreliable. Pass --encoding."
    )


def _looks_like_cyrillic(text: str) -> bool:
    cyrillic = sum(1 for ch in text[:50_000] if "Ѐ" <= ch <= "ӿ")
    return cyrillic > 20


def _is_blankish(value: Any) -> bool:
    return (
        value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""
    )


def _looks_numeric(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not pd.isna(value)
    text = str(value).strip()
    return bool(text) and bool(_DIGITS.search(text)) and not _LETTERS.search(text)


def find_header_rows(raw: pd.DataFrame, max_scan: int = 25) -> tuple[int, int]:
    """Locate the header in a sheet read with no header at all.

    Returns ``(first_header_row, n_header_rows)``. Commissions put letterhead, a title and a
    date above the table, so the first row is usually not the header. The header is taken to be
    the first row that is mostly non-blank, mostly non-numeric, and is followed by a row that is
    mostly numeric; a second header row is recognised when the row under it is still
    non-numeric and the data starts below that.
    """
    n_rows, n_cols = raw.shape
    if n_rows == 0 or n_cols == 0:
        return 0, 1

    def filled(i: int) -> float:
        return sum(not _is_blankish(v) for v in raw.iloc[i]) / n_cols

    def numericness(i: int) -> float:
        present = [v for v in raw.iloc[i] if not _is_blankish(v)]
        return (sum(_looks_numeric(v) for v in present) / len(present)) if present else 0.0

    for i in range(min(max_scan, n_rows - 1)):
        if filled(i) < 0.5 or numericness(i) > 0.4:
            continue
        # A second header row: still not numeric, and the row after it is.
        if (
            i + 2 < n_rows
            and numericness(i + 1) <= 0.4
            and filled(i + 1) >= 0.2
            and numericness(i + 2) >= 0.5
        ):
            return i, 2
        if numericness(i + 1) >= 0.5:
            return i, 1
    return 0, 1


def _join_header(raw: pd.DataFrame, first: int, n: int) -> list[str]:
    """Flatten one or two header rows into column names, filling merged cells forwards.

    A merged cell in a spreadsheet arrives as the value in its first column and blanks after it,
    so an un-filled two-row header gives several columns called the same thing. Filling forwards
    is what makes `"PartyA / votes"` and `"PartyA / share"` distinguishable.
    """
    rows = []
    for offset in range(n):
        row = ["" if _is_blankish(v) else str(v).strip() for v in raw.iloc[first + offset]]
        if offset == 0 and n > 1:
            carried = ""
            for j, cell in enumerate(row):
                if cell:
                    carried = cell
                else:
                    row[j] = carried
        rows.append(row)

    names = []
    for parts in zip(*rows, strict=True):
        seen = [p for i, p in enumerate(parts) if p and p not in parts[:i]]
        names.append(" / ".join(seen) if seen else "")
    # Blank and repeated names make a frame nobody can map a column of.
    out: list[str] = []
    for j, name in enumerate(names):
        candidate = name or f"column_{j + 1}"
        if candidate in out:
            candidate = f"{candidate}_{j + 1}"
        out.append(candidate)
    return out


def coerce_numeric(frame: pd.DataFrame) -> list[str]:
    """Parse text columns of digits into numbers, trying several separator conventions.

    A count column that arrives as ``"1 234"`` stays an object column, which drops it out of the
    numeric auto-detection in :func:`psephos.schema.autodetect` and so out of every check,
    silently. That is the failure mode this exists for: not a wrong number, a missing column.

    Mutates ``frame`` and returns one note per column converted. A column is converted only when
    a style parses **every** non-blank value in it. A style that parsed most of them would be
    choosing which rows to believe, and the rows it failed on would go missing without a word --
    which is the same bug one level down.
    """
    notes: list[str] = []
    for col in frame.columns:
        series = frame[col]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        # Non-breaking spaces are what a spreadsheet exports a thousands separator as, and they
        # are invisible in every error message, so they go first and once.
        text = series.astype("string").str.strip().str.replace(NBSP, " ", regex=False)
        filled = text[text.notna() & (text != "")]
        if filled.empty or not filled.str.contains(_DIGITS, regex=True).all():
            continue
        if filled.str.contains(_LETTERS, regex=True).any():
            continue

        # Plain digits first. These columns are only text because the file is read with
        # dtype=object, so that the header scan can see the raw cells; pandas would have
        # inferred them on its own and nothing is being repaired, so nothing is reported. A note
        # per numeric column would bury the ones that matter.
        plain = pd.to_numeric(text, errors="coerce")
        if plain[filled.index].notna().all():
            frame[col] = plain.astype(float)
            continue

        ambiguous = False
        for thousands, decimal in _SEPARATOR_STYLES:
            has_thousands = filled.str.contains(re.escape(thousands), regex=True).any()
            has_decimal = filled.str.contains(re.escape(decimal), regex=True).any()
            # Reject a reading before trying it, rather than accepting whichever one happens to
            # produce a number. Both of these would otherwise parse and both would be wrong.
            if has_thousands and not _groups_like_thousands(
                filled[filled.str.contains(re.escape(thousands), regex=True)],
                thousands,
                decimal,
            ):
                continue
            if has_decimal and not _one_decimal_mark(filled, decimal):
                continue
            # A column whose only separator groups in threes reads either way: 1,234 is a
            # thousand or it is 1.234. Thousands is the overwhelmingly likelier reading of a
            # count column, so take it and say that the other exists.
            if has_thousands and not has_decimal:
                ambiguous = True

            candidate = text
            if thousands:
                candidate = candidate.str.replace(thousands, "", regex=False)
            if decimal != ".":
                candidate = candidate.str.replace(decimal, ".", regex=False)
            parsed = pd.to_numeric(candidate, errors="coerce")
            # Every value that was there must have parsed; blanks stay missing.
            if parsed[filled.index].notna().all():
                frame[col] = parsed.astype(float)
                # Name only what the column actually contained. Reporting a convention for a
                # character that never appeared would be a note about the code's search order
                # rather than about the file.
                parts = []
                if has_thousands:
                    parts.append(
                        f"{SEPARATOR_NAMES.get(thousands, repr(thousands))} thousands separator"
                    )
                if has_decimal:
                    parts.append(f"{SEPARATOR_NAMES.get(decimal, repr(decimal))} decimal mark")
                style = " and ".join(parts) if parts else "plain digits"
                note = (
                    f"column {str(col)!r} was text and is now numeric, read as {style} "
                    f"({len(filled)} values). It would otherwise have been dropped from every "
                    "numeric check without a word."
                )
                if ambiguous and thousands:
                    sep_name = SEPARATOR_NAMES.get(thousands, repr(thousands))
                    # An example that actually contains the separator. Most values in a column
                    # of counts are under a thousand and have none, so the first value would
                    # usually illustrate nothing.
                    example = filled[filled.str.contains(re.escape(thousands), regex=True)]
                    shown = example.iloc[0]
                    note += (
                        f" Every {sep_name} here groups three digits, so {shown!r} could "
                        f"instead be a decimal: {parsed.loc[example.index[0]]:.0f} on the "
                        "thousands reading, a thousand times less on the other. The thousands "
                        "reading was taken, which is right for a count; check it if this column "
                        "is a share."
                    )
                notes.append(note)
                break
    return notes


def find_total_rows(frame: pd.DataFrame, numeric_cols: list[str] | None = None) -> list[int]:
    """Row positions that look like an aggregate rather than a precinct.

    Two signatures, either sufficient. A text cell containing a totalling word in any of the
    languages these files come in; or a row whose numeric values are each within half a per cent
    of the sum of every other row, which is what a total row looks like when it is unlabelled.

    This matters more than its size suggests. A national total left in as a precinct is one
    enormous unit: it passes every integrity check, and it distorts every size-weighted
    statistic in the tool while looking like a slightly unusual polling station.
    """
    if frame.empty:
        return []
    found: set[int] = set()

    for pos in range(len(frame)):
        for value in frame.iloc[pos]:
            if isinstance(value, str) and any(w in value.casefold() for w in TOTAL_ROW_WORDS):
                found.add(pos)
                break

    cols = numeric_cols or [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any()
    ]
    # Below three data rows there is nothing to be the sum of.
    if cols and len(frame) >= 4:
        block = frame[cols].to_numpy(dtype=float)
        column_totals = pd.DataFrame(block).sum(axis=0, skipna=True).to_numpy()
        for pos in range(len(frame)):
            row = block[pos]
            rest = column_totals - pd.Series(row).fillna(0.0).to_numpy()
            usable = (rest > 0) & pd.notna(row)
            if usable.sum() < max(1, len(cols) // 2):
                continue
            if bool(((abs(row[usable] - rest[usable]) / rest[usable]) < 0.005).all()):
                found.add(pos)
    return sorted(found)


def _sample_lines(path: Path, encoding: str, max_lines: int = 50) -> list[str]:
    with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
        out = []
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            stripped = line.rstrip("\r\n")
            if stripped.strip():
                out.append(stripped)
        return out


def sniff_separator(lines: list[str]) -> tuple[str, str]:
    """Which character separates the fields, and a sentence about the choice.

    Scored on the most common field count each candidate produces, counting only the lines that
    agree on it: a real separator splits most lines into the same number of fields, while a
    character that merely appears in the text splits them inconsistently. Ties go to the earlier
    candidate, which puts comma first.

    The preamble lines are the reason consistency is scored rather than the total count. A title
    line contains no separator at all, so a candidate is judged on the lines that look like a
    table and not on the ones above it.
    """
    best = (",", 0, 1)
    for sep in FIELD_SEPARATORS:
        counts: dict[int, int] = {}
        for line in lines:
            fields = line.count(sep) + 1
            if fields > 1:
                counts[fields] = counts.get(fields, 0) + 1
        if not counts:
            continue
        width, agreeing = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        if agreeing > best[1]:
            best = (sep, agreeing, width)
    sep, agreeing, width = best
    note = (
        f"fields separated by {SEPARATOR_NAMES.get(sep, repr(sep))}, "
        f"{width} of them on {agreeing} of the {len(lines)} lines sampled"
    )
    return sep, note


def _field_count(lines: list[str], sep: str) -> int:
    """The widest sampled line, in fields.

    The widest rather than the most common: a line wider than the real table adds empty columns,
    which the header scan names and which are harmless, where a line too narrow truncates the
    data and loses a contestant without saying so.
    """
    return max((line.count(sep) + 1 for line in lines), default=1)


def read_table(
    path: Path,
    *,
    encoding: str | None = None,
    sheet: str | int = 0,
    header_rows: int | None = None,
    **read_kwargs: Any,
) -> ReadResult:
    """Read a precinct table from CSV, TSV, parquet or Excel, recording every repair.

    ``encoding`` overrides the detection for text files. ``sheet`` picks a worksheet.
    ``header_rows`` overrides how many rows are taken as the header, for a file the scan gets
    wrong; passing it also skips the scan's report, since nothing was guessed.
    """
    suffix = path.suffix.lower()
    notes: list[str] = []

    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, **read_kwargs)
        notes.append("read as parquet, which carries its own types and encoding")
    elif suffix in {".xlsx", ".xlsm", ".xltx", ".xls", ".ods"}:
        frame, excel_notes = _read_spreadsheet(path, sheet, header_rows, read_kwargs)
        notes.extend(excel_notes)
    else:
        frame, text_notes = _read_text(path, suffix, encoding, header_rows, read_kwargs)
        notes.extend(text_notes)

    notes.extend(coerce_numeric(frame))

    totals = find_total_rows(frame)
    if totals:
        labels = []
        for pos in totals:
            text = [str(v) for v in frame.iloc[pos] if isinstance(v, str) and v.strip()]
            labels.append(f"row {pos} ({text[0]!r})" if text else f"row {pos}")
        notes.append(
            f"{len(totals)} row(s) look like an aggregate rather than a precinct: "
            f"{', '.join(labels)}. They were NOT removed, because dropping a row is a decision "
            "about the data rather than about the file. A total row left in is one enormous "
            "unit that distorts every size-weighted statistic, so remove it or confirm it is a "
            "precinct before trusting any number here."
        )

    return ReadResult(frame=frame, notes=notes, suspected_total_rows=totals)


def _read_spreadsheet(
    path: Path, sheet: str | int, header_rows: int | None, read_kwargs: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """A worksheet, with the letterhead above the table skipped and merged headers joined."""
    try:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object, **read_kwargs)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SpreadsheetError(
            f"reading {path.suffix} files needs an Excel engine, which is not installed. "
            "Install psephos with the excel extra: pip install 'psephos[excel]'"
        ) from exc

    notes = [f"read worksheet {sheet!r} of {path.suffix.lstrip('.')} with no header assumed"]
    first, n_header = find_header_rows(raw)
    if header_rows is not None:
        n_header = header_rows
        notes.append(f"header taken as {n_header} row(s) starting at row {first}, as given")
    elif first or n_header > 1:
        notes.append(
            f"the table does not start at the first row: rows 0 to {first - 1} were skipped as "
            f"a header block, and {n_header} row(s) from {first} were read as the header. "
            "This was detected, not declared; pass --header-rows if it is wrong."
            if first
            else f"{n_header} rows were read as a multi-row header, joined with ' / '."
        )
    if n_header > 1:
        notes.append(
            "merged cells in the first header row were filled forwards, so that columns under "
            "one merged heading get distinct names rather than repeating it."
        )

    names = _join_header(raw, first, n_header)
    frame = raw.iloc[first + n_header :].reset_index(drop=True)
    frame.columns = names
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame, notes


def _read_text(
    path: Path,
    suffix: str,
    encoding: str | None,
    header_rows: int | None,
    read_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """A CSV or TSV, with the encoding reported and a header block above the table skipped."""
    notes: list[str] = []
    if encoding is None:
        encoding, note = detect_encoding(path)
        notes.append(note)
    else:
        notes.append(f"read as {encoding}, as given")

    lines = _sample_lines(path, encoding)
    if "sep" in read_kwargs:
        sep = read_kwargs.pop("sep")
        notes.append(f"fields separated by {SEPARATOR_NAMES.get(sep, repr(sep))}, as given")
    elif suffix in {".tsv", ".tab"}:
        sep = "\t"
    else:
        sep, sep_note = sniff_separator(lines)
        if sep != ",":
            notes.append(sep_note)

    # pandas takes the field count from the first line, and the first line of a published file
    # is a title with no separators in it at all -- which makes every data row a parse error
    # rather than a table with a header block. So count the fields first and name that many
    # columns, and let the short preamble lines be padded out instead.
    width = _field_count(lines, sep)
    raw = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        header=None,
        names=range(width),
        dtype=object,
        skip_blank_lines=True,
        **read_kwargs,
    )
    first, n_header = find_header_rows(raw)
    if header_rows is not None:
        n_header = header_rows
        notes.append(f"header taken as {n_header} row(s) starting at row {first}, as given")
    elif first:
        notes.append(
            f"the table does not start at the first line: lines 0 to {first - 1} were skipped "
            f"as a header block. This was detected, not declared; pass --header-rows if it is "
            "wrong."
        )
    if n_header > 1:
        notes.append(f"{n_header} lines were read as a multi-row header, joined with ' / '.")

    names = _join_header(raw, first, n_header)
    frame = raw.iloc[first + n_header :].reset_index(drop=True)
    frame.columns = names
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame, notes


class SpreadsheetError(ValueError):
    """A spreadsheet could not be read, and the reason is actionable."""
