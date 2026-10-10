"""Reading the files commissions publish, and saying what had to be assumed to read them.

Issue #14. Each test injects one awkwardness from `conftest_messy.py` and asserts two things:
that the table came out right, and that the repair was **reported**. The second is not a nicety.
A loader that silently fixes a file is worse than one that fails, because the repair becomes an
invisible assumption under every number that follows, and these tests are the only thing that
keeps the notes from quietly going missing in a refactor.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest_messy import (
    CLEAN_ROWS,
    NBSP,
    write_cp1251_csv,
    write_everything_csv,
    write_header_block_csv,
    write_merged_header_xlsx,
    write_separators_csv,
    write_total_row_csv,
)

from psephos.reading import (
    coerce_numeric,
    detect_encoding,
    find_header_rows,
    find_total_rows,
    read_table,
    sniff_separator,
)
from psephos.schema import ColumnMap, ElectionData, load

REGISTERED = [r[1] for r in CLEAN_ROWS]
CAST = [r[2] for r in CLEAN_ROWS]


def _note_text(notes: list[str]) -> str:
    return " || ".join(notes).casefold()


# --------------------------------------------------------------------------- encodings


def test_a_cp1251_file_is_read_and_the_guess_is_declared(tmp_path):
    """Cyrillic column names survive, and the note says the encoding was guessed."""
    data = load(write_cp1251_csv(tmp_path / "ru.csv"))

    assert data.n == len(CLEAN_ROWS)
    assert "Внесенных в список" in data.frame.columns
    assert list(data.frame["Внесенных в список"]) == [float(v) for v in REGISTERED]

    note = _note_text(data.load_notes)
    assert "cp1251" in note
    assert "guess" in note, "a detected encoding must not be reported as a determined one"
    assert "cyrillic" in note, "the evidence for the guess belongs in the note"


def test_a_utf8_file_says_it_decoded_cleanly(tmp_path):
    data = load(write_header_block_csv(tmp_path / "x.csv"))
    assert "utf-8" in _note_text(data.load_notes)
    assert "guess" not in _note_text(data.load_notes)


def test_an_explicit_encoding_is_reported_as_given_not_as_detected(tmp_path):
    data = load(write_cp1251_csv(tmp_path / "ru.csv"), encoding="cp1251")
    note = _note_text(data.load_notes)
    assert "as given" in note
    assert "guess" not in note


def test_a_single_byte_codec_cannot_fail_and_the_note_says_so(tmp_path):
    """The honest half of encoding detection.

    cp1252 and latin-1 decode any byte sequence whatsoever, so "it decoded" is no evidence that
    it decoded correctly. A note that implied otherwise would be the most misleading thing in
    the output, because it would look like a measurement.
    """
    path = tmp_path / "w.csv"
    # 0x93/0x94 are smart quotes in cp1252 and undefined-ish in cp1251's letter range; the point
    # is only that this is not valid UTF-8 and contains no Cyrillic.
    path.write_bytes(b"name,votes\n\x93Smith\x94,500\nJones,600\nBrown,700\nGreen,800\n")
    enc, note = detect_encoding(path)
    assert enc != "utf-8"
    assert "guess" in note.casefold()
    if enc in ("cp1252", "latin-1"):
        assert "no evidence" in note.casefold()
    else:
        assert "weak evidence against" in note.casefold()


def test_an_undecodable_file_falls_back_and_warns_about_every_character(tmp_path):
    path = tmp_path / "junk.csv"
    path.write_bytes(bytes([0x81, 0x8D, 0x8F, 0x90, 0x9D]) + b",1\nb,2\n")
    enc, note = detect_encoding(path)
    assert enc in ("cp1251", "cp1252", "latin-1")
    assert "guess" in note.casefold() or "unreliable" in note.casefold()


# --------------------------------------------------------------------------- header blocks


def test_a_title_block_above_the_table_is_skipped_and_reported(tmp_path):
    """The shape of nearly every published file, and the one pandas cannot parse at all.

    pandas takes the field count from the first line, which here is a one-field title, so
    without this the file is a tokenizing error rather than a table.
    """
    data = load(write_header_block_csv(tmp_path / "h.csv", lines=3))

    assert list(data.frame.columns) == ["precinct", "registered", "ballots_cast", "party_a"]
    assert data.n == len(CLEAN_ROWS)
    note = _note_text(data.load_notes)
    assert "does not start at the first line" in note
    assert "lines 0 to 2" in note
    assert "detected, not declared" in note


@pytest.mark.parametrize("lines", [0, 1, 2, 3])
def test_the_header_scan_works_at_any_preamble_depth(tmp_path, lines):
    data = load(write_header_block_csv(tmp_path / f"h{lines}.csv", lines=lines))
    assert list(data.frame.columns)[:2] == ["precinct", "registered"]
    assert data.n == len(CLEAN_ROWS)


def test_header_rows_can_be_overridden_and_then_nothing_is_guessed(tmp_path):
    data = load(write_header_block_csv(tmp_path / "h.csv"), header_rows=1)
    assert "as given" in _note_text(data.load_notes)


def test_find_header_rows_on_a_plain_table_finds_the_first_row():
    raw = pd.DataFrame([["precinct", "votes"], ["a", "1"], ["b", "2"], ["c", "3"]])
    assert find_header_rows(raw) == (0, 1)


def test_find_header_rows_survives_a_table_with_no_rows():
    assert find_header_rows(pd.DataFrame()) == (0, 1)


# --------------------------------------------------------------------------- spreadsheets


def test_a_two_row_header_with_merged_cells_becomes_distinct_column_names(tmp_path):
    """A merged heading over two vote columns.

    The merged cell arrives as its value in the first column and a blank after it, so an
    un-filled two-row header gives two columns called "Party A" and "Party B" with no sign of
    which contest they belong to -- or, worse, two columns called the same thing.
    """
    data = load(write_merged_header_xlsx(tmp_path / "m.xlsx"))

    assert list(data.frame.columns) == [
        "Station",
        "Electorate",
        "Votes / Party A",
        "Votes / Party B",
    ]
    assert data.n == len(CLEAN_ROWS)
    assert list(data.frame["Electorate"]) == [float(v) for v in REGISTERED]

    note = _note_text(data.load_notes)
    assert "merged cells" in note
    assert "filled forwards" in note
    assert "2 row(s) from 3 were read as the header" in note


def test_a_spreadsheet_column_of_counts_is_numeric_not_text(tmp_path):
    data = load(write_merged_header_xlsx(tmp_path / "m.xlsx"))
    assert pd.api.types.is_numeric_dtype(data.frame["Votes / Party A"])


# --------------------------------------------------------------------------- separators


def test_a_semicolon_delimited_file_is_detected(tmp_path):
    """A decimal comma forces a semicolon delimiter, which is how continental Europe publishes."""
    data = load(write_separators_csv(tmp_path / "s.csv"))
    assert list(data.frame.columns) == [
        "precinct",
        "registered",
        "ballots_cast",
        "party_a",
        "turnout_pct",
    ]
    assert "semicolon" in _note_text(data.load_notes)


def test_counts_with_a_thousands_separator_become_numbers_and_say_so(tmp_path):
    """The failure this prevents is not a wrong number, it is a missing column.

    A count column that stays text drops out of the numeric auto-detection in
    `schema.autodetect`, and so out of every check, without a word.
    """
    data = load(write_separators_csv(tmp_path / "s.csv"))

    assert list(data.frame["registered"]) == [float(v) for v in REGISTERED]
    assert pd.api.types.is_numeric_dtype(data.frame["registered"])
    note = _note_text(data.load_notes)
    assert "'registered' was text and is now numeric" in note
    assert "space thousands separator" in note
    assert "dropped from every numeric check" in note


def test_a_decimal_comma_is_not_read_as_a_thousands_break():
    """The bug the grouping check exists for.

    Trying comma-as-thousands first and accepting it because the result parses turns a turnout
    of 66.7 per cent into 667, with no error and no note. A thousands separator groups in
    threes; one followed by a single digit is a decimal mark.
    """
    frame = pd.DataFrame({"turnout": ["66,7", "57,1", "120,5"]})
    notes = coerce_numeric(frame)
    assert list(frame["turnout"]) == [66.7, 57.1, 120.5]
    assert "comma decimal mark" in _note_text(notes)
    assert "thousands" not in _note_text(notes).split("decimal")[0]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1 234", "5 678", "910"], [1234.0, 5678.0, 910.0]),
        (["1.234,5", "22.000,0"], [1234.5, 22000.0]),
        (["1,234.5", "22,000.0"], [1234.5, 22000.0]),
        (["12,34", "5,6"], [12.34, 5.6]),
        (["1 234 567", "89"], [1234567.0, 89.0]),
        (["450", "500", "390"], [450.0, 500.0, 390.0]),
        ([f"1{NBSP}234", f"5{NBSP}678"], [1234.0, 5678.0]),
    ],
)
def test_every_separator_convention_parses_to_the_right_number(values, expected):
    frame = pd.DataFrame({"x": values})
    coerce_numeric(frame)
    assert list(frame["x"]) == expected


def test_a_column_that_reads_both_ways_is_converted_and_the_ambiguity_is_stated():
    """`1,234` is a thousand or it is 1.234. Thousands is right for a count column; say so."""
    frame = pd.DataFrame({"votes": ["1,234", "5,678", "910"]})
    notes = coerce_numeric(frame)
    assert list(frame["votes"]) == [1234.0, 5678.0, 910.0]
    note = _note_text(notes)
    assert "could" in note and "instead be a decimal" in note
    assert "check it if this column is a share" in note
    # The example must contain the separator; most values in a column of counts do not.
    assert "'1,234'" in note


def test_plain_integers_are_converted_without_a_note():
    """A note per numeric column would bury the ones that mean something.

    These columns are text only because the file is read with `dtype=object` so the header scan
    can see the raw cells. Nothing is being repaired, so nothing is reported.
    """
    frame = pd.DataFrame({"votes": ["450", "500", "390"]})
    notes = coerce_numeric(frame)
    assert list(frame["votes"]) == [450.0, 500.0, 390.0]
    assert notes == []


def test_a_text_column_is_left_alone():
    frame = pd.DataFrame({"name": ["UIK 1", "UIK 2"], "region": ["North", "South"]})
    assert coerce_numeric(frame) == []
    assert not pd.api.types.is_numeric_dtype(frame["name"])
    assert list(frame["name"]) == ["UIK 1", "UIK 2"]


def test_a_column_that_only_mostly_parses_is_not_converted():
    """A style that parsed most of the values would be choosing which rows to believe, and the
    rows it failed on would go missing -- the same bug one level down."""
    frame = pd.DataFrame({"x": ["1 234", "5 678", "n/a"]})
    assert coerce_numeric(frame) == []
    assert not pd.api.types.is_numeric_dtype(frame["x"])
    assert list(frame["x"]) == ["1 234", "5 678", "n/a"], "and the values are untouched"


def test_sniff_separator_prefers_the_consistent_delimiter():
    lines = ["TITLE OF THE REPORT", "a;b;c", "1;2;3", "4;5;6"]
    sep, note = sniff_separator(lines)
    assert sep == ";"
    assert "semicolon" in note
    assert "3 of them on 3 of the 4 lines" in note


def test_sniff_separator_falls_back_to_a_comma_when_nothing_splits():
    sep, _ = sniff_separator(["one line only"])
    assert sep == ","


# --------------------------------------------------------------------------- total rows


def test_a_labelled_total_row_is_found_and_left_in_place(tmp_path):
    """Detected and reported, not removed. Dropping a row is a decision about the data."""
    data = load(write_total_row_csv(tmp_path / "t.csv", label="TOTAL"))

    assert data.n == len(CLEAN_ROWS) + 1, "the row is still there"
    assert data.suspected_total_rows == [len(CLEAN_ROWS)]
    note = _note_text(data.load_notes)
    assert "aggregate rather than a precinct" in note
    assert "not removed" in note
    assert "distorts every size-weighted statistic" in note


def test_an_unlabelled_total_row_is_found_by_the_column_sums(tmp_path):
    """The case a word list cannot catch, and the one that does the damage."""
    data = load(write_total_row_csv(tmp_path / "t.csv", label=None))
    assert data.suspected_total_rows == [len(CLEAN_ROWS)]


@pytest.mark.parametrize("word", ["TOTAL", "Итого", "всего", "Gesamt", "Razem", "suma"])
def test_total_words_are_matched_across_languages(word):
    frame = pd.DataFrame({"name": ["a", "b", "c", "d", word], "votes": [1.0, 2.0, 3.0, 4.0, 99.0]})
    assert 4 in find_total_rows(frame)


def test_a_clean_table_has_no_total_row(tmp_path):
    """The quiet half. A detector that fires on an ordinary file is worse than none."""
    data = load(write_header_block_csv(tmp_path / "h.csv"))
    assert data.suspected_total_rows == []
    assert "aggregate" not in _note_text(data.load_notes)


def test_a_table_too_small_to_have_a_sum_is_not_searched_for_one():
    frame = pd.DataFrame({"votes": [10.0, 10.0, 20.0]})
    assert find_total_rows(frame) == []


# --------------------------------------------------------------------------- everything at once


def test_one_file_with_all_of_it(tmp_path):
    """cp1251, a title block, non-breaking-space separators and a labelled total, together.

    They interact: the encoding has to be right before the header scan can see the header, the
    header has to be right before the columns can be coerced, and the coercion has to be right
    before the column sums can find the total row.
    """
    data = load(write_everything_csv(tmp_path / "all.csv"))

    assert data.n == len(CLEAN_ROWS) + 1
    assert list(data.frame.columns) == [
        "Участок",
        "Внесенных в список",
        "В стационарных ящиках",
        "Кандидат А",
    ]
    assert list(data.frame["Внесенных в список"])[: len(REGISTERED)] == [
        float(v) for v in REGISTERED
    ]
    assert data.suspected_total_rows == [len(CLEAN_ROWS)]

    note = _note_text(data.load_notes)
    for expected in (
        "cp1251",
        "guess",
        "does not start at the first line",
        "is now numeric",
        "aggregate rather than a precinct",
    ):
        assert expected in note, f"{expected!r} missing from the notes"


def test_the_autodetected_map_finds_the_cyrillic_roles(tmp_path):
    """The point of all of it: the columns have to reach `autodetect` as numbers."""
    data = load(write_everything_csv(tmp_path / "all.csv"))
    assert data.columns.registered == "Внесенных в список"
    assert data.columns.ballots_cast == "В стационарных ящиках"
    assert data.columns.votes, "the contestant column must be found as numeric"


# --------------------------------------------------------------------------- unchanged paths


def test_a_plain_csv_still_loads_exactly_as_before(tmp_path):
    path = tmp_path / "plain.csv"
    pd.DataFrame({"precinct": ["a", "b"], "registered": [900, 800]}).to_csv(path, index=False)
    data = load(path)
    assert list(data.frame.columns) == ["precinct", "registered"]
    assert list(data.frame["registered"]) == [900.0, 800.0]


def test_parquet_is_read_without_any_guessing(tmp_path):
    path = tmp_path / "x.parquet"
    pd.DataFrame({"precinct": ["a", "b"], "registered": [900, 800]}).to_parquet(path)
    result = read_table(path)
    assert "carries its own types and encoding" in _note_text(result.notes)
    assert "guess" not in _note_text(result.notes)


def test_a_table_built_in_memory_has_no_load_notes():
    frame = pd.DataFrame({"precinct": ["a", "b"], "registered": [900.0, 800.0]})
    data = ElectionData(frame=frame, columns=ColumnMap(registered="registered"))
    assert data.load_notes == []
    assert data.suspected_total_rows == []
