# Reading the files commissions actually publish

`pandas.read_csv` on a UTF-8 file with one header row and plain integers is the easy case, and
it is not the common one. Issue #14 lists what arrives instead. This document records what
`psephos.reading` does about each, and the principle that decides it.

## The principle

**Every repair is reported, and no row is removed.**

A loader that silently fixes a file is worse than one that fails, because the repair becomes an
invisible assumption underneath every number that follows and the reader has no way to find it.
So each step appends a sentence to a list, the list travels with the data as
`ElectionData.load_notes`, and the text report prints it under `HOW THE FILE WAS READ` before
any statistic:

```
HOW THE FILE WAS READ
  Each line is an assumption underneath every number below. A loader that repaired this
  file silently would have hidden them.
  - not valid UTF-8 (first bad byte at offset 0), so it was read as cp1251. This is a
    GUESS. The result contains Cyrillic text, which is consistent with the guess.
  - the table does not start at the first line: lines 0 to 1 were skipped as a header
    block. This was detected, not declared; pass --header-rows if it is wrong.
  - column 'Внесенных в список' was text and is now numeric, read as space thousands
    separator (7 values). It would otherwise have been dropped from every numeric check
    without a word. ...
  - 1 row(s) look like an aggregate rather than a precinct: row 6 ('ИТОГО'). They were
    NOT removed ...
```

## Encodings

UTF-8 first, strictly. A file that decodes as strict UTF-8 almost certainly is UTF-8, because
the multi-byte sequences are too constrained to arise by accident. After that: `utf-8-sig`,
`cp1251`, `cp1252`, `latin-1`.

The honesty matters more than the guess, and this is the part worth reading twice. **A
single-byte codec cannot fail.** `cp1252` and `latin-1` decode any byte sequence whatsoever, so
"it decoded" is not evidence that it decoded *correctly*. The note says exactly that rather than
implying the encoding was determined, and where the result does contain text in the script the
encoding exists for — Cyrillic, for `cp1251` — it says so as the weak evidence it is. `--encoding`
overrides, and is then reported as given rather than as detected.

## The table does not start at the first line

A published file opens with letterhead, a title and a date. This is also the case plain pandas
cannot read at all, rather than reading wrongly: it takes the field count from the first line,
which is a one-field title, so every data row is a tokenizing error.

So the field count is sniffed from the widest of the first fifty lines, the file is read with no
header assumed, and the header is located by scanning for the first row that is mostly
non-blank, mostly non-numeric, and followed by a row that is mostly numeric. `--header-rows`
overrides it.

## Multi-row headers and merged cells

A merged heading arrives as its value in the first column it spans and blanks after it. An
un-filled two-row header therefore gives several columns with the same name, or names with no
sign of which contest they belong to. The first header row is filled forwards and the rows are
joined with `/`:

| row 4 | `Station` | `Electorate` | `Votes` | *(merged)* |
| row 5 | | | `Party A` | `Party B` |

becomes `Station`, `Electorate`, `Votes / Party A`, `Votes / Party B`.

## Field separators

Sniffed from `,`, `;`, tab and `|`, scored on the most common field count each produces counting
only the lines that agree on it — a real separator splits most lines the same way, while a
character that merely appears in the prose does not. Consistency rather than frequency, because
the preamble lines contain no separator at all and a candidate should be judged on the lines
that look like a table.

Semicolon is not an afterthought. A continental European file writes decimals with a comma, so
it cannot also separate fields with one, and semicolon-delimited CSV is the normal published
form there.

## Thousands separators and decimal marks

The failure this prevents is **not a wrong number, it is a missing column**. A count column that
arrives as `1 234` stays text, which drops it out of the numeric auto-detection in
`schema.autodetect`, and so out of every check, without a word.

Five conventions are tried — `1,234.5`, `1.234,5`, `1 234,5`, `1 234.5`, `1'234.5` — plus the
non-breaking space, which is what a spreadsheet exports a thousands separator as and which is
invisible in every error message it ever appears in.

A reading is **rejected before it is tried**, not accepted because it produced a number:

- A thousands separator groups in threes, always. One followed by anything but exactly three
  digits is not a thousands separator. Checked on the integer part, since `1.234,5` is a
  perfectly good grouped number with a decimal tail.
- A decimal mark occurs at most once in a number.

This is what stops `66,7` being read as `667`. Trying comma-as-thousands first and accepting it
because the result parses is precisely the silent repair this issue is about: no error, no note,
and a turnout of 66.7 per cent recorded as 667.

A column is converted only when a style parses **every** non-blank value in it. A style that
parsed most of them would be choosing which rows to believe, and the rows it failed on would go
missing — the same bug one level down.

Where a column reads both ways — every separator groups three digits, so `1,234` is a thousand
or it is 1.234 — the thousands reading is taken, because that is right for a column of counts,
**and the note says the other reading exists** with an example that actually contains the
separator. Plain integers are converted without a note, since nothing was repaired and a note
per numeric column would bury the ones that mean something.

## A total row appended to the precinct table

The quietest large error in the tool. A national total passes every other integrity check — its
counts are positive, its turnout is plausible, its parties sum — and it is one unit the size of
the country, so every size-weighted statistic is dominated by it and the size strata put it
alone in the top band.

Two signatures, either sufficient: a text cell one of whose **whole words** is a totalling word
in any of fourteen languages, or a row whose numeric values are each within half a per cent of the sum of every
other row, which is what an **unlabelled** total looks like and is the case a word list cannot
catch.

The word match is whole-word and not a substring. Matching substrings was a real bug: Totalan is
a municipality in Malaga and Sumas is an ordinary place name, and both raised a STRONG "remove
it and rerun" finding on a file whose only sin was having a station there. A detector that fires
on an ordinary file is worse than none, and this one tells the user to delete a real precinct.

Detected rows are **not removed**. Dropping a row is a decision about the data rather than about
the file, and the loader does not get to make it. What makes the decision unavoidable instead is
`integrity.check_total_rows`, which flags it `STRONG` in the section the report prints before any
statistic. The check re-detects rather than trusting the loader, so constructing `ElectionData`
directly cannot skip it.

## What is not handled

- **Wide formats**, where each contestant is a row rather than a column. That is a reshape, not
  a repair, and guessing at it would be the loader deciding what the table means.
- **Several tables in one sheet**, separated by blank rows.
- **Footnote markers** attached to counts (`1 234 *`), which currently fail the all-or-nothing
  conversion and so leave the column text, reported but unusable.
- **`.xls`** older than the OOXML format, which needs `xlrd` rather than `openpyxl`.

Each would be a further issue. They are listed because a loader's limits should be written down
rather than discovered.

## Using it

```
psephos columns results.xlsx --sheet "Precincts"
psephos audit results.csv --encoding cp1251
psephos audit results.xlsx --header-rows 2 --registered "Electorate"
```

`psephos columns` uses the same reader the audit does and prints the same notes. It used to peek
with a plain `read_csv`, which meant that on exactly the files this document is about it reported
columns that were not in the table the audit then ran on.

Spreadsheets need an Excel engine: `pip install 'psephos[excel]'`.
