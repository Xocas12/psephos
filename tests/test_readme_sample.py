"""The README's sample output is a recorded run, so it is re-run.

The sample at the top of the README is the first output most readers see, and it went stale once
already: it kept printing `stat= p= effect=` with no interval and no adjusted p after the
renderer had moved the effect and its interval to the front of the line and added `adj_p`
(issue #31). Nothing checked it, so nothing noticed.

The sample is the output of the command printed above it, run on a synthetic election from
conftest.py: `clean_election()` with whole-number turnout injected into 15 per cent of precincts
by `with_rounded_turnout()`, its three parties renamed party_a, party_b and party_c. These tests
write that file, run that command through the CLI, and require every line the README shows to be
a line the run prints, in the same order. A line that is just `...` stands for lines left out,
and a line ending in ` ...` is cut short; everything else must match.

Two strengths of "match". The line format, the labels and the order must match on every
supported Python. The numbers must match exactly only on NumPy 2.5 or later, the NumPy the sample
was recorded on: NumPy 2.4 and 2.5 draw different binomial streams from the same seed for some
precinct sizes and rates, so the Monte Carlo figures in the sample move by a few units on the
older NumPy that Python 3.10 and 3.11 resolve to, while everything else stays put.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import numpy as np
import pytest
from conftest import clean_election, with_rounded_turnout

from psephos.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RECORDED_ON_NUMPY = (2, 5)


def _sample() -> tuple[list[str], list[str]]:
    """The README's sample command, as argv, and the output lines printed under it."""
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```console\n(.*?)\n```", text, flags=re.S)
    sample = [b for b in blocks if b.startswith("$ psephos audit ")]
    assert len(sample) == 1, "the README should carry exactly one sample audit run"
    lines = sample[0].split("\n")

    command = lines.pop(0)
    while command.endswith("\\"):
        command = command[:-1] + " " + lines.pop(0).strip()
    argv = shlex.split(command)
    assert argv[:2] == ["$", "psephos"], argv
    assert lines and lines[0] == "", "a blank line separates the command from its output"
    return argv[2:], lines[1:]


def _literal(text: str, *, any_number: bool) -> str:
    if not any_number:
        return re.escape(text)
    return r"\d+".join(re.escape(part) for part in re.split(r"\d+", text))


def _pattern(lines: list[str], *, any_number: bool = False) -> str:
    """A regular expression matching an output that the sample abridges. With `any_number`,
    every run of digits in the sample matches any run of digits in the output."""
    parts = []
    for line in lines:
        if line.strip() == "...":
            parts.append(r"(?:[^\n]*\n)*?")
        elif line.endswith(" ..."):
            parts.append(_literal(line[: -len(" ...")], any_number=any_number) + r"[^\n]*\n")
        else:
            parts.append(_literal(line, any_number=any_number) + r"\n")
    return "".join(parts)


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory) -> tuple[list[str], str]:
    """The README's sample lines, and what its command prints today."""
    argv, lines = _sample()
    assert argv[:2] == ["audit", "results.csv"], argv

    where = tmp_path_factory.mktemp("readme")
    frame = with_rounded_turnout(clean_election()).rename(
        columns={
            "party_incumbent": "party_a",
            "party_second": "party_b",
            "party_third": "party_c",
        }
    )
    frame.to_csv(where / "results.csv", index=False)

    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        mp.chdir(where)
        code = cli_main(argv)
    assert code == 0
    return lines, out.getvalue()


def _stale(output: str) -> str:
    return (
        "README.md's sample output is no longer what its command prints. Re-record it from "
        "this run, keeping the `...` cuts:\n\n" + "\n".join(output.split("\n")[:30])
    )


def test_the_readme_sample_shows_a_finding_in_full():
    """A sample abridged down to its header would match any output at all, so it has to keep
    the lines the issue was about: the flagged count and a finding's effect line."""
    _, lines = _sample()
    shown = [line for line in lines if line.strip() != "..." and not line.endswith(" ...")]
    assert any(line.startswith("FLAGGED FINDINGS (") for line in shown)
    assert any(line.lstrip().startswith("effect=") for line in shown), (
        "the README sample no longer shows a finding's effect line"
    )


def test_the_readme_sample_prints_the_effect_and_its_interval_before_the_p_value():
    """The README says every effect carries an interval, printed before its p-value. Its own
    sample is the first place a reader would check that, so the sample has to show it."""
    _, lines = _sample()
    effect_lines = [line.strip() for line in lines if line.lstrip().startswith("effect=")]
    assert effect_lines
    for line in effect_lines:
        assert re.match(r"effect=\S+ \[\S+, \S+\]  ", line), f"no interval on: {line}"
        assert "  p=" in line, f"no p-value on: {line}"
        assert "  adj_p=" in line, f"no adjusted p-value on: {line}"


def test_the_readme_sample_has_the_line_format_its_command_prints_today(sample_run):
    """Every line the sample shows is a line the run prints, in order, with any number allowed
    where the sample has a number. This is what catches a renderer change: a field added,
    dropped or moved, a section renamed, a check renamed."""
    lines, output = sample_run
    assert re.match(_pattern(lines, any_number=True), output + "\n"), _stale(output)


@pytest.mark.skipif(
    tuple(int(part) for part in np.__version__.split(".")[:2]) < RECORDED_ON_NUMPY,
    reason="the sample's Monte Carlo figures were recorded on NumPy 2.5, whose binomial "
    "stream differs from older NumPy's for the same seed",
)
def test_the_readme_sample_has_the_numbers_its_command_prints_today(sample_run):
    lines, output = sample_run
    assert re.match(_pattern(lines), output + "\n"), _stale(output)
