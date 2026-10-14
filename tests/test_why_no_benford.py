"""The absent first-digit Benford test: the absence is a behaviour, so it is tested.

docs/why_no_benford.md carries the argument. These tests keep it true. The first-digit test
must keep rejecting honest data, or the document's premise has changed and the document has to
be rewritten; the last-digit check the repository does ship must keep coming back quiet on the
same data, because that contrast is the argument; and the CLI must answer a request for the
test by pointing at the document rather than refusing silently or running one.

The document's numbers are recorded output, and they are checked as such: every snippet in it
is re-run and must print exactly the block recorded under it, and every figure the prose quotes
from those runs must be the figure a fresh run gives. Issue #31 is why: the generator changed,
the one line that was checked was re-recorded, and the table above it, the prose around it and
the last-digit result kept describing the old generator.

The first-digit computation lives only here and in the document's runnable snippet. It is
deliberately absent from the package, so that no first-digit Benford test is importable from
psephos.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import clean_election
from scipy import stats

from psephos._types import Finding, Flag
from psephos.cli import build_parser
from psephos.cli import main as cli_main
from psephos.methods.digits import last_digit_uniformity
from psephos.report import _fmt_finding

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "why_no_benford.md"

# A runnable snippet in the document, and the output recorded straight after it.
RECORDED_RUN = re.compile(
    r"```console\n\$ uv run python - <<'PY'\n(?P<code>.*?)\nPY\n```\n\n"
    r"```text\n(?P<output>.*?)\n```",
    flags=re.S,
)


def _recorded_runs() -> list[tuple[str, str]]:
    text = DOC.read_text(encoding="utf-8")
    return [(m["code"], m["output"]) for m in RECORDED_RUN.finditer(text)]


def _first_digits(counts: np.ndarray) -> np.ndarray:
    x = np.rint(counts).astype(np.int64)
    x = x[x > 0]  # a count of zero has no first digit
    return x // 10 ** np.floor(np.log10(x)).astype(np.int64)


def _first_digit_benford(counts: np.ndarray) -> tuple[float, float, float]:
    """Chi-square of first digits against Benford's law, with its p-value and TVD.

    Benford's law gives the first significant digit d probability log10(1 + 1/d). The test is
    the usual chi-square goodness of fit on 8 degrees of freedom, exactly as the document's
    snippet runs it.
    """
    first = _first_digits(counts)
    observed = np.bincount(first, minlength=10)[1:10].astype(float)
    benford = np.log10(1.0 + 1.0 / np.arange(1, 10))
    expected = benford * observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(chi2, df=8))
    tvd = float(0.5 * np.abs(observed / observed.sum() - benford).sum())
    return chi2, pvalue, tvd


def test_first_digit_benford_rejects_the_clean_generator():
    """The document's premise: the first-digit test rejects honest data hard enough that the
    finding would be STRONG by the conventions every shipped check follows."""
    chi2, pvalue, tvd = _first_digit_benford(clean_election()["ballots_cast"].to_numpy(float))
    assert pvalue <= 0.001
    assert tvd >= 0.02
    assert chi2 > 100.0


@pytest.mark.parametrize("seed", range(20))
def test_first_digit_benford_rejects_honest_elections_across_seeds(seed):
    """The document claims the rejection is routine, not a one-off: every independent honest
    election from the generator must reject. The document's own hundred-election run found
    100 of 100; twenty seeds keep that claim checked on every CI run."""
    counts = clean_election(seed=seed)["ballots_cast"].to_numpy(float)
    _, pvalue, _ = _first_digit_benford(counts)
    assert pvalue <= 0.05


def test_the_shipped_last_digit_check_stays_quiet_on_the_same_counts():
    """The contrast the document turns on: same counts, same family, opposite verdict."""
    counts = clean_election()["ballots_cast"].to_numpy(float)
    finding = last_digit_uniformity(counts)
    assert finding.flag is Flag.OK
    assert finding.pvalue > 0.05


def test_the_diagnosis_is_the_span_not_the_arithmetic():
    """Widen the precinct sizes to several orders of magnitude, leave the counts exactly as
    honest, and the same test stops rejecting. This is the control that pins the failure on
    the scale of the data rather than on the test machinery."""
    rng = np.random.default_rng(11)
    size = np.clip(rng.lognormal(mean=7.5, sigma=1.8, size=4000), 10, 10_000_000)
    turnout = np.clip(rng.normal(0.55, 0.10, size=4000), 0.05, 0.98)
    counts = rng.binomial(np.rint(size).astype(int), turnout).astype(float)
    _, pvalue, _ = _first_digit_benford(counts)
    assert pvalue > 0.05


def test_every_snippet_in_the_document_has_its_output_recorded_under_it():
    """Guards the next test: a snippet the pattern misses would be a block nothing checks."""
    assert DOC.is_file(), "docs/why_no_benford.md is missing"
    text = DOC.read_text(encoding="utf-8")
    assert text.count("<<'PY'") == len(_recorded_runs()) == 4


@pytest.mark.parametrize("index", range(4), ids=lambda i: f"run{i + 1}")
def test_each_recorded_output_block_is_what_its_snippet_prints_today(index):
    """The demonstration's numbers are recorded output, not an illustration. Run each snippet
    the way the document says to, from the repository root, and require the block recorded
    under it to be exactly what it prints, every line of it. If the generator, the extraction or
    a shipped check changes, the document must be re-recorded from one run."""
    code, recorded = _recorded_runs()[index]
    result = subprocess.run(
        [sys.executable, "-"],
        input=code,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    printed = result.stdout.rstrip("\n")
    assert printed == recorded, (
        f"docs/why_no_benford.md records an output its snippet no longer prints.\n"
        f"--- recorded\n{recorded}\n--- printed today\n{printed}"
    )


def test_the_prose_quotes_the_figures_of_the_same_run():
    """The prose restates figures from the recorded runs, and a restatement can go stale while
    the block it came from is re-recorded. Every figure the prose quotes is rebuilt here from a
    fresh run, including the line a psephos report would print for the finding, which is built
    by the report's own renderer so that a change to the line format shows up here too."""
    counts = clean_election()["ballots_cast"].to_numpy(float)
    counts = counts[counts > 0]
    chi2, pvalue, tvd = _first_digit_benford(counts)
    q1, q3 = np.percentile(counts, [25, 75])
    share_of_ones = float(np.mean(_first_digits(counts) == 1))
    finding = Finding(
        check="first_digit_benford",
        title="First digits against Benford's law.",
        flag=Flag.STRONG,
        statistic=chi2,
        pvalue=pvalue,
        effect=tvd,
        n_used=len(counts),
        confounds=["Precinct sizes that span too little to give the first digit any freedom."],
    )
    report_line = next(
        line.strip() for line in _fmt_finding(finding) if line.strip().startswith("effect=")
    )
    first_law = np.log10(1.0 + 1.0 / np.arange(1, 10))
    second_law = np.array(
        [sum(np.log10(1.0 + 1.0 / (10 * k + d)) for k in range(1, 10)) for d in range(10)]
    )

    expected = [
        f"the counts run from {int(counts.min())} to {int(counts.max())} ballots and the middle "
        f"half of them lies within a factor of {q3 / q1:.1f}.",
        f"it comes back at p = {pvalue:.2e}.",
        f"Benford predicts {100 * first_law[0]:.1f} per cent of counts to start with 1; "
        f"{100 * share_of_ones:.1f} per cent of these honest counts do. Chi-square {chi2:.1f} on "
        f"8 degrees of freedom, p = {pvalue:.2e}, total variation distance from Benford "
        f"{tvd:.3f}.",
        f"`{report_line}`",
        f"{100 * second_law[0]:.1f} per cent for digit 0 down to {100 * second_law[9]:.1f} per "
        f"cent for digit 9 where the first-digit law runs from {100 * first_law[0]:.1f} to "
        f"{100 * first_law[8]:.1f}",
    ]
    # Prose wraps wherever it likes, so compare with every run of whitespace made one space.
    prose = " ".join(DOC.read_text(encoding="utf-8").split())
    missing = [figure for figure in expected if " ".join(figure.split()) not in prose]
    assert not missing, f"the document's prose no longer quotes a fresh run: {missing}"


def test_the_document_is_linked_from_the_readme():
    assert DOC.is_file(), "docs/why_no_benford.md is missing"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/why_no_benford.md" in readme, "the README no longer links the document"


def test_benford_is_listed_in_the_cli_help():
    assert "benford" in build_parser().format_help()


def test_cli_benford_points_at_the_document(capsys):
    """Asked for a Benford test, the CLI answers with the document instead of refusing. A
    nonexistent path must still exit 0: the command reads nothing."""
    code = cli_main(["benford", "no_such_file.csv"])
    out = capsys.readouterr().out
    assert code == 0, "pointing at the document is an answer, not a failure"
    assert "You asked for a first-digit Benford test" in out
    assert "docs/why_no_benford.md" in out


def test_cli_benford_answers_without_a_path(capsys):
    code = cli_main(["benford"])
    out = capsys.readouterr().out
    assert code == 0
    assert "docs/why_no_benford.md" in out
