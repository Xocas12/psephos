"""The absent first-digit Benford test: the absence is a behaviour, so it is tested.

docs/why_no_benford.md carries the argument. These tests keep it true. The first-digit test
must keep rejecting honest data, or the document's premise has changed and the document has to
be rewritten; the last-digit check the repository does ship must keep coming back quiet on the
same data, because that contrast is the argument; and the CLI must answer a request for the
test by pointing at the document rather than refusing silently or running one.

The first-digit computation lives only here and in the document's runnable snippet. It is
deliberately absent from the package, so that no first-digit Benford test is importable from
psephos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import clean_election
from scipy import stats

from psephos._types import Flag
from psephos.cli import build_parser
from psephos.cli import main as cli_main
from psephos.methods.digits import last_digit_uniformity

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "why_no_benford.md"


def _first_digit_benford(counts: np.ndarray) -> tuple[float, float, float]:
    """Chi-square of first digits against Benford's law, with its p-value and TVD.

    Benford's law gives the first significant digit d probability log10(1 + 1/d). The test is
    the usual chi-square goodness of fit on 8 degrees of freedom, exactly as the document's
    snippet runs it.
    """
    x = np.rint(counts).astype(np.int64)
    x = x[x > 0]  # a count of zero has no first digit
    first = x // 10 ** np.floor(np.log10(x)).astype(np.int64)
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


def test_the_document_carries_the_recorded_output_of_a_fresh_run():
    """The demonstration's numbers are recorded output, not an illustration. Re-run the
    computation and require the document to contain exactly the summary line a fresh run
    produces; if the generator or the extraction changes, the document must be re-recorded."""
    assert DOC.is_file(), "docs/why_no_benford.md is missing"
    chi2, pvalue, tvd = _first_digit_benford(clean_election()["ballots_cast"].to_numpy(float))
    recorded = f"chi2={chi2:.1f} df=8 p={pvalue:.2e} tvd={tvd:.4f}"
    text = DOC.read_text(encoding="utf-8")
    assert recorded in text, f"the document no longer carries the real output line: {recorded}"


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
