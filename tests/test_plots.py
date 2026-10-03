"""The plots and the HTML report.

The tests that matter most here are the count tests. A plot that draws a different sample from
the statistic it illustrates is a picture of something else, and nothing on screen would say so;
the only defence is to assert, for each plot, that the units it used and excluded are the units
the matching finding used and excluded. The fixture forces exclusions, because the clean
generator never makes a precinct small enough to be excluded and a count test on data with no
exclusions passes whatever the rule is.
"""

from __future__ import annotations

import sys
from html import escape

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from psephos import audit, to_html
from psephos.cli import main as cli_main
from psephos.plots import (
    Count,
    Plot,
    build_plots,
    last_digit_bars,
    threshold_sweep,
    turnout_histogram,
    turnout_share_histogram,
)
from psephos.schema import ColumnMap, ElectionData

N_MC = 40


@pytest.fixture
def messy_data(clean_df, column_map):
    """The clean election with exclusions forced into every check the plots illustrate."""
    df = clean_df.copy()
    df["registered"] = df["registered"].astype(float)
    df["ballots_cast"] = df["ballots_cast"].astype(float)
    df["party_third"] = df["party_third"].astype(float)
    df.loc[df.index[:25], "registered"] = np.nan  # no turnout at all
    df.loc[df.index[25:40], "registered"] = 60.0  # below every size threshold
    df.loc[df.index[25:40], "ballots_cast"] = 30.0
    df.loc[df.index[40:55], "party_third"] = 7.0  # below the last-digit minimum
    return ElectionData(frame=df, columns=column_map, source="synthetic-messy")


@pytest.fixture
def messy_report(messy_data):
    return audit(messy_data, n_mc=N_MC, seed=0)


def _counts(count: Count) -> tuple[int, int]:
    return count.n_used, count.n_excluded


def test_turnout_histogram_uses_the_units_the_integer_check_used(messy_data, messy_report):
    finding = next(
        f
        for f in messy_report.findings
        if f.check == "integer_percentage" and f.slice_name == "turnout, min_denominator=100"
    )
    plot = turnout_histogram(messy_data, min_denominator=100)
    assert finding.n_excluded > 0
    assert _counts(plot.counts[0]) == (finding.n_used, finding.n_excluded)


def test_turnout_share_histogram_uses_the_units_the_dependence_check_used(messy_data, messy_report):
    finding = next(f for f in messy_report.findings if f.check == "turnout_share_dependence")
    plot = turnout_share_histogram(messy_data, winner=messy_report.meta["winner"])
    assert finding.n_excluded > 0
    assert _counts(plot.counts[0]) == (finding.n_used, finding.n_excluded)


def test_last_digit_bars_use_the_units_each_digit_check_used(messy_data, messy_report):
    plot = last_digit_bars(messy_data)
    by_label = {c.label: c for c in plot.counts}
    findings = [f for f in messy_report.findings if f.check == "last_digit"]
    assert len(findings) == len(by_label) == 3
    for f in findings:
        assert _counts(by_label[f.slice_name]) == (f.n_used, f.n_excluded), f.slice_name
    assert by_label["party_third"].n_excluded > 0


def test_threshold_sweep_plots_every_sweep_finding_and_no_stratum(messy_report):
    plot = threshold_sweep(messy_report)
    sweep = [
        f
        for f in messy_report.findings
        if f.check == "integer_percentage" and "min_denominator=" in f.slice_name
    ]
    strata = [
        f
        for f in messy_report.findings
        if f.check == "integer_percentage" and "min_denominator=" not in f.slice_name
    ]
    assert strata, "the fixture should produce stratified re-tests to exclude"
    assert len(plot.counts) == len(sweep) == 8  # two series at four thresholds
    by_label = {c.label: c for c in plot.counts}
    for f in sweep:
        # "turnout, min_denominator=100" is plotted as "turnout at min_denominator=100".
        label = f.slice_name.replace(", min_denominator=", " at min_denominator=")
        assert _counts(by_label[label]) == (f.n_used, f.n_excluded), label
    assert any(c.n_excluded > 0 for c in plot.counts)


def test_every_caption_states_used_and_excluded(messy_data, messy_report):
    plots, missing = build_plots(messy_report, messy_data)
    assert [p.name for p in plots] == [
        "turnout_histogram",
        "turnout_share_histogram",
        "threshold_sweep",
        "last_digit_bars",
    ]
    assert missing == []
    for p in plots:
        for c in p.counts:
            assert f"n={c.n_used} used, {c.n_excluded} excluded" in p.caption, p.name
        # The caption is stamped onto the figure too, so an exported image keeps it.
        stamped = " ".join(t.get_text().replace("\n", " ") for t in p.figure.texts)
        assert f"n={p.counts[0].n_used} used" in stamped, p.name
        assert p.png().startswith(b"\x89PNG")


def test_a_plot_cannot_be_built_without_its_counts(messy_data):
    fig = turnout_histogram(messy_data).figure
    with pytest.raises(ValueError, match="sample size"):
        Plot(name="x", title="t", figure=fig, caption="no counts", counts=())
    count = Count("turnout", 10, 2, "reasons")
    with pytest.raises(ValueError, match="omits the counts"):
        Plot(name="x", title="t", figure=fig, caption="lost the sentence", counts=(count,))


def test_without_registered_voters_the_turnout_plots_are_named_as_absent(clean_df):
    cmap = ColumnMap(votes={c: c for c in ("party_incumbent", "party_second", "party_third")})
    data = ElectionData(frame=clean_df, columns=cmap, source="no-registered")
    report = audit(data, n_mc=N_MC, seed=0)
    plots, missing = build_plots(report, data)
    assert [p.name for p in plots] == ["last_digit_bars"]
    assert len(missing) == 1 and "registered-voters column" in missing[0]
    page = to_html(report, data)
    assert "registered-voters column" in page


# ---------------------------------------------------------------- html


def test_html_puts_the_interpretation_before_every_finding_and_figure(messy_data, messy_report):
    page = to_html(messy_report, messy_data)
    interp = page.index('id="how-to-read-this"')
    assert interp < page.index("Flagged findings")
    assert interp < page.index("<figure")
    assert interp < page.index("<h2>Integrity</h2>")
    assert "Anomalies are not evidence of fraud on their own" in page
    assert page.count("<figure") == 4
    assert page.count('src="data:image/png;base64,') == 4
    assert "http://" not in page and "https://" not in page  # nothing loaded from the network


def test_html_carries_every_caption_and_the_confounds_of_flagged_findings(clean_df, column_map):
    # Rounded turnout is flagged, so the page must print its innocent explanations.
    from conftest import with_rounded_turnout

    data = ElectionData(
        frame=with_rounded_turnout(clean_df, fraction=0.15), columns=column_map, source="r"
    )
    report = audit(data, n_mc=N_MC, seed=0)
    assert report.flagged
    page = to_html(report, data)
    for f in report.flagged:
        for c in f.confounds:
            assert escape(c) in page
    plots, _ = build_plots(report, data)
    for p in plots:
        assert f"n={p.counts[0].n_used} used" in page


def test_html_escapes_text_from_the_data(clean_df, column_map):
    data = ElectionData(frame=clean_df, columns=column_map, source="<script>alert(1)</script>")
    page = to_html(audit(data, n_mc=N_MC, seed=0), data)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_html_without_matplotlib_names_the_extra(monkeypatch, messy_data, messy_report):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    monkeypatch.delitem(sys.modules, "psephos.plots", raising=False)
    with pytest.raises(ImportError, match=r"psephos\[plots\]"):
        to_html(messy_report, messy_data)


# ---------------------------------------------------------------- cli


def test_cli_writes_the_html_report(tmp_path, clean_df, capsys):
    csv = tmp_path / "results.csv"
    clean_df.to_csv(csv, index=False)
    out = tmp_path / "report.html"
    code = cli_main(
        [
            "audit",
            str(csv),
            "--registered",
            "registered",
            "--ballots-cast",
            "ballots_cast",
            "--votes",
            "party_incumbent,party_second,party_third",
            "--mc",
            str(N_MC),
            "--html",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert page.index('id="how-to-read-this"') < page.index("<figure")
    assert f"wrote {out}" in capsys.readouterr().out
