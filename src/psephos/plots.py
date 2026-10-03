"""The four pictures the text report argues from.

A z-score asks the reader to trust it. A picture lets the reader check it: the sawtooth in a
fine turnout histogram is visible or it is not, and a signal that dies as the size threshold
rises is a small-precinct artefact the moment it is drawn. These plots exist so that a reader can
look at the thing a statistic summarises.

Two rules, enforced by :class:`Plot` rather than by convention:

- Every plot carries the number of units it used and the number it excluded, in its caption and
  stamped onto the figure itself, so the image cannot be lifted out of the report without them.
- Every plot selects its units with the same rule as the check it illustrates. A histogram of a
  different sample from the one the statistic was computed on would be a picture of something
  else, and the tests assert the counts match the finding's.

Needs the ``plots`` extra (``pip install 'psephos[plots]'``); the core install stays on numpy,
scipy and pandas. Figures are built with :class:`matplotlib.figure.Figure` directly, never through
``pyplot``, so nothing here touches global state or needs a display.
"""

from __future__ import annotations

import inspect
import io
import textwrap
from dataclasses import dataclass

import numpy as np

try:
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - exercised by a test that hides matplotlib
    raise ImportError(
        "psephos plots need matplotlib, which the core install does not include. "
        "Install the extra: pip install 'psephos[plots]'"
    ) from exc

from psephos._types import AuditReport, Finding
from psephos.audit import DEFAULT_THRESHOLDS
from psephos.methods.digits import MIN_COUNT_FOR_LAST_DIGIT
from psephos.methods.integer_pct import integer_excess
from psephos.schema import ElectionData

#: How close to a whole number the integer check counts a percentage as landing on it. Read from
#: the check itself, so the histogram cannot drift from the statistic it illustrates.
WHOLE_NUMBER_TOLERANCE: float = inspect.signature(integer_excess).parameters["tolerance"].default

#: Width of one turnout bin, in percentage points: twice the tolerance, so the bin centred on a
#: whole number holds exactly the units the integer check counts as landing on it.
TURNOUT_BIN_WIDTH = 2 * WHOLE_NUMBER_TOLERANCE


@dataclass(frozen=True)
class Count:
    """How many units one series used and how many it excluded, and why."""

    label: str
    n_used: int
    n_excluded: int
    excluded_because: str

    def sentence(self) -> str:
        return (
            f"{self.label}: n={self.n_used} used, {self.n_excluded} excluded "
            f"({self.excluded_because})."
        )


@dataclass(frozen=True)
class Plot:
    """One figure with its caption. Cannot be built without its counts.

    ``caption`` is the description followed by one sentence per series stating what it used and
    what it excluded; :func:`_plot` builds it, and the check below refuses a caption that lost
    any of them.
    """

    name: str
    title: str
    figure: Figure
    caption: str
    counts: tuple[Count, ...]

    def __post_init__(self) -> None:
        if not self.counts:
            raise ValueError(f"{self.name}: a plot must state its sample size and exclusions")
        for c in self.counts:
            if c.sentence() not in self.caption:
                raise ValueError(f"{self.name}: the caption omits the counts for {c.label!r}")

    def png(self, dpi: int = 110) -> bytes:
        buf = io.BytesIO()
        self.figure.savefig(buf, format="png", dpi=dpi)
        return buf.getvalue()


def _plot(name: str, title: str, fig: Figure, description: str, counts: list[Count]) -> Plot:
    """Assemble a Plot and stamp its full caption onto the figure's lower margin."""
    caption = description + " " + " ".join(c.sentence() for c in counts)
    width, height = fig.get_size_inches()
    # About 18 characters of 7.5-point text per inch, so the caption uses the figure's width.
    lines = textwrap.wrap(caption, width=int(18 * width))
    margin = 0.13 * len(lines) + 0.2
    total = height + margin
    fig.set_size_inches(width, total)
    # Room below for the axis label above the caption, and above for the title over any
    # panel titles.
    fig.subplots_adjust(bottom=(margin + 0.55) / total, top=1 - 0.6 / total, hspace=0.6)
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7.5, va="bottom", ha="left", color="#444444")
    fig.suptitle(title, fontsize=11)
    return Plot(name=name, title=title, figure=fig, caption=caption, counts=tuple(counts))


# ---------------------------------------------------------------- 1. turnout histogram


def turnout_histogram(data: ElectionData, *, min_denominator: int | None = None) -> Plot:
    """Turnout at 0.1-point resolution with whole numbers marked.

    Units are selected exactly as :func:`psephos.methods.integer_pct.integer_excess` selects
    them for turnout: ballots counted and registered voters both present, at least
    ``min_denominator`` registered, and no more ballots than voters. The default threshold is the
    smallest one the audit sweeps.
    """
    t = min(DEFAULT_THRESHOLDS) if min_denominator is None else min_denominator
    num = data.total_votes()
    den = data.registered()
    usable = np.isfinite(num) & np.isfinite(den) & (den >= t) & (num >= 0) & (num <= den)
    pct = 100.0 * num[usable] / den[usable]

    fig = Figure(figsize=(9, 3.6))
    ax = fig.add_subplot()
    # Bins centred on multiples of the width, so one bin is centred on each whole number.
    n_bins = round(100.0 / TURNOUT_BIN_WIDTH) + 1
    edges = (np.arange(n_bins + 1) - 0.5) * TURNOUT_BIN_WIDTH
    heights, _ = np.histogram(pct, bins=edges)
    centres = np.arange(n_bins) * TURNOUT_BIN_WIDTH
    whole = np.isclose(centres, np.round(centres))
    ax.bar(centres[~whole], heights[~whole], width=TURNOUT_BIN_WIDTH, color="#3b6ea5", lw=0)
    ax.bar(centres[whole], heights[whole], width=TURNOUT_BIN_WIDTH, color="#c0392b", lw=0)
    if pct.size:
        lo, hi = np.quantile(pct, [0.005, 0.995])
        ax.set_xlim(max(0.0, lo - 2), min(100.0, hi + 2))
    ax.set_xlabel(
        f"turnout, per cent (bins of {TURNOUT_BIN_WIDTH:g} points; red: on a whole number)"
    )
    ax.set_ylabel("units")

    count = Count(
        "turnout",
        int(usable.sum()),
        int(usable.size - usable.sum()),
        f"fewer than {t} registered, missing, or more ballots than voters",
    )
    return _plot(
        "turnout_histogram",
        "Turnout at fine resolution",
        fig,
        f"Red bars hold the units within {WHOLE_NUMBER_TOLERANCE:g} points of a whole number, "
        "which is what the integer check counts. Rounding shows as red bars standing above "
        "their blue neighbours; red bars no taller than the blue is what binomial noise looks "
        "like.",
        [count],
    )


# ---------------------------------------------------------------- 2. turnout against share


def turnout_share_histogram(data: ElectionData, *, winner: str | None = None) -> Plot:
    """The vote-weighted two-dimensional histogram the dependence check summarises.

    Units are selected exactly as
    :func:`psephos.methods.turnout.turnout_share_dependence` selects them in the audit.
    """
    label = winner or data.winner_label()
    t = data.turnout()
    s = data.share(label, of="valid")
    w = data.votes_frame().sum(axis=1).to_numpy(dtype=float)
    usable = np.isfinite(t) & np.isfinite(s) & np.isfinite(w) & (t >= 0) & (t <= 100)

    fig = Figure(figsize=(6, 5.2))
    ax = fig.add_subplot()
    edges = np.arange(0.0, 101.0, 1.0)
    *_, image = ax.hist2d(
        t[usable], s[usable], bins=[edges, edges], weights=w[usable], cmap="viridis", cmin=1e-9
    )
    fig.colorbar(image, ax=ax, label="valid votes")
    ax.set_xlabel("turnout, per cent")
    ax.set_ylabel(f"{label} share of valid votes, per cent")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    count = Count(
        "units",
        int(usable.sum()),
        int(usable.size - usable.sum()),
        "turnout or share missing, or turnout outside 0 to 100",
    )
    return _plot(
        "turnout_share_histogram",
        f"Turnout against {label} share, weighted by votes",
        fig,
        "Added votes raise turnout and the winner's share together, which pulls a tail toward "
        "the top-right corner. Genuine turnout and partisanship are correlated in most "
        "countries, so a tilt is common and a corner tail is the shape to look for.",
        [count],
    )


# ---------------------------------------------------------------- 3. threshold sweep


def _sweep_findings(report: AuditReport) -> dict[str, list[Finding]]:
    """The integer-percentage findings of the size-threshold sweep, by series label.

    The stratified re-tests share the turnout hypothesis but are not points on the sweep; they
    run at a minimum of one within a size band, so they are recognised by the slice name.
    """
    series: dict[str, list[Finding]] = {}
    for f in report.findings:
        if f.check != "integer_percentage" or "min_denominator=" not in f.slice_name:
            continue
        if "min_denominator" not in f.details:
            continue
        series.setdefault(f.details.get("label", f.slice_name), []).append(f)
    for findings in series.values():
        findings.sort(key=lambda f: f.details["min_denominator"])
    return series


def threshold_sweep(report: AuditReport) -> Plot | None:
    """Excess whole-number percentages against the size threshold, from the audit's findings.

    Reads the findings rather than recomputing, so the points are the statistics the text report
    prints. Returns ``None`` when the audit ran no sweep, which happens without a
    registered-voters column.
    """
    series = _sweep_findings(report)
    if not series:
        return None

    fig = Figure(figsize=(7.5, 4.2))
    ax = fig.add_subplot()
    counts: list[Count] = []
    skipped: list[str] = []
    for label, findings in series.items():
        xs = [f.details["min_denominator"] for f in findings if f.effect is not None]
        ys = [100.0 * f.effect for f in findings if f.effect is not None]
        ax.plot(xs, ys, marker="o", label=label)
        for f in findings:
            t = f.details["min_denominator"]
            if f.effect is None:
                skipped.append(f"{label} at {t}")
            counts.append(
                Count(
                    f"{label} at min_denominator={t}",
                    f.n_used,
                    f.n_excluded,
                    f"denominator below {t}, missing, or out of range",
                )
            )
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("minimum denominator (log scale)")
    ax.set_ylabel("excess whole-number units, per cent of those tested")
    ax.legend(fontsize=8)

    description = (
        "A signal that shrinks toward zero as the threshold rises lives in small precincts, "
        "where whole-number percentages are arithmetic; one that holds at the strictest "
        "threshold does not."
    )
    if skipped:
        description += f" Not drawn, too few units to test: {', '.join(skipped)}."
    return _plot("threshold_sweep", "Effect against the size threshold", fig, description, counts)


# ---------------------------------------------------------------- 4. last digits


def last_digit_bars(data: ElectionData, *, min_count: int = MIN_COUNT_FOR_LAST_DIGIT) -> Plot:
    """Last-digit frequencies for each contestant against the uniform 10 per cent.

    Units are selected exactly as :func:`psephos.methods.digits.last_digit_uniformity` selects
    them. The band is two standard errors of a uniform proportion at that sample size, so a bar
    outside it is a departure noise would rarely produce in one digit, and with ten digits per
    panel one bar outside it is unremarkable.
    """
    labels = list(data.columns.votes)
    cols = min(3, len(labels))
    rows = int(np.ceil(len(labels) / cols))
    fig = Figure(figsize=(3.4 * cols, 2.7 * rows))
    counts: list[Count] = []
    for i, label in enumerate(labels):
        ax = fig.add_subplot(rows, cols, i + 1)
        x = data.frame[data.columns.votes[label]].to_numpy(dtype=float)
        usable = np.isfinite(x) & (x >= min_count)
        n = int(usable.sum())
        digits = np.rint(x[usable]).astype(np.int64) % 10
        props = np.bincount(digits, minlength=10) / n if n else np.zeros(10)
        ax.bar(range(10), 100.0 * props, color="#3b6ea5")
        if n:
            se = 100.0 * np.sqrt(0.1 * 0.9 / n)
            ax.axhspan(10.0 - 2 * se, 10.0 + 2 * se, color="#999999", alpha=0.25, linewidth=0)
        ax.axhline(10.0, color="#c0392b", linewidth=0.9)
        ax.set_xticks(range(10))
        ax.set_title(f"{label} (n={n})", fontsize=9)
        ax.set_ylabel("per cent", fontsize=8)
        counts.append(Count(label, n, int(usable.size - n), f"count below {min_count} or missing"))
    return _plot(
        "last_digit_bars",
        "Last digits against uniform",
        fig,
        "The red line is the uniform 10 per cent and the grey band two standard errors around "
        "it. Invented numbers tend to be uneven; small or rounded counts are too.",
        counts,
    )


# ---------------------------------------------------------------- all of them


def build_plots(report: AuditReport, data: ElectionData) -> tuple[list[Plot], list[str]]:
    """Every plot the data supports, and a sentence for each one it does not.

    The second list says why a plot is absent, so a report never silently has fewer pictures.
    """
    plots: list[Plot] = []
    missing: list[str] = []
    if not data.columns.votes:
        return plots, ["No contestant columns are mapped, so nothing can be drawn."]

    if data.columns.registered:
        thresholds = report.meta.get("settings", {}).get("thresholds") or DEFAULT_THRESHOLDS
        plots.append(turnout_histogram(data, min_denominator=min(thresholds)))
        plots.append(turnout_share_histogram(data, winner=report.meta.get("winner")))
        sweep = threshold_sweep(report)
        if sweep is not None:
            plots.append(sweep)
        else:
            missing.append("The threshold sweep is absent because the audit ran no sweep.")
    else:
        missing.append(
            "The turnout histogram, the turnout-share histogram and the threshold sweep are "
            "absent because no registered-voters column is mapped."
        )
    plots.append(last_digit_bars(data))
    return plots, missing
