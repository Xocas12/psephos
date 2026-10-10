"""Rendering an audit as one self-contained HTML file.

The interpretation section comes first, before any number or picture. In the text report it
closes the output; here a reader scrolls, and a reader who stops at the first striking plot must
already have passed the statement of what a plot cannot show. That ordering is the reason this
module exists in its present shape, so it is fixed: there is no option to move or drop the
section, and the tests check that it precedes every finding and every figure.

The file embeds its images as data URIs and loads nothing from the network, so it can be mailed,
archived and opened offline as the record of one audit.

Importing this module needs only the core dependencies; :func:`to_html` imports the plots, and
with them matplotlib, when it is called.
"""

from __future__ import annotations

import base64
import re
from html import escape

from psephos._types import AuditReport, Finding, Flag
from psephos.report import INTERPRETATION, _multiple_testing_lines
from psephos.schema import ElectionData

_STYLE = """
:root { --fg:#1d1d1f; --muted:#5a5a60; --bg:#ffffff; --panel:#f5f5f7; --rule:#d9d9de;
        --strong:#9b1c1c; --notable:#8a5a00; --ok:#2f6b3a; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#ececf0; --muted:#a2a2aa; --bg:#16161a; --panel:#202026; --rule:#34343c;
          --strong:#f08a8a; --notable:#e8b860; --ok:#8fd19e; }
}
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:920px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:18px; margin:36px 0 10px; padding-top:12px; border-top:1px solid var(--rule); }
.meta { color:var(--muted); margin:0 0 20px; }
.interpretation { background:var(--panel); border-left:4px solid var(--strong);
                  padding:4px 18px 10px; border-radius:6px; }
.interpretation h2 { border-top:none; margin-top:12px; }
.finding { border:1px solid var(--rule); border-radius:6px; padding:10px 14px; margin:10px 0; }
.flag { font-weight:600; text-transform:uppercase; font-size:12px; letter-spacing:.04em; }
.flag-strong { color:var(--strong); } .flag-notable { color:var(--notable); }
.flag-ok { color:var(--ok); } .flag-underpowered, .flag-not_applicable { color:var(--muted); }
.stats { font-family:ui-monospace, Consolas, monospace; font-size:13px; color:var(--muted); }
figure { margin:18px 0; } figure img { max-width:100%; height:auto; background:#fff;
                                       border-radius:4px; }
figcaption { font-size:13px; color:var(--muted); margin-top:6px; }
ul { padding-left:20px; }
"""


def _paragraphs(text: str) -> str:
    """Turn the plain-text interpretation into paragraphs and one ordered list.

    Blocks are separated by blank lines; a block that starts with "1." to "9." is a list item,
    and consecutive items share one list.
    """
    out: list[str] = []
    in_list = False
    for block in re.split(r"\n\s*\n", text.strip()):
        flat = " ".join(line.strip() for line in block.splitlines())
        item = re.match(r"^\d+\.\s+(.*)$", flat)
        if item:
            if not in_list:
                out.append("<ol>")
                in_list = True
            out.append(f"<li>{escape(item.group(1))}</li>")
            continue
        if in_list:
            out.append("</ol>")
            in_list = False
        out.append(f"<p>{escape(flat)}</p>")
    if in_list:
        out.append("</ol>")
    return "\n".join(out)


def _interpretation_html() -> str:
    body = INTERPRETATION.strip()
    heading, _, rest = body.partition("\n")
    return (
        '<section class="interpretation" id="how-to-read-this">'
        f"<h2>{escape(heading.strip().capitalize())}</h2>{_paragraphs(rest)}</section>"
    )


def _finding_html(f: Finding) -> str:
    bits = []
    # Effect first, then the interval, then the p-value: same order as the text report, and
    # for the same reason.
    if f.effect is not None:
        if f.ci_low is not None and f.ci_high is not None:
            bits.append(f"effect={f.effect:.4g} [{f.ci_low:.4g}, {f.ci_high:.4g}]")
        else:
            bits.append(f"effect={f.effect:.4g}")
    if f.statistic is not None:
        bits.append(f"stat={f.statistic:.4g}")
    if f.pvalue is not None:
        bits.append(f"p={f.pvalue:.4g}")
    if f.adjusted_pvalue is not None:
        bits.append(f"adj_p={f.adjusted_pvalue:.4g}")
    bits.append(f"n={f.n_used}")
    if f.n_excluded:
        bits.append(f"excluded={f.n_excluded}")
    parts = [
        '<div class="finding">',
        f'<div><span class="flag flag-{f.flag.value}">{escape(f.flag.value)}</span> '
        f"<strong>{escape(f.check)}</strong> ({escape(f.slice_name)})</div>",
        f"<div>{escape(f.title)}</div>",
        f'<div class="stats">{escape("  ".join(bits))}</div>',
    ]
    if f.confounds:
        parts.append("<div>Could also be:</div><ul>")
        parts.extend(f"<li>{escape(c)}</li>" for c in f.confounds)
        parts.append("</ul>")
    parts.append("</div>")
    return "\n".join(parts)


def to_html(report: AuditReport, data: ElectionData, *, verbose: bool = False) -> str:
    """One self-contained HTML page: interpretation, then findings, then the four plots.

    ``data`` must be the table the report was computed from; the plots are drawn from it and
    their unit counts are the ones the matching findings report.

    Raises ``ImportError`` naming the ``plots`` extra when matplotlib is not installed.
    """
    from psephos.plots import build_plots

    plots, missing = build_plots(report, data)

    meta = [f"{report.n_units} units"]
    if report.meta.get("winner"):
        meta.append(f"leading contestant: {report.meta['winner']}")

    html: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>psephos audit: {escape(report.source)}</title>",
        f"<style>{_STYLE}</style></head><body><main>",
        f"<h1>psephos audit: {escape(report.source)}</h1>",
        f'<p class="meta">{escape(" · ".join(meta))}</p>',
        _interpretation_html(),
    ]

    flagged = report.flagged
    html.append(f"<h2>Flagged findings ({len(flagged)} of {len(report.findings)})</h2>")
    if not flagged:
        html.append("<p>Nothing departed from its null enough to flag.</p>")
    html.extend(_finding_html(f) for f in flagged)

    mt = _multiple_testing_lines(report)
    if mt:
        html.append("<h2>Multiple testing</h2>")
        html.extend(f"<p>{escape(line.strip())}</p>" for line in mt[2:])

    html.append("<h2>Pictures</h2>")
    for p in plots:
        data_uri = base64.b64encode(p.png()).decode("ascii")
        html.append(
            f'<figure id="{escape(p.name)}"><img alt="{escape(p.title)}" '
            f'src="data:image/png;base64,{data_uri}">'
            f"<figcaption><strong>{escape(p.title)}.</strong> {escape(p.caption)}</figcaption>"
            "</figure>"
        )
    for note in missing:
        html.append(f"<p>{escape(note)}</p>")

    notes = report.meta.get("columns", {}).get("notes") or []
    if notes or report.meta.get("size_note"):
        html.append("<h2>Data notes</h2><ul>")
        html.extend(f"<li>{escape(n)}</li>" for n in notes)
        if report.meta.get("size_note"):
            html.append(f"<li>{escape(report.meta['size_note'])}</li>")
        html.append("</ul>")

    html.append("<h2>Integrity</h2>")
    for f in report.integrity:
        if verbose or f.flag is not Flag.OK:
            html.append(_finding_html(f))
        else:
            html.append(f"<p>{escape(f.check)}: {escape(f.title)}</p>")

    rest = [f for f in report.findings if f not in flagged]
    if rest:
        html.append(f"<h2>Other findings ({len(rest)})</h2>")
        if verbose:
            html.extend(_finding_html(f) for f in rest)
        else:
            html.append("<details><summary>Show the findings that were not flagged</summary>")
            html.extend(_finding_html(f) for f in rest)
            html.append("</details>")

    html.append("</main></body></html>")
    return "\n".join(html)
