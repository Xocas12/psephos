"""The progress reporter writes to stderr, stays off when asked, and clears up after itself."""

from __future__ import annotations

import io

from psephos.progress import StderrProgress


def test_disabled_reporter_hands_back_no_callback():
    """None rather than a no-op, so the check can skip the call entirely."""
    assert StderrProgress(enabled=False).for_check("x") is None


def test_report_goes_to_the_given_stream_and_names_the_check():
    buf = io.StringIO()
    prog = StderrProgress(stream=buf, min_interval=0.0)
    report = prog.for_check("integer percentage, turnout, min_denominator=250")
    assert report is not None
    report(250, 500)
    out = buf.getvalue()
    assert "integer percentage, turnout, min_denominator=250" in out
    assert "250/500" in out
    assert "50%" in out
    assert out.startswith("\r"), "progress rewrites one line rather than scrolling"
    assert "\n" not in out, "a newline would break the line it rewrites"


def test_rate_limit_drops_intermediate_reports_but_never_the_last():
    """A finished check must report, or the line is never cleared."""
    buf = io.StringIO()
    prog = StderrProgress(stream=buf, min_interval=1000.0)
    report = prog.for_check("check")
    assert report is not None
    report(1, 500)
    first = buf.getvalue()
    report(2, 500)
    assert buf.getvalue() == first, "the second report is inside the interval and is dropped"
    report(500, 500)
    assert "500/500" in buf.getvalue()


def test_done_clears_the_line_so_the_report_starts_at_column_zero():
    buf = io.StringIO()
    prog = StderrProgress(stream=buf, min_interval=0.0)
    report = prog.for_check("a long check name to pad over")
    assert report is not None
    report(1, 2)
    buf.truncate(0)
    buf.seek(0)
    prog.done()
    cleared = buf.getvalue()
    assert cleared.startswith("\r") and cleared.endswith("\r")
    assert cleared.strip("\r ") == ""


def test_done_is_safe_when_nothing_was_reported():
    buf = io.StringIO()
    prog = StderrProgress(stream=buf, min_interval=0.0)
    prog.done()
    assert buf.getvalue() == ""


def test_a_shorter_line_erases_the_tail_of_a_longer_one():
    """Without padding, '100/100' over '1/1000' would leave stray characters behind."""
    buf = io.StringIO()
    prog = StderrProgress(stream=buf, min_interval=0.0)
    long_report = prog.for_check("a very long check name indeed")
    assert long_report is not None
    long_report(1, 1000)
    width = len(buf.getvalue().lstrip("\r"))
    buf.truncate(0)
    buf.seek(0)
    short = prog.for_check("short")
    assert short is not None
    short(1000, 1000)
    assert len(buf.getvalue().lstrip("\r")) == width
