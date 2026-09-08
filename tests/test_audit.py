"""Schema, integrity, audit orchestration, report rendering and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import clean_election, with_rounded_turnout

from psephos._types import Finding, Flag
from psephos.audit import audit
from psephos.cli import main as cli_main
from psephos.integrity import run_integrity
from psephos.report import INTERPRETATION, to_json, to_text
from psephos.schema import ColumnMap, ElectionData, SchemaError, autodetect, load
from psephos.size import integer_reachability, stratify

# ---------------------------------------------------------------- the guardrail


def test_a_flagged_finding_cannot_omit_confounds():
    """The single most important invariant in the package: nothing gets flagged without an
    innocent explanation printed beside it."""
    with pytest.raises(ValueError, match="must name at least one confound"):
        Finding(check="x", title="t", flag=Flag.STRONG)
    with pytest.raises(ValueError, match="must name at least one confound"):
        Finding(check="x", title="t", flag=Flag.NOTABLE)
    Finding(check="x", title="t", flag=Flag.OK)  # unflagged needs none


def test_reports_always_carry_the_interpretation(clean_data):
    rep = audit(clean_data, n_mc=60)
    text = to_text(rep)
    assert "does not detect fraud" in text
    assert INTERPRETATION.strip()[:40] in text
    payload = json.loads(to_json(rep))
    assert "interpretation" in payload and "disclaimer" in payload


# ---------------------------------------------------------------- schema


def test_autodetect_records_every_guess(clean_df):
    cmap = autodetect(clean_df, vote_pattern="^party_")
    assert cmap.registered == "registered"
    assert len(cmap.votes) == 3
    assert cmap.notes, "auto-detection must explain itself"


def test_autodetect_without_a_pattern_warns_about_unclaimed_columns(clean_df):
    cmap = autodetect(clean_df)
    assert any("unclaimed numeric columns" in n for n in cmap.notes)


def test_column_map_require_raises_with_an_actionable_message():
    with pytest.raises(SchemaError, match="--registered"):
        ColumnMap().require("registered")


def test_empty_table_is_rejected(column_map):
    with pytest.raises(SchemaError, match="no rows"):
        ElectionData(frame=pd.DataFrame(columns=["registered"]), columns=column_map)


def test_missing_vote_column_is_rejected(clean_df):
    bad = ColumnMap(votes={"ghost": "not_a_column"}, registered="registered")
    with pytest.raises(SchemaError, match="not in the table"):
        ElectionData(frame=clean_df, columns=bad)


def test_share_denominators_differ_and_the_caller_chooses(clean_data):
    valid = clean_data.share("party_incumbent", of="valid")
    cast = clean_data.share("party_incumbent", of="cast")
    reg = clean_data.share("party_incumbent", of="registered")
    assert np.nanmean(reg) < np.nanmean(cast) <= np.nanmean(valid) + 1e-9
    with pytest.raises(ValueError, match="of must be"):
        clean_data.share("party_incumbent", of="nonsense")


def test_winner_is_the_largest_total(clean_data):
    assert clean_data.winner_label() == "party_incumbent"


def test_load_roundtrip_csv(tmp_path, clean_df):
    p = tmp_path / "e.csv"
    clean_df.to_csv(p, index=False)
    data = load(p, vote_pattern="^party_")
    assert data.n == len(clean_df)
    assert data.columns.registered == "registered"


# ---------------------------------------------------------------- integrity


def test_integrity_is_quiet_on_a_coherent_table(clean_data):
    findings = run_integrity(clean_data)
    bad = [f for f in findings if f.flag is Flag.STRONG]
    assert not bad, f"clean table flagged: {[f.title for f in bad]}"


def test_integrity_catches_turnout_over_100(clean_df, column_map):
    df = clean_df.copy()
    df.loc[df.index[:20], "ballots_cast"] = df.loc[df.index[:20], "registered"] + 5
    findings = {f.check: f for f in run_integrity(ElectionData(frame=df, columns=column_map))}
    assert findings["turnout_over_100"].flag is Flag.STRONG
    assert findings["turnout_over_100"].statistic == 20
    assert findings["turnout_over_100"].confounds


def test_integrity_catches_duplicates_and_negatives(clean_df, column_map):
    df = pd.concat([clean_df, clean_df.iloc[:5]], ignore_index=True)
    df.loc[df.index[0], "party_second"] = -1
    findings = {f.check: f for f in run_integrity(ElectionData(frame=df, columns=column_map))}
    assert findings["duplicate_units"].flag is Flag.STRONG
    assert findings["negative_counts"].flag is Flag.STRONG


def test_integrity_reports_not_applicable_rather_than_failing(clean_df):
    minimal = ColumnMap(votes={"party_incumbent": "party_incumbent"})
    findings = {f.check: f for f in run_integrity(ElectionData(frame=clean_df, columns=minimal))}
    assert findings["duplicate_units"].flag is Flag.NOT_APPLICABLE
    assert findings["turnout_over_100"].flag is Flag.NOT_APPLICABLE


# ---------------------------------------------------------------- size


def test_integer_reachability_is_one_at_a_hundred_and_falls_away():
    assert integer_reachability(100) == pytest.approx(1.0)
    assert integer_reachability(2000) < 0.15
    assert integer_reachability(100) > integer_reachability(500) > integer_reachability(5000)


def test_stratify_partitions_without_overlap(clean_df):
    reg = clean_df["registered"].to_numpy(float)
    strata = stratify(reg)
    stacked = np.vstack([s.mask for s in strata])
    assert stacked.sum(axis=0).max() <= 1, "strata overlap"
    assert stacked.any(axis=0).sum() == int((reg > 0).sum())


# ---------------------------------------------------------------- audit


def test_audit_of_clean_data_flags_close_to_the_nominal_rate(clean_data):
    """An audit runs about eighteen checks. Demanding that NONE of them flags on clean data
    asserts that eighteen tests at alpha 0.05 all got lucky, which is expecting a bit under
    two in five audits.

    Measured over twenty independent clean elections: 28 flags out of 360 findings, about 7.8
    per cent against a 5 per cent nominal, or 1.4 flags per audit. That mild excess is exactly
    why psephos reports the flagged count against the total, and why issue #12 (a
    multiple-testing correction) is on the v0.2 milestone. This test pins the rate so it
    cannot drift upward unnoticed.
    """
    rep = audit(clean_data, n_mc=200, seed=0)
    assert rep.n_units == clean_data.n
    assert not [f for f in rep.flagged if f.flag is Flag.STRONG], (
        f"a STRONG flag on clean data: {[f.title for f in rep.flagged]}"
    )

    total = flagged = 0
    for s in range(12):
        df = clean_election(n=4000, seed=400 + s)
        r = audit(ElectionData(frame=df, columns=clean_data.columns, source="clean"), n_mc=120)
        total += len(r.findings)
        flagged += len(r.flagged)
    rate = flagged / total
    assert rate < 0.15, f"clean-data flag rate {rate:.1%} is too far above the 5% nominal"


def test_audit_of_rounded_data_flags_the_percentage_family(clean_df, column_map):
    dirty = ElectionData(
        frame=with_rounded_turnout(clean_df, fraction=0.15),
        columns=column_map,
        source="synthetic-rounded",
    )
    rep = audit(dirty, n_mc=200, seed=0)
    checks = {f.check for f in rep.flagged}
    assert "integer_percentage" in checks
    # The signal must survive raising the size threshold, which is what separates it from an
    # artefact of small precincts.
    strict = [
        f
        for f in rep.flagged
        if f.check == "integer_percentage" and "min_denominator=1000" in f.slice_name
    ]
    assert strict, "signal vanished at the strict threshold"


def test_audit_runs_the_threshold_sweep(clean_data):
    rep = audit(clean_data, thresholds=(100, 500), n_mc=60)
    slices = {f.slice_name for f in rep.findings if f.check == "integer_percentage"}
    assert any("min_denominator=100" in s for s in slices)
    assert any("min_denominator=500" in s for s in slices)


def test_audit_without_registered_says_what_is_missing(clean_df):
    cmap = ColumnMap(votes={"party_incumbent": "party_incumbent"})
    rep = audit(ElectionData(frame=clean_df, columns=cmap), n_mc=50)
    na = [f for f in rep.findings if f.flag is Flag.NOT_APPLICABLE]
    assert any("registered-voters column" in f.title for f in na)


def test_audit_never_raises_on_a_broken_check(clean_df, column_map, monkeypatch):
    import psephos.methods.integer_pct as ip

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(ip, "integer_excess", boom)
    rep = audit(ElectionData(frame=clean_df, columns=column_map), n_mc=30)
    assert any("Check failed" in f.title for f in rep.findings)


def test_audit_is_reproducible(clean_data):
    a = audit(clean_data, n_mc=80, seed=3)
    b = audit(clean_data, n_mc=80, seed=3)
    assert [f.statistic for f in a.findings] == [f.statistic for f in b.findings]


# ---------------------------------------------------------------- cli


def test_cli_columns(tmp_path, clean_df, capsys):
    p = tmp_path / "e.csv"
    clean_df.to_csv(p, index=False)
    assert cli_main(["columns", str(p), "--vote-pattern", "^party_"]) == 0
    out = capsys.readouterr().out
    assert "AUTODETECTED MAP" in out and "registered" in out


def test_cli_audit_writes_json_and_exits_zero_even_when_flagged(tmp_path, clean_df, capsys):
    p = tmp_path / "e.csv"
    with_rounded_turnout(clean_df, fraction=0.2).to_csv(p, index=False)
    out_json = tmp_path / "r.json"
    code = cli_main(
        [
            "audit",
            str(p),
            "--registered",
            "registered",
            "--ballots-cast",
            "ballots_cast",
            "--votes",
            "party_incumbent,party_second,party_third",
            "--mc",
            "80",
            "--json",
            str(out_json),
        ]
    )
    assert code == 0, "a flagged finding must not become a failing exit code"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_units"] == len(clean_df)
    assert payload["disclaimer"]
    assert "does not detect fraud" in capsys.readouterr().out


def test_cli_missing_file_exits_one(capsys):
    assert cli_main(["audit", "no_such_file.csv"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_cli_bad_vote_column_exits_one(tmp_path, clean_df, capsys):
    p = tmp_path / "e.csv"
    clean_df.to_csv(p, index=False)
    assert cli_main(["audit", str(p), "--votes", "nope"]) == 1
    assert "not in the table" in capsys.readouterr().err
