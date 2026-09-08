"""Column-map files: writing one from auto-detection and reading one back.

The map encodes the decisions that change the answer, most of all which of the available
denominators counts as turnout, so it has to exist as a reviewable file rather than a run of
flags. These tests check that the file is a faithful record: what auto-detection guessed is
written with the guess visible to a reviewer, the loader preserves the notes verbatim, and
anything ambiguous or misspelled in the file is an error rather than a silent no-op.
"""

from __future__ import annotations

import json
import sys

import pytest

from psephos.cli import main as cli_main
from psephos.schema import ColumnMap, SchemaError, autodetect, load_column_map, write_column_map

# ---------------------------------------------------------------- writing


def test_written_map_round_trips_through_the_loader(clean_df, tmp_path):
    p = tmp_path / "map.yaml"
    guess = autodetect(clean_df, vote_pattern="^party_")
    write_column_map(guess, p)
    loaded = load_column_map(p)
    assert loaded.registered == "registered"
    assert loaded.ballots_cast == "ballots_cast"
    assert loaded.precinct_id == "precinct"
    assert loaded.region == "region"
    assert loaded.invalid is None, "an unset role must come back unset, not be omitted"
    assert loaded.votes == {
        "party_incumbent": "party_incumbent",
        "party_second": "party_second",
        "party_third": "party_third",
    }
    assert loaded.notes == guess.notes, "the loader must preserve the notes verbatim"


def test_written_map_shows_the_guesses_to_a_reviewer(clean_df, tmp_path):
    p = tmp_path / "map.yaml"
    guess = autodetect(clean_df)
    write_column_map(guess, p)
    text = p.read_text(encoding="utf-8")
    for note in guess.notes:
        assert f"# {note}" in text, "each guess must be visible as a comment in the file"
    loaded = load_column_map(p)
    assert any("unclaimed numeric columns" in n for n in loaded.notes)


def test_written_notes_stay_on_one_line_so_a_hand_edit_cannot_join_them(tmp_path):
    # A note wrapped across two lines is valid YAML, but a reviewer appending a note after the
    # continuation line produces a file that still parses, with the two notes silently joined.
    p = tmp_path / "map.yaml"
    long_note = (
        "no vote pattern given, so several unclaimed numeric columns were treated as "
        "contestants. Check that list before trusting any result."
    )
    write_column_map(ColumnMap(votes={}, notes=[long_note]), p)
    text = p.read_text(encoding="utf-8")
    assert f"- {long_note}" in text, "each note must be a single line a reviewer can append after"
    assert load_column_map(p).notes == [long_note]


def test_written_map_keeps_non_ascii_names_as_themselves(tmp_path):
    # Column names in the datasets this tool exists for are not ASCII, so the file must hold
    # them as themselves rather than as escapes, or the diff a reviewer reads is unreadable.
    # This source stays ASCII; the escapes decode to a Cyrillic word at run time.
    label = "\u041f\u0430\u0440\u0442\u0438\u044f A"
    p = tmp_path / "map.yaml"
    write_column_map(ColumnMap(votes={label: label}), p)
    assert label in p.read_text(encoding="utf-8")
    assert load_column_map(p).votes == {label: label}


# ---------------------------------------------------------------- reading


def test_loader_reads_a_hand_written_map(tmp_path):
    p = tmp_path / "election.yaml"
    p.write_text(
        'registered: "Number of voters on the register"\n'
        'ballots_cast: "Ballots found in stationary boxes"\n'
        'invalid: "Invalid ballots"\n'
        'precinct_id: "uik"\n'
        'region: "region"\n'
        "votes:\n"
        '  "Party A": "1. PARTY A"\n'
        '  "Party B": "2. PARTY B"\n'
        "notes:\n"
        '  - "Turnout uses ballots in boxes, not ballots issued. '
        'The two differ by lost ballots."\n',
        encoding="utf-8",
    )
    cmap = load_column_map(p)
    assert cmap.registered == "Number of voters on the register"
    assert cmap.ballots_cast == "Ballots found in stationary boxes"
    assert cmap.invalid == "Invalid ballots"
    assert cmap.precinct_id == "uik"
    assert cmap.region == "region"
    assert cmap.votes == {"Party A": "1. PARTY A", "Party B": "2. PARTY B"}
    assert cmap.notes == [
        "Turnout uses ballots in boxes, not ballots issued. The two differ by lost ballots."
    ]


def test_loader_leaves_unset_roles_unset(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("registered: reg\n", encoding="utf-8")
    cmap = load_column_map(p)
    assert cmap.registered == "reg"
    assert cmap.ballots_cast is None
    assert cmap.votes == {}
    assert cmap.notes == []


def test_loader_rejects_unknown_keys_instead_of_ignoring_a_typo(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("registerd: voters\nregistered: registered\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="registerd"):
        load_column_map(p)


def test_loader_rejects_a_file_that_is_not_a_mapping(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="mapping"):
        load_column_map(p)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SchemaError, match="empty"):
        load_column_map(empty)


def test_loader_rejects_unparsable_yaml(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("registered: [unclosed\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="not valid YAML"):
        load_column_map(p)


def test_loader_rejects_a_votes_block_that_is_not_a_mapping(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("votes:\n  - party_a\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="'votes'"):
        load_column_map(p)


def test_loader_rejects_a_role_given_a_list_of_names(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("registered:\n  - voters\n  - electorate\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="registered"):
        load_column_map(p)


def test_loader_rejects_an_empty_note(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("notes:\n  - something\n  -\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="notes"):
        load_column_map(p)


def test_loader_and_writer_degrade_without_pyyaml(tmp_path, clean_df, monkeypatch):
    # The venv has pyyaml; dropping it from sys.modules is how the import-time absence is
    # simulated, since `import yaml` raises ImportError when the entry is None.
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(SchemaError, match="pyyaml"):
        load_column_map(tmp_path / "map.yaml")
    with pytest.raises(SchemaError, match="pyyaml"):
        write_column_map(autodetect(clean_df), tmp_path / "map.yaml")


# ---------------------------------------------------------------- cli


def test_cli_audit_accepts_a_map_file_and_reports_its_notes(tmp_path, clean_df, capsys):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    mapfile = tmp_path / "map.yaml"
    write_column_map(autodetect(clean_df, vote_pattern="^party_"), mapfile)
    report_json = tmp_path / "r.json"
    code = cli_main(
        ["audit", str(results), "--map", str(mapfile), "--mc", "50", "--json", str(report_json)]
    )
    assert code == 0, "auditing with a map file is the normal path, not an error"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["n_units"] == len(clean_df)
    notes = payload["meta"]["columns"]["notes"]
    assert any("loaded from" in n for n in notes), "the report must say where the map came from"
    assert any("guessed registered" in n for n in notes), "the file's notes must reach the report"


def test_cli_audit_refuses_a_map_file_overridden_by_flags(tmp_path, clean_df, capsys):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    mapfile = tmp_path / "map.yaml"
    write_column_map(autodetect(clean_df, vote_pattern="^party_"), mapfile)
    code = cli_main(["audit", str(results), "--map", str(mapfile), "--registered", "registered"])
    assert code == 1
    assert "both" in capsys.readouterr().err


def test_cli_audit_refuses_a_map_file_with_a_vote_pattern(tmp_path, clean_df, capsys):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    mapfile = tmp_path / "map.yaml"
    write_column_map(autodetect(clean_df, vote_pattern="^party_"), mapfile)
    code = cli_main(["audit", str(results), "--map", str(mapfile), "--vote-pattern", "^party_"])
    assert code == 1
    assert "--vote-pattern" in capsys.readouterr().err


def test_cli_audit_with_a_missing_map_file_exits_one(tmp_path, clean_df, capsys):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    code = cli_main(["audit", str(results), "--map", str(tmp_path / "nope.yaml")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_audit_says_what_is_missing_when_pyyaml_is_absent(
    tmp_path, clean_df, monkeypatch, capsys
):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    monkeypatch.setitem(sys.modules, "yaml", None)
    code = cli_main(["audit", str(results), "--map", str(tmp_path / "map.yaml")])
    assert code == 1
    assert "pyyaml" in capsys.readouterr().err


def test_cli_columns_writes_a_starting_map(tmp_path, clean_df, capsys):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    draft = tmp_path / "draft.yaml"
    code = cli_main(["columns", str(results), "--write-map", str(draft)])
    assert code == 0
    assert str(draft) in capsys.readouterr().out
    assert load_column_map(draft).notes == autodetect(clean_df).notes


def test_cli_columns_says_what_is_missing_when_pyyaml_is_absent(
    tmp_path, clean_df, monkeypatch, capsys
):
    results = tmp_path / "e.csv"
    clean_df.to_csv(results, index=False)
    monkeypatch.setitem(sys.modules, "yaml", None)
    code = cli_main(["columns", str(results), "--write-map", str(tmp_path / "draft.yaml")])
    assert code == 1
    assert "pyyaml" in capsys.readouterr().err


def test_columns_command_errors_cleanly_on_a_missing_file(capsys):
    """`psephos columns` used to traceback where `psephos audit` exits 1 with a message."""
    from psephos.cli import main as cli_main

    assert cli_main(["columns", "definitely_not_here.csv"]) == 1
    assert "no such file" in capsys.readouterr().err
