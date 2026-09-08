"""Command line interface.

    psephos audit results.csv --registered "Registered voters" --votes "PartyA,PartyB"
    psephos audit results.csv --vote-pattern "^party_" --json report.json
    psephos columns results.csv
    psephos columns results.csv --write-map draft.yaml
    psephos audit results.csv --map election.yaml

Exit codes: 0 when the audit ran, 1 when it could not run at all (a missing file, an unreadable
table, a column map that names something absent). A flagged finding is NOT an error exit,
because flagged findings are the normal output of the tool and turning them into a failing exit
code invites a script to treat "anomaly" as "fraud".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psephos import __version__
from psephos.audit import DEFAULT_THRESHOLDS, audit
from psephos.report import to_json, to_text
from psephos.schema import (
    ColumnMap,
    SchemaError,
    autodetect,
    load,
    load_column_map,
    write_column_map,
)


def _build_map(args: argparse.Namespace, df_columns: list[str]) -> ColumnMap | None:
    """An explicit map if the user gave one, else None so the loader auto-detects."""
    if not any([args.votes, args.registered, args.ballots_cast, args.invalid, args.precinct_id]):
        return None
    cmap = ColumnMap(
        registered=args.registered,
        ballots_cast=args.ballots_cast,
        invalid=args.invalid,
        precinct_id=args.precinct_id,
        region=args.region,
        district=args.district,
    )
    if args.votes:
        labels = [c.strip() for c in args.votes.split(",") if c.strip()]
        missing = [c for c in labels if c not in df_columns]
        if missing:
            raise SchemaError(f"vote columns not in the table: {missing}")
        cmap.votes = {c: c for c in labels}
        cmap.notes.append(f"vote columns given explicitly: {len(labels)}")
    return cmap


def _cmd_columns(args: argparse.Namespace) -> int:
    import pandas as pd

    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1
    df = (
        pd.read_parquet(path)
        if path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
    )
    cmap = autodetect(df, vote_pattern=args.vote_pattern)
    print(f"{path}: {len(df)} rows, {len(df.columns)} columns\n")
    print("COLUMNS")
    for c in df.columns:
        print(f"  {c!r}  ({df[c].dtype})")
    print("\nAUTODETECTED MAP")
    for role in ("registered", "ballots_cast", "invalid", "precinct_id", "region", "district"):
        print(f"  {role:14} {getattr(cmap, role)!r}")
    print(f"  {'votes':14} {len(cmap.votes)} columns: {list(cmap.votes)[:8]}")
    print("\nNOTES")
    for n in cmap.notes:
        print(f"  - {n}")
    if args.write_map:
        try:
            write_column_map(cmap, args.write_map)
        except SchemaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"\nwrote {args.write_map}: the map above, with every guess recorded as a note. "
            "Edit what is wrong, then audit with --map."
        )
    print(
        "\nAuto-detection is a convenience, not a guarantee. Pass --registered and --votes "
        "explicitly for anything you intend to rely on."
    )
    return 0


def _reject_map_conflicts(args: argparse.Namespace) -> None:
    """Refuse a mapping given twice.

    A map file exists so the mapping can be reviewed as a file. Letting a shell history line
    silently override it would make the reviewed artifact decorative, so the two ways to set the
    mapping do not combine.
    """
    flags = ("registered", "ballots_cast", "invalid", "precinct_id", "region", "district", "votes")
    clashes = [f"--{f.replace('_', '-')}" for f in flags if getattr(args, f)]
    if clashes:
        raise SchemaError(
            f"the mapping is given both by --map and by {', '.join(clashes)}. Use one or the "
            "other: the map file is meant to be the reviewed source of truth."
        )
    if args.vote_pattern:
        raise SchemaError(
            "--map already names the vote columns, so --vote-pattern would be ignored. Drop one."
        )


def _cmd_audit(args: argparse.Namespace) -> int:
    import pandas as pd

    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1
    try:
        peek = (
            pd.read_parquet(path)
            if path.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(
                path, nrows=5, sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
            )
        )
        if args.map:
            _reject_map_conflicts(args)
            cmap = load_column_map(args.map)
            cmap.notes.append(f"column map loaded from {args.map}")
        else:
            cmap = _build_map(args, list(peek.columns))
        data = load(path, columns=cmap, vote_pattern=args.vote_pattern)
    except (SchemaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = audit(
        data,
        thresholds=tuple(args.thresholds) if args.thresholds else DEFAULT_THRESHOLDS,
        n_mc=args.mc,
        seed=args.seed,
        by_size=not args.no_strata,
    )

    if args.json:
        Path(args.json).write_text(to_json(report), encoding="utf-8")
        print(f"wrote {args.json}")
    if not args.quiet:
        print(to_text(report, verbose=args.verbose))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="psephos",
        description=(
            "Statistical anomaly detection for precinct-level election results. "
            "Flags patterns worth explaining; it does not detect fraud."
        ),
    )
    p.add_argument("--version", action="version", version=f"psephos {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    cols = sub.add_parser("columns", help="show the table's columns and the auto-detected map")
    cols.add_argument("path")
    cols.add_argument("--vote-pattern", default=None, help="regex selecting contestant columns")
    cols.add_argument(
        "--write-map",
        default=None,
        metavar="PATH",
        help="write the auto-detected map to a YAML file, as a starting point to edit",
    )
    cols.set_defaults(func=_cmd_columns)

    aud = sub.add_parser("audit", help="run every applicable check")
    aud.add_argument("path")
    aud.add_argument("--registered", default=None, help="registered-voters column")
    aud.add_argument("--ballots-cast", default=None, help="ballots-counted column")
    aud.add_argument("--invalid", default=None, help="invalid-ballots column")
    aud.add_argument("--precinct-id", default=None, help="precinct identifier column")
    aud.add_argument("--region", default=None)
    aud.add_argument("--district", default=None)
    aud.add_argument("--votes", default=None, help="comma-separated contestant columns")
    aud.add_argument(
        "--map",
        default=None,
        metavar="PATH",
        help="load the column map from a YAML file (see columns --write-map)",
    )
    aud.add_argument("--vote-pattern", default=None, help="regex selecting contestant columns")
    aud.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=None,
        help="precinct-size thresholds for the percentage sweep",
    )
    aud.add_argument("--mc", type=int, default=500, help="Monte Carlo replicates (default 500)")
    aud.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    aud.add_argument("--no-strata", action="store_true", help="skip the per-size-band view")
    aud.add_argument("--json", default=None, metavar="PATH", help="also write a JSON report")
    aud.add_argument("-v", "--verbose", action="store_true", help="show every finding")
    aud.add_argument("-q", "--quiet", action="store_true", help="suppress the text report")
    aud.set_defaults(func=_cmd_audit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
