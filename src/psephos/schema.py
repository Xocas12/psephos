"""Loading precinct results and mapping whatever columns they came with.

Election results arrive with different column names in different languages, so the mapping is
explicit and inspectable. Auto-detection exists for convenience and always records what it
guessed; it never guesses silently. A mapping can also be written to a YAML file and read back,
so the decisions it encodes can be committed, diffed and argued with rather than buried in a
command line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Candidate column names for the roles psephos needs, lowercased and matched as substrings.
#: Extend these rather than special-casing a dataset. Cyrillic terms are included because the
#: largest openly published precinct-level datasets use them.
HINTS: dict[str, tuple[str, ...]] = {
    "registered": (
        "registered",
        "electorate",
        "eligible",
        "num_voters",
        "внесенных в список",
    ),
    "ballots_cast": (
        "ballots_cast",
        "votes_cast",
        "total_votes",
        "turnout_count",
        "в стационарных ящиках",
    ),
    "invalid": ("invalid", "spoilt", "spoiled", "rejected", "недействительных"),
    "precinct_id": ("precinct", "uik", "station", "polling", "участ", "unit_id"),
    "region": ("region", "province", "oblast", "область", "регион"),
    "district": ("district", "tik", "constituency", "county", "муниципал"),
}


#: The role names a column map can set, in the order they are written out.
MAP_ROLES: tuple[str, ...] = (
    "registered",
    "ballots_cast",
    "invalid",
    "precinct_id",
    "region",
    "district",
)

#: Every key a map file may contain. Anything else is a typo, and the loader says so rather than
#: skipping it, because a misspelled role left unset quietly disables the checks that need it.
MAP_KEYS: frozenset[str] = frozenset(MAP_ROLES) | {"votes", "notes"}


class SchemaError(ValueError):
    """The data does not carry what a check needs, and psephos will not guess."""


@dataclass
class ColumnMap:
    """Which column plays which role.

    ``votes`` maps a contestant label to its column. Everything else is optional, but each
    missing role disables the checks that need it, and the report says which.
    """

    votes: dict[str, str] = field(default_factory=dict)
    registered: str | None = None
    ballots_cast: str | None = None
    invalid: str | None = None
    precinct_id: str | None = None
    region: str | None = None
    district: str | None = None
    notes: list[str] = field(default_factory=list)

    def require(self, *roles: str) -> None:
        missing = [r for r in roles if not getattr(self, r, None)]
        if missing:
            raise SchemaError(
                f"this check needs {missing}, which the column map does not provide. "
                "Pass it explicitly, for example --registered COLUMN."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "votes": dict(self.votes),
            "registered": self.registered,
            "ballots_cast": self.ballots_cast,
            "invalid": self.invalid,
            "precinct_id": self.precinct_id,
            "region": self.region,
            "district": self.district,
            "notes": list(self.notes),
        }


def _match(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {c: str(c).lower() for c in columns}
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def autodetect(df: pd.DataFrame, vote_pattern: str | None = None) -> ColumnMap:
    """Guess a column map, recording every guess in ``notes``.

    ``vote_pattern`` is a regular expression selecting the contestant columns. Without it, any
    numeric column not claimed by another role is treated as a contestant, which is right often
    enough to be useful and wrong often enough that the notes matter.
    """
    cols = list(df.columns)
    cmap = ColumnMap()
    for role in MAP_ROLES:
        found = _match(cols, HINTS[role])
        if found is not None:
            setattr(cmap, role, found)
            cmap.notes.append(f"guessed {role} = {found!r}")

    claimed = {getattr(cmap, r) for r in MAP_ROLES} - {None}

    if vote_pattern:
        rx = re.compile(vote_pattern)
        vote_cols = [c for c in cols if rx.search(str(c))]
        cmap.notes.append(f"vote columns from pattern {vote_pattern!r}: {len(vote_cols)} matched")
    else:
        vote_cols = [c for c in cols if c not in claimed and pd.api.types.is_numeric_dtype(df[c])]
        cmap.notes.append(
            f"no vote pattern given, so {len(vote_cols)} unclaimed numeric columns were "
            "treated as contestants. Check that list before trusting any result."
        )
    cmap.votes = {str(c): c for c in vote_cols}
    return cmap


def _import_yaml() -> Any:
    """pyyaml is deliberately not a core dependency, so it is imported where it is used and its
    absence is reported as the missing piece it is, not as an ImportError from nowhere."""
    try:
        import yaml
    except ImportError as exc:
        raise SchemaError(
            "column-map files need pyyaml, which is not installed. "
            "Install psephos with the yaml extra: pip install 'psephos[yaml]'"
        ) from exc
    return yaml


def _as_name(value: Any, what: str) -> str:
    """One column name out of a YAML scalar. A list where a single name belongs is a mistake."""
    if value is None:
        raise SchemaError(f"{what} is empty; give a column name or drop the key")
    if isinstance(value, (dict, list)):
        raise SchemaError(f"{what} must be a single column name, not a {type(value).__name__}")
    return str(value)


def load_column_map(path: str | Path) -> ColumnMap:
    """Read a column map from a YAML file.

    The file is the reviewed artifact, so the loader is strict on purpose. An unknown key is an
    error rather than something to skip, because a misspelled role would otherwise sit silently
    unset and disable every check that needs it. ``notes`` entries are carried through verbatim;
    they are the record of what was guessed rather than chosen.

    Raises :class:`SchemaError` if the file is not YAML, is empty, is not a mapping, or holds
    something a map cannot set.
    """
    yaml = _import_yaml()
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SchemaError(f"{path}: not valid YAML: {exc}") from exc

    if raw is None:
        raise SchemaError(f"{path}: the file is empty")
    if not isinstance(raw, dict):
        raise SchemaError(
            f"{path}: a column map is a mapping of roles to column names, not a "
            f"{type(raw).__name__}"
        )

    unknown = sorted(str(k) for k in raw if k not in MAP_KEYS)
    if unknown:
        raise SchemaError(
            f"{path}: unknown map key(s) {', '.join(unknown)}. A map can hold "
            f"{', '.join(sorted(MAP_KEYS))}. An unknown key is a typo, and skipping it would "
            "leave the role it meant to set unset."
        )

    cmap = ColumnMap()
    for role in MAP_ROLES:
        if raw.get(role) is not None:
            setattr(cmap, role, _as_name(raw[role], f"{path}: {role}"))

    if raw.get("votes") is not None:
        votes = raw["votes"]
        if not isinstance(votes, dict):
            raise SchemaError(
                f"{path}: 'votes' must map a contestant label to its column, not be a "
                f"{type(votes).__name__}"
            )
        for label, col in votes.items():
            cmap.votes[_as_name(label, f"{path}: vote label")] = _as_name(
                col, f"{path}: vote column for {label!r}"
            )

    if raw.get("notes") is not None:
        notes = raw["notes"]
        if not isinstance(notes, list):
            raise SchemaError(f"{path}: 'notes' must be a list of short text entries")
        for note in notes:
            if isinstance(note, (dict, list)):
                raise SchemaError(
                    f"{path}: 'notes' entries must be text, not a {type(note).__name__}"
                )
            if note is None:
                raise SchemaError(f"{path}: 'notes' has an empty entry")
            cmap.notes.append(str(note))
    return cmap


def write_column_map(cmap: ColumnMap, path: str | Path) -> None:
    """Write a column map to a YAML file, as ``psephos columns --write-map`` does.

    The file is meant to be committed, diffed and argued with, so the notes go in twice on
    purpose: once as comments at the top, where a reviewer reading the file sees them, and once
    as ``notes`` entries, so they survive being loaded back. Unset roles are written as null
    rather than omitted, because the roles a dataset lacks are part of the record.
    """
    yaml = _import_yaml()
    lines = [
        "# Column map for psephos, written by: psephos columns --write-map",
        "#",
        "# Everything below was auto-detected, not chosen. A guess that is wrong changes the",
        "# answer, most obviously which of the available columns serves as the denominator for",
        "# turnout. Edit this file, then audit with --map, so the decision lives in the file",
        "# rather than in somebody's shell history.",
        "#",
        "# What auto-detection guessed, as comments here and as notes entries below:",
    ]
    for note in cmap.notes:
        parts = str(note).splitlines() or [""]
        lines.extend(f"# {part}" for part in parts)
    data: dict[str, Any] = {role: getattr(cmap, role) for role in MAP_ROLES}
    data["votes"] = dict(cmap.votes)
    data["notes"] = list(cmap.notes)
    # width keeps every scalar on one line. A note wrapped across two lines is valid YAML, but
    # a reviewer appending a note after a continuation line produces a file that still parses,
    # with the two notes silently joined; one line per entry leaves nothing to append into.
    lines.append(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100000).rstrip("\n")
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class ElectionData:
    """A precinct table plus the map that says what its columns mean."""

    frame: pd.DataFrame
    columns: ColumnMap
    source: str = "<memory>"

    def __post_init__(self) -> None:
        if len(self.frame) == 0:
            raise SchemaError("the table has no rows")
        for role in ("registered", "ballots_cast", "invalid"):
            col = getattr(self.columns, role)
            if col is not None and col not in self.frame.columns:
                raise SchemaError(f"{role} column {col!r} is not in the table")
        for label, col in self.columns.votes.items():
            if col not in self.frame.columns:
                raise SchemaError(f"vote column {col!r} for {label!r} is not in the table")

    @property
    def n(self) -> int:
        return len(self.frame)

    def registered(self) -> np.ndarray:
        self.columns.require("registered")
        return self.frame[self.columns.registered].to_numpy(dtype=float)

    def votes_frame(self) -> pd.DataFrame:
        if not self.columns.votes:
            raise SchemaError("no vote columns are mapped")
        return self.frame[list(self.columns.votes.values())].astype(float)

    def total_votes(self) -> np.ndarray:
        """Ballots counted: the mapped total if there is one, else the contestant sum plus
        invalid ballots where that column exists."""
        if self.columns.ballots_cast:
            return self.frame[self.columns.ballots_cast].to_numpy(dtype=float)
        total = self.votes_frame().sum(axis=1).to_numpy(dtype=float)
        if self.columns.invalid:
            total = total + self.frame[self.columns.invalid].to_numpy(dtype=float)
        return total

    def turnout(self) -> np.ndarray:
        """Ballots counted over registered voters, as a percentage. ``nan`` where the
        denominator is not positive."""
        reg = self.registered()
        cast = self.total_votes()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(reg > 0, 100.0 * cast / reg, np.nan)

    def winner_label(self) -> str:
        """The contestant with the largest total. Ties break on column order."""
        totals = self.votes_frame().sum(axis=0)
        col = totals.idxmax()
        for label, mapped in self.columns.votes.items():
            if mapped == col:
                return label
        return str(col)

    def share(self, label: str, of: str = "valid") -> np.ndarray:
        """Vote share of one contestant, as a percentage.

        ``of="valid"`` divides by the contestant sum, ``of="cast"`` by all ballots counted
        including invalid ones, ``of="registered"`` by the electorate. The three answer
        different questions and the choice belongs to the caller.
        """
        if label not in self.columns.votes:
            raise SchemaError(f"unknown contestant {label!r}; have {list(self.columns.votes)}")
        num = self.frame[self.columns.votes[label]].to_numpy(dtype=float)
        if of == "valid":
            den = self.votes_frame().sum(axis=1).to_numpy(dtype=float)
        elif of == "cast":
            den = self.total_votes()
        elif of == "registered":
            den = self.registered()
        else:
            raise ValueError(f"of must be 'valid', 'cast' or 'registered'; got {of!r}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, 100.0 * num / den, np.nan)


def load(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    vote_pattern: str | None = None,
    **read_kwargs: Any,
) -> ElectionData:
    """Read a precinct table from CSV, TSV or parquet and map its columns."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, **read_kwargs)
    elif suffix in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t", **read_kwargs)
    else:
        df = pd.read_csv(path, **read_kwargs)
    cmap = columns if columns is not None else autodetect(df, vote_pattern=vote_pattern)
    return ElectionData(frame=df, columns=cmap, source=str(path))
