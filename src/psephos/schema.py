"""Loading precinct results and mapping whatever columns they came with.

Election results arrive with different column names in different languages, so the mapping is
explicit and inspectable. Auto-detection exists for convenience and always records what it
guessed; it never guesses silently.
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
    roles = ("registered", "ballots_cast", "invalid", "precinct_id", "region", "district")
    for role in roles:
        found = _match(cols, HINTS[role])
        if found is not None:
            setattr(cmap, role, found)
            cmap.notes.append(f"guessed {role} = {found!r}")

    claimed = {getattr(cmap, r) for r in roles} - {None}

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
