"""SQLite-backed catalog store for dataset profiles.

Schema
------
``datasets``
    One row per file path, upserted on re-scan. Tracks ``first_seen`` /
    ``last_scanned`` timestamps plus the mtime and content fingerprint used
    for change detection.
``columns``
    One row per column of each successfully profiled dataset; replaced
    wholesale whenever its dataset is re-profiled.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from datacat.profiler import ColumnProfile, DatasetProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id             INTEGER PRIMARY KEY,
    path           TEXT NOT NULL UNIQUE,
    format         TEXT NOT NULL,
    status         TEXT NOT NULL,
    error          TEXT,
    size_bytes     INTEGER NOT NULL,
    mtime          REAL NOT NULL,
    row_count      INTEGER,
    fingerprint    TEXT,
    candidate_keys TEXT NOT NULL DEFAULT '[]',
    first_seen     TEXT NOT NULL,
    last_scanned   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS columns (
    id         INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL,
    name       TEXT NOT NULL,
    dtype      TEXT NOT NULL,
    null_pct   REAL NOT NULL,
    min_value  TEXT,
    max_value  TEXT,
    is_datetime INTEGER NOT NULL DEFAULT 0,
    date_min   TEXT,
    date_max   TEXT
);

CREATE INDEX IF NOT EXISTS idx_columns_dataset ON columns(dataset_id);
CREATE INDEX IF NOT EXISTS idx_columns_name ON columns(name);
CREATE INDEX IF NOT EXISTS idx_datasets_fingerprint ON datasets(fingerprint);
"""


def _utcnow() -> str:
    """Current UTC time as a second-resolution ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class UpsertResult:
    """Outcome of storing one profile: ``new``, ``updated``, ``unchanged`` or ``error``."""

    path: str
    outcome: str


class Catalog:
    """A dataset catalog persisted in a single SQLite file."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def needs_rescan(self, path: str, mtime: float) -> bool:
        """Return False only if ``path`` is already cataloged with this mtime.

        Used to skip re-profiling files that have not been touched since the
        last scan; content changes are confirmed via fingerprint at upsert.
        """
        row = self._conn.execute(
            "SELECT mtime, status FROM datasets WHERE path = ?", (path,)
        ).fetchone()
        return row is None or row["status"] == "error" or row["mtime"] != mtime

    def touch(self, path: str) -> None:
        """Refresh ``last_scanned`` for an unchanged dataset."""
        self._conn.execute(
            "UPDATE datasets SET last_scanned = ? WHERE path = ?", (_utcnow(), path)
        )
        self._conn.commit()

    def upsert(self, profile: DatasetProfile) -> UpsertResult:
        """Insert or update the catalog entry for ``profile.path``.

        Returns an :class:`UpsertResult` whose outcome distinguishes brand-new
        files, changed files (different mtime or fingerprint), unchanged
        files, and parse errors.
        """
        now = _utcnow()
        existing = self._conn.execute(
            "SELECT id, fingerprint, mtime FROM datasets WHERE path = ?",
            (profile.path,),
        ).fetchone()

        if profile.status == "error":
            outcome = "error"
        elif existing is None:
            outcome = "new"
        elif (
            existing["fingerprint"] == profile.fingerprint
            and existing["mtime"] == profile.mtime
        ):
            outcome = "unchanged"
        else:
            outcome = "updated"

        cur = self._conn.execute(
            """
            INSERT INTO datasets (
                path, format, status, error, size_bytes, mtime, row_count,
                fingerprint, candidate_keys, first_seen, last_scanned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                format = excluded.format,
                status = excluded.status,
                error = excluded.error,
                size_bytes = excluded.size_bytes,
                mtime = excluded.mtime,
                row_count = excluded.row_count,
                fingerprint = excluded.fingerprint,
                candidate_keys = excluded.candidate_keys,
                last_scanned = excluded.last_scanned
            """,
            (
                profile.path,
                profile.format,
                profile.status,
                profile.error,
                profile.size_bytes,
                profile.mtime,
                profile.row_count,
                profile.fingerprint,
                json.dumps(profile.candidate_keys),
                now,
                now,
            ),
        )
        dataset_id = cur.lastrowid if existing is None else existing["id"]

        self._conn.execute("DELETE FROM columns WHERE dataset_id = ?", (dataset_id,))
        self._conn.executemany(
            """
            INSERT INTO columns (
                dataset_id, position, name, dtype, null_pct,
                min_value, max_value, is_datetime, date_min, date_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    dataset_id,
                    idx,
                    col.name,
                    col.dtype,
                    col.null_pct,
                    col.min_value,
                    col.max_value,
                    int(col.is_datetime),
                    col.date_min,
                    col.date_max,
                )
                for idx, col in enumerate(profile.columns)
            ],
        )
        self._conn.commit()
        return UpsertResult(path=profile.path, outcome=outcome)

    # -- reads -------------------------------------------------------------

    def list_datasets(self) -> list[sqlite3.Row]:
        """All datasets ordered by path."""
        return self._conn.execute("SELECT * FROM datasets ORDER BY path").fetchall()

    def get_dataset(self, ref: str) -> sqlite3.Row | None:
        """Look up one dataset by numeric id, exact path, or unique path suffix."""
        if ref.isdigit():
            row = self._conn.execute(
                "SELECT * FROM datasets WHERE id = ?", (int(ref),)
            ).fetchone()
            if row is not None:
                return row
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE path = ?", (str(Path(ref).resolve()),)
        ).fetchone()
        if row is not None:
            return row
        matches = self._conn.execute(
            "SELECT * FROM datasets WHERE path LIKE ? ORDER BY path",
            (f"%{ref}",),
        ).fetchall()
        return matches[0] if len(matches) == 1 else None

    def get_columns(self, dataset_id: int) -> list[sqlite3.Row]:
        """Column profiles for one dataset in original file order."""
        return self._conn.execute(
            "SELECT * FROM columns WHERE dataset_id = ? ORDER BY position",
            (dataset_id,),
        ).fetchall()

    def search(self, term: str) -> list[sqlite3.Row]:
        """Case-insensitive substring match on file names, column names and dtypes.

        Returns one row per (dataset, match reason) pair.
        """
        pattern = f"%{term}%"
        return self._conn.execute(
            """
            SELECT d.id, d.path, d.row_count, d.status,
                   'file name' AS matched, d.path AS detail
            FROM datasets d WHERE d.path LIKE ?
            UNION
            SELECT d.id, d.path, d.row_count, d.status,
                   'column: ' || c.name AS matched, c.dtype AS detail
            FROM datasets d JOIN columns c ON c.dataset_id = d.id
            WHERE c.name LIKE ? OR c.dtype LIKE ?
            ORDER BY path, matched
            """,
            (pattern, pattern, pattern),
        ).fetchall()

    def duplicates(self) -> list[sqlite3.Row]:
        """Datasets sharing a content fingerprint with at least one other file."""
        return self._conn.execute(
            """
            SELECT d.fingerprint, d.id, d.path, d.row_count, d.size_bytes
            FROM datasets d
            JOIN (
                SELECT fingerprint FROM datasets
                WHERE status = 'ok' AND fingerprint IS NOT NULL
                GROUP BY fingerprint HAVING COUNT(*) > 1
            ) dupes ON dupes.fingerprint = d.fingerprint
            ORDER BY d.fingerprint, d.path
            """
        ).fetchall()


def column_from_row(row: sqlite3.Row) -> ColumnProfile:
    """Rehydrate a :class:`ColumnProfile` from a ``columns`` table row."""
    return ColumnProfile(
        name=row["name"],
        dtype=row["dtype"],
        null_pct=row["null_pct"],
        min_value=row["min_value"],
        max_value=row["max_value"],
        is_datetime=bool(row["is_datetime"]),
        date_min=row["date_min"],
        date_max=row["date_max"],
    )
