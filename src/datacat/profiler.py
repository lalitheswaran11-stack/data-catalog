"""File discovery and dataset profiling.

Walks a directory tree for tabular data files (CSV, gzipped CSV, Parquet,
JSON records / JSON lines), loads each into a DataFrame, and produces a
:class:`DatasetProfile` describing its schema and contents. All profiling
statistics are computed with vectorized pandas operations; a file that cannot
be parsed yields a profile with ``status="error"`` rather than raising.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

#: File-name suffixes we know how to load, mapped to a format label.
_SUFFIX_FORMATS: dict[str, str] = {
    ".csv": "csv",
    ".csv.gz": "csv.gz",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}

#: Fraction of non-null values that must parse as datetimes for a string
#: column to be flagged as a date column.
_DATETIME_THRESHOLD = 0.9

#: Maximum number of columns considered when searching for composite
#: (two-column) primary-key candidates, to cap combinatorics.
_MAX_KEY_SEARCH_COLUMNS = 8


@dataclass
class ColumnProfile:
    """Profile of a single column within a dataset."""

    name: str
    dtype: str
    null_pct: float
    min_value: str | None = None
    max_value: str | None = None
    is_datetime: bool = False
    date_min: str | None = None
    date_max: str | None = None


@dataclass
class DatasetProfile:
    """Profile of one data file.

    ``status`` is ``"ok"`` for successfully profiled files and ``"error"``
    for files that could not be parsed, in which case ``error`` holds the
    reason and the statistical fields are empty.
    """

    path: str
    format: str
    size_bytes: int
    mtime: float
    status: str = "ok"
    error: str | None = None
    row_count: int | None = None
    fingerprint: str | None = None
    columns: list[ColumnProfile] = field(default_factory=list)
    candidate_keys: list[list[str]] = field(default_factory=list)


def detect_format(path: Path) -> str | None:
    """Return the format label for ``path``, or ``None`` if unsupported."""
    name = path.name.lower()
    if name.endswith(".csv.gz"):
        return "csv.gz"
    return _SUFFIX_FORMATS.get(path.suffix.lower())


def discover_files(root: Path) -> list[Path]:
    """Recursively find all supported data files under ``root``, sorted by path."""
    found = [p for p in root.rglob("*") if p.is_file() and detect_format(p) is not None]
    return sorted(found)


def _json_is_records(path: Path) -> bool:
    """Return True if a ``.json`` file holds a top-level array (records orient)."""
    with open(path, encoding="utf-8") as fh:
        chunk = fh.read(4096).lstrip()
    return chunk.startswith("[")


def load_dataframe(path: Path, fmt: str) -> pd.DataFrame:
    """Load ``path`` into a DataFrame according to its format label.

    Raises whatever the underlying reader raises for malformed input; the
    caller is responsible for converting failures into error profiles.
    """
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "csv.gz":
        # Validate the gzip stream eagerly: pandas can silently stop at a
        # truncated member, so decompress up front to surface corruption.
        with gzip.open(path, "rb") as fh:
            fh.read()
        return pd.read_csv(path, compression="gzip")
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    if fmt == "json":
        return pd.read_json(path, lines=not _json_is_records(path))
    raise ValueError(f"unsupported format: {fmt}")


def _dtype_label(dtype: object) -> str:
    """Map a pandas dtype to a short, reader-friendly label."""
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype):
        return "string"
    return str(dtype)


def _detect_datetime(series: pd.Series) -> pd.Series | None:
    """Try to parse a string column as datetimes.

    Returns the parsed series if at least :data:`_DATETIME_THRESHOLD` of the
    non-null values convert, otherwise ``None``. Purely numeric values are
    rejected so integer-like ID columns are not mistaken for epoch dates.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    as_str = non_null.astype(str)
    # Reject columns that are just numbers (e.g. "1024") — to_datetime would
    # happily parse some of them as dates like "1024-01-01".
    if pd.to_numeric(as_str, errors="coerce").notna().all():
        return None
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    ratio = parsed.notna().sum() / len(non_null)
    return parsed if ratio >= _DATETIME_THRESHOLD else None


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    """Compute a content fingerprint for a DataFrame.

    The hash covers the sorted column names and every row's values (via
    :func:`pandas.util.hash_pandas_object`), so byte-identical copies of a
    dataset stored under different file names produce the same fingerprint.
    """
    cols = sorted(str(c) for c in df.columns)
    digest = hashlib.sha256()
    digest.update("\x1f".join(cols).encode("utf-8"))
    if not df.empty and cols:
        row_hashes = pd.util.hash_pandas_object(df[cols], index=False)
        digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def _candidate_keys(df: pd.DataFrame) -> list[list[str]]:
    """Find single columns and column pairs that are 100% unique and non-null.

    Pair search is capped at :data:`_MAX_KEY_SEARCH_COLUMNS` columns (chosen
    by descending cardinality) and skips pairs containing a column that is
    already a key on its own.
    """
    if df.empty:
        return []
    n = len(df)
    nunique = df.nunique(dropna=True)
    non_null = df.notna().all()

    singles = [c for c in df.columns if non_null[c] and nunique[c] == n]
    keys: list[list[str]] = [[str(c)] for c in singles]

    # Composite candidates: highest-cardinality non-key columns first.
    pool = [c for c in df.columns if c not in singles and non_null[c] and nunique[c] > 1]
    pool = sorted(pool, key=lambda c: int(nunique[c]), reverse=True)
    pool = pool[:_MAX_KEY_SEARCH_COLUMNS]
    for a, b in combinations(pool, 2):
        if not df.duplicated(subset=[a, b]).any():
            keys.append(sorted([str(a), str(b)]))
    return keys


def _profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    """Build per-column profiles (dtype, null %, ranges) using vectorized ops."""
    # fillna guards the zero-row case, where mean() of an empty column is NaN.
    null_pct = (df.isna().mean() * 100).fillna(0.0).round(2)
    profiles: list[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        profile = ColumnProfile(
            name=str(col),
            dtype=_dtype_label(series.dtype),
            null_pct=float(null_pct[col]),
        )
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
            series.dtype
        ):
            if series.notna().any():
                profile.min_value = repr(series.min().item())
                profile.max_value = repr(series.max().item())
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            profile.is_datetime = True
            profile.dtype = "datetime"
            if series.notna().any():
                profile.date_min = series.min().isoformat()
                profile.date_max = series.max().isoformat()
        elif pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
            parsed = _detect_datetime(series)
            if parsed is not None and parsed.notna().any():
                profile.is_datetime = True
                profile.date_min = parsed.min().isoformat()
                profile.date_max = parsed.max().isoformat()
        profiles.append(profile)
    return profiles


def profile_file(path: Path) -> DatasetProfile:
    """Profile a single data file, never raising for bad content.

    Unreadable or malformed files produce a profile with ``status="error"``
    and the exception message recorded in ``error``.
    """
    fmt = detect_format(path) or "unknown"
    stat = path.stat()
    profile = DatasetProfile(
        path=str(path.resolve()),
        format=fmt,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
    )
    try:
        df = load_dataframe(path, fmt)
        if not isinstance(df, pd.DataFrame):  # scalar / malformed JSON edge cases
            raise ValueError(f"parsed to {type(df).__name__}, not a table")
    except Exception as exc:  # noqa: BLE001 — any parse failure becomes an error entry
        profile.status = "error"
        profile.error = f"{type(exc).__name__}: {exc}"
        return profile

    profile.row_count = int(len(df))
    profile.fingerprint = fingerprint_dataframe(df)
    profile.columns = _profile_columns(df)
    profile.candidate_keys = _candidate_keys(df)
    return profile


def profile_directory(root: Path) -> list[DatasetProfile]:
    """Discover and profile every supported file under ``root``."""
    return [profile_file(path) for path in discover_files(root)]
