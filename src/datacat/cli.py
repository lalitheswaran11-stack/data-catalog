"""``datacat`` command-line interface.

Subcommands
-----------
- ``datacat scan <dir>``: profile every supported file and upsert the catalog.
- ``datacat list``: tabular overview of all cataloged datasets.
- ``datacat show <path-or-id>``: full profile of one dataset.
- ``datacat search <term>``: match file names, column names, and dtypes.
- ``datacat dupes``: groups of datasets sharing a content fingerprint.

The catalog location defaults to ``./datacat.db`` and can be overridden with
``--db`` or the ``DATACAT_DB`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from datacat import __version__
from datacat.catalog import Catalog
from datacat.profiler import discover_files, profile_file
from datacat.render import format_table, human_size

_DEFAULT_DB = "datacat.db"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="datacat",
        description="Profile directories of data files and build a searchable catalog.",
    )
    parser.add_argument("--version", action="version", version=f"datacat {__version__}")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("DATACAT_DB", _DEFAULT_DB)),
        help=f"catalog database file (default: {_DEFAULT_DB}, env: DATACAT_DB)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="profile a directory and update the catalog")
    p_scan.add_argument("directory", type=Path, help="directory to scan recursively")
    p_scan.add_argument(
        "--force",
        action="store_true",
        help="re-profile every file even if its mtime is unchanged",
    )

    sub.add_parser("list", help="tabular overview of the catalog")

    p_show = sub.add_parser("show", help="full profile of one dataset")
    p_show.add_argument("ref", help="dataset id, path, or unique path suffix")

    p_search = sub.add_parser("search", help="search file names, column names, dtypes")
    p_search.add_argument("term", help="substring to match (case-insensitive)")

    sub.add_parser("dupes", help="datasets sharing a content fingerprint")
    return parser


def _cmd_scan(catalog: Catalog, directory: Path, force: bool) -> int:
    """Profile all files under ``directory``, upsert them, and print a summary."""
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    files = discover_files(directory)
    counts: Counter[str] = Counter()
    rows: list[list[object]] = []
    for path in files:
        resolved = str(path.resolve())
        if not force and not catalog.needs_rescan(resolved, path.stat().st_mtime):
            catalog.touch(resolved)
            counts["unchanged"] += 1
            rows.append([path.name, "unchanged", ""])
            continue
        profile = profile_file(path)
        result = catalog.upsert(profile)
        counts[result.outcome] += 1
        rows.append([path.name, result.outcome, profile.error or ""])

    print(format_table(["file", "outcome", "note"], rows))
    print(
        f"\nScanned {len(files)} files: "
        f"{counts['new']} new, {counts['updated']} updated, "
        f"{counts['unchanged']} unchanged, {counts['error']} errors"
    )
    return 0


def _cmd_list(catalog: Catalog) -> int:
    """Print a one-line-per-dataset overview of the catalog."""
    datasets = catalog.list_datasets()
    if not datasets:
        print("Catalog is empty. Run `datacat scan <dir>` first.")
        return 0
    rows = [
        [
            d["id"],
            Path(d["path"]).name,
            d["format"],
            d["status"],
            d["row_count"],
            len(catalog.get_columns(d["id"])) or None,
            human_size(d["size_bytes"]),
        ]
        for d in datasets
    ]
    print(format_table(["id", "file", "format", "status", "rows", "cols", "size"], rows))
    return 0


def _cmd_show(catalog: Catalog, ref: str) -> int:
    """Print the full stored profile for one dataset."""
    dataset = catalog.get_dataset(ref)
    if dataset is None:
        print(f"error: no unique dataset matching {ref!r}", file=sys.stderr)
        return 1

    print(f"Path:         {dataset['path']}")
    print(f"Format:       {dataset['format']}")
    print(f"Status:       {dataset['status']}")
    if dataset["error"]:
        print(f"Error:        {dataset['error']}")
    print(f"Size:         {human_size(dataset['size_bytes'])}")
    print(f"Rows:         {dataset['row_count'] if dataset['row_count'] is not None else '-'}")
    fingerprint = dataset["fingerprint"]
    print(f"Fingerprint:  {fingerprint[:16] if fingerprint else '-'}")
    print(f"First seen:   {dataset['first_seen']}")
    print(f"Last scanned: {dataset['last_scanned']}")

    keys = json.loads(dataset["candidate_keys"])
    if keys:
        rendered = ", ".join("(" + ", ".join(k) + ")" for k in keys)
        print(f"Candidate keys: {rendered}")

    columns = catalog.get_columns(dataset["id"])
    if columns:
        rows = [
            [
                c["name"],
                f"{c['dtype']} (date)" if c["is_datetime"] and c["dtype"] != "datetime" else c["dtype"],
                f"{c['null_pct']:.1f}%",
                c["date_min"] if c["is_datetime"] else c["min_value"],
                c["date_max"] if c["is_datetime"] else c["max_value"],
            ]
            for c in columns
        ]
        print()
        print(format_table(["column", "dtype", "nulls", "min", "max"], rows))
    return 0


def _cmd_search(catalog: Catalog, term: str) -> int:
    """Print datasets matching ``term`` in file name, column name, or dtype."""
    results = catalog.search(term)
    if not results:
        print(f"No matches for {term!r}.")
        return 0
    rows = [
        [r["id"], Path(r["path"]).name, r["matched"], r["detail"] if r["matched"] != "file name" else ""]
        for r in results
    ]
    print(format_table(["id", "file", "matched", "dtype"], rows))
    return 0


def _cmd_dupes(catalog: Catalog) -> int:
    """Print groups of datasets that share a content fingerprint."""
    rows = catalog.duplicates()
    if not rows:
        print("No duplicate datasets found.")
        return 0
    current: str | None = None
    group = 0
    for row in rows:
        if row["fingerprint"] != current:
            current = row["fingerprint"]
            group += 1
            print(f"Group {group} (fingerprint {current[:16]}, {row['row_count']} rows):")
        print(f"  [{row['id']}] {row['path']} ({human_size(row['size_bytes'])})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _build_parser().parse_args(argv)
    with Catalog(args.db) as catalog:
        if args.command == "scan":
            return _cmd_scan(catalog, args.directory, args.force)
        if args.command == "list":
            return _cmd_list(catalog)
        if args.command == "show":
            return _cmd_show(catalog, args.ref)
        if args.command == "search":
            return _cmd_search(catalog, args.term)
        if args.command == "dupes":
            return _cmd_dupes(catalog)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
