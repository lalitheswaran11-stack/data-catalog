"""Tests for the SQLite catalog store."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from datacat.catalog import Catalog
from datacat.profiler import profile_directory, profile_file


def _scan_into(catalog: Catalog, root: Path) -> dict[str, int]:
    """Profile ``root`` and upsert everything, returning outcome counts."""
    counts: dict[str, int] = {}
    for profile in profile_directory(root):
        outcome = catalog.upsert(profile).outcome
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


class TestUpsert:
    def test_first_scan_is_all_new(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            counts = _scan_into(catalog, data_dir)
            assert counts == {"new": 6, "error": 1}
            assert len(catalog.list_datasets()) == 7

    def test_rescan_without_changes_is_unchanged(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            counts = _scan_into(catalog, data_dir)
            assert counts.get("new", 0) == 0
            assert counts.get("unchanged", 0) == 6
            # Path count must not grow on re-scan.
            assert len(catalog.list_datasets()) == 7

    def test_modified_file_is_updated(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            target = data_dir / "trades.csv"
            df = pd.read_csv(target)
            df.loc[0, "qty"] = 12345
            df.to_csv(target, index=False)

            profile = profile_file(target)
            assert catalog.upsert(profile).outcome == "updated"

    def test_needs_rescan_tracks_mtime(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            target = data_dir / "trades.csv"
            profile = profile_file(target)
            catalog.upsert(profile)
            assert not catalog.needs_rescan(profile.path, profile.mtime)
            assert catalog.needs_rescan(profile.path, profile.mtime + 1.0)
            assert catalog.needs_rescan("/nowhere/else.csv", 0.0)

    def test_first_seen_survives_rescan(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            target = data_dir / "trades.csv"
            catalog.upsert(profile_file(target))
            first = catalog.get_dataset(str(target))
            assert first is not None

            time.sleep(1.1)  # ISO timestamps have second resolution
            catalog.upsert(profile_file(target))
            second = catalog.get_dataset(str(target))
            assert second is not None
            assert second["first_seen"] == first["first_seen"]
            assert second["last_scanned"] > first["last_scanned"]

    def test_error_file_recorded_with_status(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            profile = profile_file(data_dir / "sub" / "broken.csv")
            assert catalog.upsert(profile).outcome == "error"
            row = catalog.get_dataset(str(data_dir / "sub" / "broken.csv"))
            assert row is not None
            assert row["status"] == "error"
            assert row["error"]

    def test_columns_replaced_not_duplicated(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            target = data_dir / "trades.csv"
            catalog.upsert(profile_file(target))
            catalog.upsert(profile_file(target))
            row = catalog.get_dataset(str(target))
            assert row is not None
            assert len(catalog.get_columns(row["id"])) == 5

    def test_candidate_keys_round_trip(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            catalog.upsert(profile_file(data_dir / "trades.csv"))
            row = catalog.get_dataset(str(data_dir / "trades.csv"))
            assert row is not None
            keys = json.loads(row["candidate_keys"])
            assert ["trade_id"] in keys


class TestLookup:
    def test_get_by_id_path_and_suffix(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            by_path = catalog.get_dataset(str(data_dir / "trades.parquet"))
            assert by_path is not None
            assert catalog.get_dataset(str(by_path["id"])) is not None
            by_suffix = catalog.get_dataset("trades.parquet")
            assert by_suffix is not None
            assert by_suffix["id"] == by_path["id"]

    def test_ambiguous_suffix_returns_none(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            # Both trades.csv and trades_copy_old.csv end in ".csv".
            assert catalog.get_dataset(".csv") is None


class TestSearch:
    def test_search_by_column_name(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            results = catalog.search("trade_id")
            paths = {r["path"] for r in results}
            # All six parseable files share the trade_id column.
            assert len(paths) == 6
            assert all("trade_id" in r["matched"] for r in results)

    def test_search_by_file_name_and_dtype(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            by_file = catalog.search("copy_old")
            assert {Path(r["path"]).name for r in by_file} == {"trades_copy_old.csv"}
            by_dtype = catalog.search("float")
            assert all(r["detail"] == "float" for r in by_dtype)
            assert len(by_dtype) == 6

    def test_search_no_results(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            assert catalog.search("no_such_thing_xyz") == []


class TestDuplicates:
    def test_duplicate_detection_across_names(self, tmp_path: Path, data_dir: Path) -> None:
        with Catalog(tmp_path / "cat.db") as catalog:
            _scan_into(catalog, data_dir)
            rows = catalog.duplicates()
            names = {Path(r["path"]).name for r in rows}
            # The CSV twins plus every same-content re-encoding of trades
            # share a fingerprint; at minimum the byte-copies must be flagged.
            assert {"trades.csv", "trades_copy_old.csv"} <= names
            fingerprints = {r["fingerprint"] for r in rows}
            assert len(fingerprints) >= 1
