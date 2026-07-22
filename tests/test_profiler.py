"""Tests for file discovery and profiling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datacat.profiler import (
    DatasetProfile,
    detect_format,
    discover_files,
    fingerprint_dataframe,
    profile_directory,
    profile_file,
)


def _by_name(profiles: list[DatasetProfile], name: str) -> DatasetProfile:
    matches = [p for p in profiles if Path(p.path).name == name]
    assert len(matches) == 1, f"expected exactly one profile named {name}"
    return matches[0]


class TestDiscovery:
    def test_finds_all_supported_formats_recursively(self, data_dir: Path) -> None:
        names = {p.name for p in discover_files(data_dir)}
        assert names == {
            "trades.csv",
            "trades.parquet",
            "trades_records.json",
            "trades_lines.jsonl",
            "trades.csv.gz",
            "trades_copy_old.csv",
            "broken.csv",
        }

    def test_ignores_unsupported_extensions(self, data_dir: Path) -> None:
        assert detect_format(data_dir / "readme.txt") is None

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("x.csv", "csv"),
            ("x.CSV", "csv"),
            ("x.csv.gz", "csv.gz"),
            ("x.parquet", "parquet"),
            ("x.json", "json"),
            ("x.jsonl", "jsonl"),
            ("x.ndjson", "jsonl"),
        ],
    )
    def test_detect_format(self, filename: str, expected: str) -> None:
        assert detect_format(Path(filename)) == expected


class TestProfileFile:
    def test_csv_basic_stats(self, data_dir: Path) -> None:
        profile = profile_file(data_dir / "trades.csv")
        assert profile.status == "ok"
        assert profile.row_count == 5
        assert profile.size_bytes > 0
        assert [c.name for c in profile.columns] == ["trade_id", "ts", "ticker", "qty", "price"]

        by_name = {c.name: c for c in profile.columns}
        assert by_name["trade_id"].dtype == "integer"
        assert by_name["qty"].dtype == "integer"
        assert by_name["price"].dtype == "float"
        assert by_name["ticker"].dtype == "string"

    def test_null_percentage(self, data_dir: Path) -> None:
        profile = profile_file(data_dir / "trades.csv")
        by_name = {c.name: c for c in profile.columns}
        assert by_name["price"].null_pct == pytest.approx(20.0)
        assert by_name["trade_id"].null_pct == 0.0

    def test_numeric_min_max(self, data_dir: Path) -> None:
        profile = profile_file(data_dir / "trades.csv")
        by_name = {c.name: c for c in profile.columns}
        assert by_name["trade_id"].min_value == "1"
        assert by_name["trade_id"].max_value == "5"
        assert by_name["price"].min_value == "182.5"
        assert by_name["price"].max_value == "411.25"

    def test_datetime_detection_with_range(self, data_dir: Path) -> None:
        profile = profile_file(data_dir / "trades.csv")
        ts = next(c for c in profile.columns if c.name == "ts")
        assert ts.is_datetime
        assert ts.date_min == "2026-01-05T09:31:00"
        assert ts.date_max == "2026-01-06T11:15:00"

    def test_integer_ids_not_flagged_as_dates(self, data_dir: Path) -> None:
        profile = profile_file(data_dir / "trades.csv")
        trade_id = next(c for c in profile.columns if c.name == "trade_id")
        assert not trade_id.is_datetime

    def test_candidate_keys_single_and_pair(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "account": ["A", "A", "B", "B"],
                "day": ["mon", "tue", "mon", "tue"],
                "value": [1.0, 1.0, 1.0, 1.0],
            }
        )
        path = tmp_path / "keys.csv"
        df.to_csv(path, index=False)
        profile = profile_file(path)
        assert ["id"] in profile.candidate_keys
        assert ["account", "day"] in profile.candidate_keys
        # Constant column can never be part of a minimal key pair.
        assert not any("value" in key for key in profile.candidate_keys)

    def test_corrupt_file_becomes_error_entry(self, data_dir: Path) -> None:
        profile = profile_file(data_dir / "sub" / "broken.csv")
        assert profile.status == "error"
        assert profile.error
        assert profile.row_count is None
        assert profile.fingerprint is None

    def test_empty_csv_with_headers_only(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("a,b,c\n")
        profile = profile_file(path)
        assert profile.status == "ok"
        assert profile.row_count == 0
        assert [c.name for c in profile.columns] == ["a", "b", "c"]
        assert all(c.null_pct == 0.0 for c in profile.columns)

    def test_truncated_gzip_is_an_error(self, tmp_path: Path) -> None:
        import gzip

        blob = gzip.compress(b"a,b\n1,2\n3,4\n" * 50)
        (tmp_path / "trunc.csv.gz").write_bytes(blob[: len(blob) // 2])
        profile = profile_file(tmp_path / "trunc.csv.gz")
        assert profile.status == "error"


class TestFingerprint:
    def test_same_content_different_names_match(self, data_dir: Path) -> None:
        a = profile_file(data_dir / "trades.csv")
        b = profile_file(data_dir / "sub" / "trades_copy_old.csv")
        assert a.fingerprint == b.fingerprint

    def test_different_content_differs(self, trades_df: pd.DataFrame) -> None:
        other = trades_df.copy()
        other.loc[0, "qty"] = 999
        assert fingerprint_dataframe(trades_df) != fingerprint_dataframe(other)

    def test_column_order_does_not_matter(self, trades_df: pd.DataFrame) -> None:
        shuffled = trades_df[list(reversed(trades_df.columns))]
        assert fingerprint_dataframe(trades_df) == fingerprint_dataframe(shuffled)


class TestFormats:
    @pytest.mark.parametrize(
        "filename",
        ["trades.parquet", "trades_records.json", "trades_lines.jsonl", "trades.csv.gz"],
    )
    def test_all_formats_profile_ok(self, data_dir: Path, filename: str) -> None:
        profile = profile_file(data_dir / filename)
        assert profile.status == "ok", profile.error
        assert profile.row_count == 5
        assert {c.name for c in profile.columns} == {"trade_id", "ts", "ticker", "qty", "price"}

    def test_profile_directory_counts(self, data_dir: Path) -> None:
        profiles = profile_directory(data_dir)
        assert len(profiles) == 7
        assert sum(p.status == "error" for p in profiles) == 1
        assert _by_name(profiles, "broken.csv").status == "error"
