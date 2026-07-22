"""End-to-end tests for the ``datacat`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from datacat.cli import main


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "catalog.db"


def run(db_path: Path, *args: str) -> int:
    """Invoke the CLI as ``datacat --db <db> <args...>``."""
    return main(["--db", str(db_path), *args])


class TestScan:
    def test_scan_summary(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert run(db_path, "scan", str(data_dir)) == 0
        out = capsys.readouterr().out
        assert "Scanned 7 files: 6 new, 0 updated, 0 unchanged, 1 errors" in out
        assert "broken.csv" in out

    def test_rescan_reports_unchanged(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "scan", str(data_dir)) == 0
        out = capsys.readouterr().out
        assert "6 unchanged" in out

    def test_scan_force_reprofiles(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "scan", "--force", str(data_dir)) == 0
        out = capsys.readouterr().out
        assert "6 unchanged" in out  # content fingerprints still match

    def test_scan_missing_directory_fails(self, db_path: Path, tmp_path: Path) -> None:
        assert run(db_path, "scan", str(tmp_path / "nope")) == 2


class TestList:
    def test_empty_catalog(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert run(db_path, "list") == 0
        assert "Catalog is empty" in capsys.readouterr().out

    def test_lists_all_datasets(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "list") == 0
        out = capsys.readouterr().out
        for name in ("trades.csv", "trades.parquet", "broken.csv"):
            assert name in out
        assert "error" in out


class TestShow:
    def test_show_by_suffix(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "show", "trades.parquet") == 0
        out = capsys.readouterr().out
        assert "Rows:         5" in out
        assert "trade_id" in out
        assert "Candidate keys" in out

    def test_show_unknown_ref(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "show", "does_not_exist.csv") == 1
        assert "no unique dataset" in capsys.readouterr().err


class TestSearch:
    def test_search_column(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "search", "ticker") == 0
        out = capsys.readouterr().out
        assert "column: ticker" in out
        assert "trades.parquet" in out

    def test_search_no_match(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "search", "zzz_nothing") == 0
        assert "No matches" in capsys.readouterr().out


class TestDupes:
    def test_reports_duplicate_group(self, db_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(db_path, "scan", str(data_dir))
        capsys.readouterr()
        assert run(db_path, "dupes") == 0
        out = capsys.readouterr().out
        assert "trades_copy_old.csv" in out
        assert "Group 1" in out

    def test_no_dupes(self, db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        solo = tmp_path / "solo"
        solo.mkdir()
        (solo / "only.csv").write_text("a,b\n1,2\n")
        run(db_path, "scan", str(solo))
        capsys.readouterr()
        assert run(db_path, "dupes") == 0
        assert "No duplicate" in capsys.readouterr().out
