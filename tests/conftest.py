"""Shared fixtures: deterministic sample files generated into tmp_path."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
import pytest


def make_trades_df() -> pd.DataFrame:
    """A small deterministic trades table used across tests."""
    return pd.DataFrame(
        {
            "trade_id": [1, 2, 3, 4, 5],
            "ts": [
                "2026-01-05 09:31:00",
                "2026-01-05 09:45:00",
                "2026-01-05 10:02:00",
                "2026-01-06 09:31:00",
                "2026-01-06 11:15:00",
            ],
            "ticker": ["AAPL", "MSFT", "AAPL", "NVDA", "MSFT"],
            "qty": [100, 250, 50, 300, 125],
            "price": [182.5, 410.0, 183.1, None, 411.25],
        }
    )


@pytest.fixture
def trades_df() -> pd.DataFrame:
    return make_trades_df()


@pytest.fixture
def data_dir(tmp_path: Path, trades_df: pd.DataFrame) -> Path:
    """A directory of mixed sample files, including a duplicate and a corrupt file."""
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)

    trades_df.to_csv(root / "trades.csv", index=False)
    trades_df.to_parquet(root / "trades.parquet", index=False)
    trades_df.to_json(root / "trades_records.json", orient="records")
    trades_df.to_json(root / "trades_lines.jsonl", orient="records", lines=True)
    with gzip.open(root / "trades.csv.gz", "wt", newline="") as fh:
        trades_df.to_csv(fh, index=False)

    # Byte-identical duplicate under a different name in a subdirectory.
    shutil.copyfile(root / "trades.csv", root / "sub" / "trades_copy_old.csv")

    # Corrupt CSV: unclosed quote and ragged rows.
    (root / "sub" / "broken.csv").write_text('a,b\n1,"oops\n2,3,4,5,6\n\x00junk')

    # A file type we do not support, which must be ignored.
    (root / "readme.txt").write_text("not data")
    return root
