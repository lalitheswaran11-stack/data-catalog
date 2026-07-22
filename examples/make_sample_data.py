"""Generate a deliberately messy sample data directory for the README demo.

Creates ``examples/sample_data/`` containing a dozen mixed files: CSVs,
gzipped CSV, Parquet, JSON records, JSON lines, overlapping schemas, one
byte-for-byte duplicate under a different name, and one corrupt file.
Deterministic (fixed seed), no network.

Usage::

    uv run python examples/make_sample_data.py [output_dir]
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260720
_MARKER_FILENAME = ".datacat-sample-data"


def _prepare_output(out_dir: Path, force: bool = False) -> Path:
    """Create a safe output directory for generated sample files."""
    resolved = out_dir.expanduser().resolve()
    protected = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected:
        raise ValueError(f"refusing to use protected directory: {resolved}")

    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not force:
            raise FileExistsError(
                f"output directory is not empty: {resolved}; pass --force to replace generated data"
            )
        if not (resolved / _MARKER_FILENAME).is_file():
            raise ValueError(f"refusing to replace unmarked directory: {resolved}")
        shutil.rmtree(resolved)

    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / _MARKER_FILENAME).write_text(
        "Created by examples/make_sample_data.py\n",
        encoding="utf-8",
    )
    return resolved


def _trades(rng: np.random.Generator, n: int, start: str) -> pd.DataFrame:
    """Synthetic equity trade blotter."""
    tickers = np.array(["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG"])
    ts = pd.date_range(start, periods=n, freq="47min")
    return pd.DataFrame(
        {
            "trade_id": np.arange(1, n + 1),
            "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": rng.choice(tickers, size=n),
            "side": rng.choice(["BUY", "SELL"], size=n),
            "qty": rng.integers(1, 5000, size=n),
            "price": (rng.uniform(90, 900, size=n)).round(2),
        }
    )


def _quotes(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Synthetic top-of-book quotes with some missing sizes."""
    ts = pd.date_range("2026-03-02 09:30", periods=n, freq="s")
    bid = rng.uniform(99, 101, size=n).round(4)
    df = pd.DataFrame(
        {
            "quote_time": ts,
            "symbol": rng.choice(["AAPL", "MSFT", "NVDA"], size=n),
            "bid": bid,
            "ask": (bid + rng.uniform(0.01, 0.05, size=n)).round(4),
            "bid_size": rng.integers(100, 10_000, size=n).astype(float),
        }
    )
    df.loc[rng.random(n) < 0.07, "bid_size"] = np.nan
    return df


def _positions(rng: np.random.Generator) -> pd.DataFrame:
    """End-of-day positions keyed by (date, account, ticker)."""
    dates = pd.date_range("2026-01-05", periods=10, freq="B")
    accounts = ["ALPHA", "BETA"]
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    idx = pd.MultiIndex.from_product(
        [dates, accounts, tickers], names=["as_of_date", "account", "ticker"]
    )
    df = idx.to_frame(index=False)
    df["as_of_date"] = df["as_of_date"].dt.strftime("%Y-%m-%d")
    df["net_qty"] = rng.integers(-20_000, 20_000, size=len(df))
    df["mkt_value"] = (df["net_qty"] * rng.uniform(90, 900, size=len(df))).round(2)
    return df


def _orders(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Order log sharing column names with the trade files."""
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
            "ticker": rng.choice(["AAPL", "TSLA", "AMZN", "META"], size=n),
            "created_at": pd.date_range("2026-04-01", periods=n, freq="13min").strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "qty": rng.integers(1, 2000, size=n),
            "limit_price": np.where(
                rng.random(n) < 0.2, np.nan, rng.uniform(50, 700, size=n).round(2)
            ),
        }
    )


def _fx_rates(rng: np.random.Generator) -> pd.DataFrame:
    """Daily FX fixes."""
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    idx = pd.MultiIndex.from_product([dates, pairs], names=["fix_date", "pair"])
    df = idx.to_frame(index=False)
    df["fix_date"] = df["fix_date"].dt.strftime("%Y-%m-%d")
    df["rate"] = rng.uniform(0.8, 160, size=len(df)).round(5)
    return df


def main(out_dir: Path, force: bool = False) -> None:
    """Build the sample directory, replacing only marked generated output."""
    rng = np.random.default_rng(SEED)
    out_dir = _prepare_output(out_dir, force=force)
    (out_dir / "vendor" / "refinitiv").mkdir(parents=True)
    (out_dir / "research").mkdir(parents=True)

    # 1-2: two months of trades, overlapping schema.
    _trades(rng, 400, "2026-01-05 09:30").to_csv(
        out_dir / "trades_2026_01.csv", index=False
    )
    _trades(rng, 380, "2026-02-02 09:30").to_csv(
        out_dir / "trades_2026_02.csv", index=False
    )
    # 3: gzipped CSV of trades.
    with gzip.open(out_dir / "trades_2026_03.csv.gz", "wt", newline="") as fh:
        _trades(rng, 420, "2026-03-02 09:30").to_csv(fh, index=False)
    # 4: quotes as Parquet.
    _quotes(rng, 5000).to_parquet(
        out_dir / "vendor" / "quotes_march.parquet", index=False
    )
    # 5: positions as CSV.
    _positions(rng).to_csv(out_dir / "positions_eod.csv", index=False)
    # 6: orders as JSON lines.
    _orders(rng, 250).to_json(
        out_dir / "orders_stream.jsonl", orient="records", lines=True
    )
    # 7: FX rates as JSON records array.
    _fx_rates(rng).to_json(
        out_dir / "vendor" / "refinitiv" / "fx_fixes.json", orient="records"
    )
    # 8: a small reference table.
    pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META"],
            "name": [
                "Apple",
                "Microsoft",
                "NVIDIA",
                "Tesla",
                "Amazon",
                "Alphabet",
                "Meta",
            ],
            "sector": [
                "Tech",
                "Tech",
                "Tech",
                "Autos",
                "Retail",
                "Tech",
                "Tech",
            ],
        }
    ).to_csv(out_dir / "research" / "ticker_reference.csv", index=False)
    # 9: duplicate of the January trades under a misleading name.
    shutil.copyfile(
        out_dir / "trades_2026_01.csv",
        out_dir / "research" / "trades_backup_final_v2.csv",
    )
    # 10: corrupt CSV (truncated mid-write, ragged rows).
    (out_dir / "vendor" / "broken_export.csv").write_text(
        'trade_id,ts,ticker\n1,"2026-01-05 09:31,AAPL\n2,,,,,,EXTRA,COLS\n\x00\x00garbage'
    )
    # 11: JSON that is not a table at all.
    (out_dir / "research" / "notes.json").write_text(
        '{"author": "lali", "note": "todo"}'
    )
    # 12: empty CSV with headers only.
    (out_dir / "vendor" / "signals_pending.csv").write_text("signal_id,name,decay\n")

    n_files = sum(
        1
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != _MARKER_FILENAME
    )
    print(f"Wrote {n_files} files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "sample_data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing directory created by this script",
    )
    args = parser.parse_args()
    try:
        main(args.output_dir, force=args.force)
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
