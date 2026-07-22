# data-catalog

[![CI](https://github.com/lalitheswaran11-stack/data-catalog/actions/workflows/ci.yml/badge.svg)](https://github.com/lalitheswaran11-stack/data-catalog/actions/workflows/ci.yml)

`datacat` inventories local data files in SQLite. I built it to answer simple
questions about an unfamiliar research directory without opening every CSV or
Parquet file by hand: What contains `order_id`? Which files have the same
contents? Which export stopped parsing?

It currently reads CSV, gzipped CSV, Parquet, JSON records, and JSON Lines.
For each file it stores:

- path, format, size, row count, and scan timestamps;
- column names, inferred types, null rates, and numeric or date ranges;
- uniqueness-based candidate keys; and
- a content fingerprint used to group duplicate datasets.

## Quick start

```sh
uv sync
uv run python examples/make_sample_data.py
uv run datacat scan examples/sample_data
uv run datacat list
uv run datacat search order_id
uv run datacat dupes
```

The sample generator creates twelve small files with mixed schemas, one
duplicate, and one broken CSV. Rerunning it requires `--force`; that option
only replaces directories previously marked by the generator.

The catalog defaults to `./datacat.db`. Use `--db PATH` or set `DATACAT_DB`
to keep it elsewhere.

## Commands

| Command | Purpose |
| --- | --- |
| `datacat scan <dir>` | recursively profile supported files |
| `datacat list` | show the datasets already in the catalog |
| `datacat show <path-or-id>` | print one stored profile |
| `datacat search <term>` | search file paths, columns, and dtypes |
| `datacat dupes` | group matching content fingerprints |

Example duplicate output:

```text
Group 1 (fingerprint 25a6367013b08637, 400 rows):
  [5] .../research/trades_backup_final_v2.csv (17.6 KB)
  [6] .../trades_2026_01.csv (17.6 KB)
```

Parse failures are stored with `status = error`, so one damaged file does not
stop the rest of a directory scan.

## Implementation notes

The profiler uses pandas for schema statistics and `sqlite3` for storage.
Catalog rows are keyed by absolute path. A rescan skips a file when its
modification time has not changed; changed files are read again and compared
using their content fingerprints.

Candidate keys are suggestions based on non-null uniqueness, not semantic
primary-key declarations. Composite-key search is limited to eight
high-cardinality columns to avoid an unbounded combination search.

The current profiler loads each file into memory. Fingerprints normalize
column order but remain sensitive to row order and inferred dtypes. Those are
intentional limits for a local discovery tool rather than a distributed data
catalog.

Run the tests with:

```sh
uv run pytest
```
