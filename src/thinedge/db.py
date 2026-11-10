"""DuckDB access: an in-memory connection with a view per processed Parquet table."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import PROCESSED_DIR, TABLES


def connect(processed_dir: Path = PROCESSED_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for table in TABLES:
        path = processed_dir / f"{table}.parquet"
        if path.exists():
            quoted = path.as_posix().replace("'", "''")
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{quoted}')")
    return con
