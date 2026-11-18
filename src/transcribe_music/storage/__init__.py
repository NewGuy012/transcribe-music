"""Shared SQLite connection helpers for local cache and score catalogs."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_sqlite(
    db_path: Path,
    *,
    read_only: bool = False,
    wal: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with transaction handling and guaranteed close."""
    if read_only:
        con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=10.0)
    else:
        con = sqlite3.connect(str(db_path), timeout=10.0)
        if wal:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
    try:
        with con:
            yield con
    finally:
        con.close()
