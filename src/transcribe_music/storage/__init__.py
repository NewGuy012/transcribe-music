"""Shared SQLite connection helpers for local cache and score catalogs."""

from __future__ import annotations

import os
import sqlite3
import tempfile
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


@contextmanager
def atomic_sqlite(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Build a fresh SQLite DB in a secure temp file and atomically replace `db_path`."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=db_path.parent,
        prefix=f".{db_path.stem}_",
        suffix=".sqlite.tmp",
        delete=False,
    ) as fh:
        tmp_db = Path(fh.name)
    try:
        with open_sqlite(tmp_db) as con:
            con.execute("PRAGMA journal_mode=OFF")
            con.execute("PRAGMA synchronous=OFF")
            yield con
        os.replace(tmp_db, db_path)
    finally:
        tmp_db.unlink(missing_ok=True)


def count_sqlite_rows(db_path: Path, table: str) -> int:
    """Count rows in `table` if `db_path` exists and is readable, returning 0 otherwise."""
    if not table.isidentifier():
        raise ValueError(f"Invalid SQLite table identifier: {table!r}")
    if not db_path.exists():
        return 0
    try:
        with open_sqlite(db_path, read_only=True) as con:
            row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
