"""Embedded SQLite catalog for content-addressed score job caching."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transcribe_music.models import (
    Instrument,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.storage import open_sqlite

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_hash TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    instrument TEXT NOT NULL,
    config_json TEXT NOT NULL,
    musicxml_path TEXT NOT NULL,
    measure_count INTEGER NOT NULL,
    note_count INTEGER NOT NULL,
    chord_count INTEGER NOT NULL,
    fingering_count INTEGER NOT NULL,
    backend TEXT,
    selected_part TEXT,
    elapsed_sec REAL NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def hash_file(path: Path) -> str:
    """Compute SHA-256 digest of a local file in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compute_job_hash(
    source_type: SourceType,
    source_identity: str,
    config: TranscriptionConfig,
) -> str:
    """Compute a deterministic cache key for a score conversion/OMR job."""
    payload = {
        "pipeline_version": "v6",
        "source_type": source_type.value,
        "source_identity": source_identity,
        "instrument": config.instrument.value,
        "part": config.part,
        "title": config.title,
        "composer": config.composer,
        "enrich_chords": config.enrich_chords,
        "chord_window_beats": config.chord_window_beats if config.enrich_chords else None,
        "include_slash_chords": config.include_slash_chords if config.enrich_chords else None,
        "roman_numerals": config.roman_numerals if config.enrich_chords else None,
        "enrich_fingering": config.enrich_fingering,
        "preserve_existing_chords": config.preserve_existing_chords,
        "preserve_existing_fingering": config.preserve_existing_fingering,
        "preserve_existing_tab": config.preserve_existing_tab if config.instrument == Instrument.GUITAR else None,
        "guitar_staff": config.guitar_staff.value if config.instrument == Instrument.GUITAR else None,
        "quantize_divisors": list(config.quantize_divisors),
        "pdf_dpi": config.pdf_dpi if source_type == SourceType.PDF else None,
        "omr_backend": config.omr_backend.value if source_type == SourceType.PDF else None,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


class CatalogStore:
    """Single-file SQLite catalog for caching MusicXML enrichment jobs."""

    def __init__(self, cache_dir: Path = Path(".transcribe_cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.artifacts_dir = self.cache_dir / "artifacts"
        self.db_path = self.cache_dir / "catalog.sqlite"

    def artifact_dir(self, job_hash: str) -> Path:
        path = self.artifacts_dir / job_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cached_job(self, job_hash: str) -> TranscriptionResult | None:
        if not self.db_path.exists():
            return None
        try:
            with open_sqlite(self.db_path, read_only=True) as con:
                row = con.execute(
                    """
                    SELECT job_hash, source_type, source_ref, instrument, musicxml_path,
                           measure_count, note_count, chord_count, fingering_count,
                           elapsed_sec, warnings_json, backend, selected_part
                    FROM jobs
                    WHERE job_hash = ?
                    """,
                    [job_hash],
                ).fetchone()
        except sqlite3.Error:
            return None

        if row is None:
            return None

        musicxml_path = Path(row[4])
        artifacts_root = self.artifacts_dir.resolve()
        try:
            resolved_out = musicxml_path.resolve()
        except OSError:
            return None
        if not resolved_out.is_relative_to(artifacts_root) or not resolved_out.is_file():
            return None

        return TranscriptionResult(
            job_hash=row[0],
            source_type=SourceType(row[1]),
            source_ref=row[2],
            instrument=Instrument(row[3]),
            musicxml_path=musicxml_path,
            cached=True,
            elapsed_sec=float(row[9]),
            measure_count=int(row[5]),
            note_count=int(row[6]),
            chord_count=int(row[7]),
            fingering_count=int(row[8]),
            backend=row[11],
            selected_part=row[12],
            warnings=json.loads(row[10]),
        )

    def record_job(
        self,
        result: TranscriptionResult,
        config: TranscriptionConfig,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
        with open_sqlite(self.db_path, wal=True) as con:
            con.execute(_SCHEMA_SQL)
            con.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    job_hash, source_type, source_ref, instrument,
                    config_json, musicxml_path, measure_count, note_count, chord_count,
                    fingering_count, backend, selected_part, elapsed_sec, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.job_hash,
                    result.source_type.value,
                    result.source_ref,
                    result.instrument.value,
                    json.dumps(asdict(config), default=str),
                    str(result.musicxml_path),
                    result.measure_count,
                    result.note_count,
                    result.chord_count,
                    result.fingering_count,
                    result.backend,
                    result.selected_part,
                    result.elapsed_sec,
                    json.dumps(result.warnings),
                    now,
                ],
            )

    def list_recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            with open_sqlite(self.db_path, read_only=True) as con:
                rows = con.execute(
                    """
                    SELECT job_hash, source_type, source_ref, instrument,
                           measure_count, note_count, chord_count, fingering_count,
                           backend, selected_part, elapsed_sec, musicxml_path, created_at
                    FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
        except sqlite3.Error:
            return []
        cols = [
            "job_hash",
            "source_type",
            "source_ref",
            "instrument",
            "measure_count",
            "note_count",
            "chord_count",
            "fingering_count",
            "backend",
            "selected_part",
            "elapsed_sec",
            "musicxml_path",
            "created_at",
        ]
        return [dict(zip(cols, row, strict=True)) for row in rows]

    def clear_cache(self) -> int:
        count = 0
        if self.db_path.exists():
            try:
                with open_sqlite(self.db_path) as con:
                    count_row = con.execute("SELECT COUNT(*) FROM jobs").fetchone()
                    count = int(count_row[0]) if count_row else 0
                    con.execute("DELETE FROM jobs")
            except sqlite3.Error:
                pass
        if self.artifacts_dir.exists():
            shutil.rmtree(self.artifacts_dir, ignore_errors=True)
        return count
