"""Embedded DuckDB catalog for content-addressed score job caching."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from transcribe_music.models import (
    Instrument,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_hash VARCHAR PRIMARY KEY,
    source_type VARCHAR NOT NULL,
    source_ref VARCHAR NOT NULL,
    instrument VARCHAR NOT NULL,
    config_json VARCHAR NOT NULL,
    musicxml_path VARCHAR NOT NULL,
    measure_count INTEGER NOT NULL,
    note_count INTEGER NOT NULL,
    chord_count INTEGER NOT NULL,
    fingering_count INTEGER NOT NULL,
    backend VARCHAR,
    selected_part VARCHAR,
    elapsed_sec DOUBLE NOT NULL,
    warnings_json VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
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
        "pipeline_version": "v2",
        "source_type": source_type.value,
        "source_identity": source_identity,
        "instrument": config.instrument.value,
        "part": config.part,
        "enrich_chords": config.enrich_chords,
        "enrich_fingering": config.enrich_fingering,
        "preserve_existing_chords": config.preserve_existing_chords,
        "preserve_existing_fingering": config.preserve_existing_fingering,
        "quantize_divisors": list(config.quantize_divisors),
        "pdf_dpi": config.pdf_dpi if source_type == SourceType.PDF else None,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


class CatalogStore:
    """Single-file DuckDB catalog for caching MusicXML enrichment jobs."""

    def __init__(self, cache_dir: Path = Path(".transcribe_cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.cache_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "catalog.duckdb"
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(_SCHEMA_SQL)
            con.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS backend VARCHAR")
            con.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS selected_part VARCHAR")

    def artifact_dir(self, job_hash: str) -> Path:
        path = self.artifacts_dir / job_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cached_job(self, job_hash: str) -> TranscriptionResult | None:
        with self._connect() as con:
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

        if row is None:
            return None

        musicxml_path = Path(row[4])
        if not musicxml_path.exists():
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
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._connect() as con:
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
                    config.model_dump_json(),
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
        with self._connect() as con:
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
        with self._connect() as con:
            count_row = con.execute("SELECT COUNT(*) FROM jobs").fetchone()
            count = int(count_row[0]) if count_row else 0
            con.execute("DELETE FROM jobs")
        return count
