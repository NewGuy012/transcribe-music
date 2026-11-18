"""Optional local PDMX (Public Domain MusicXML, Zenodo 15571083) SQLite catalog & archive extractor."""

from __future__ import annotations

import csv
import logging
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcribe_music.ingest._io import (
    download_https_stream,
    normalize_ascii,
    read_bounded_tar_member,
    resolve_catalog_dir,
    safe_resolve_member_path,
    save_catalog_dir,
)
from transcribe_music.models import Instrument
from transcribe_music.storage import open_sqlite

logger = logging.getLogger(__name__)

PDMX_ZENODO_RECORD_ID = "15571083"
PDMX_CSV_URL = f"https://zenodo.org/api/records/{PDMX_ZENODO_RECORD_ID}/files/PDMX.csv/content"
PDMX_MXL_TAR_URL = f"https://zenodo.org/api/records/{PDMX_ZENODO_RECORD_ID}/files/mxl.tar.gz/content"
PDMX_MID_TAR_URL = f"https://zenodo.org/api/records/{PDMX_ZENODO_RECORD_ID}/files/mid.tar.gz/content"


@dataclass(frozen=True)
class PdmxStatus:
    """Status summary of local PDMX catalog and score archives."""

    pdmx_dir: Path
    csv_exists: bool
    db_exists: bool
    score_count: int
    mxl_available: bool
    mid_available: bool


def resolve_pdmx_dir(cache_dir: Path = Path(".transcribe_cache")) -> Path:
    """Resolve the active PDMX directory (custom configured dir or `<cache_dir>/pdmx`)."""
    return resolve_catalog_dir(cache_dir, "pdmx", "pdmx_dir")


def _pdmx_archive_available(pdmx_dir: Path, kind: str) -> bool:
    return (pdmx_dir / f"{kind}.tar.gz").exists() or (pdmx_dir / kind).is_dir()


def get_pdmx_status(cache_dir: Path = Path(".transcribe_cache")) -> PdmxStatus:
    """Inspect local PDMX CSV, SQLite index, and MXL/MIDI archive availability."""
    pdmx_dir = resolve_pdmx_dir(cache_dir)
    csv_path = pdmx_dir / "PDMX.csv"
    db_path = pdmx_dir / "pdmx.sqlite"
    count = 0
    if db_path.exists():
        try:
            with open_sqlite(db_path, read_only=True) as con:
                row = con.execute("SELECT COUNT(*) FROM pdmx_scores").fetchone()
                count = int(row[0]) if row else 0
        except sqlite3.Error:
            count = 0
    return PdmxStatus(
        pdmx_dir=pdmx_dir,
        csv_exists=csv_path.exists(),
        db_exists=db_path.exists(),
        score_count=count,
        mxl_available=_pdmx_archive_available(pdmx_dir, "mxl"),
        mid_available=_pdmx_archive_available(pdmx_dir, "mid"),
    )


def _csv_str(row: dict[str, str | None], key: str) -> str | None:
    raw = (row.get(key) or "").strip()
    return None if not raw or raw == "NA" else raw


def _csv_int(row: dict[str, str | None], key: str) -> int | None:
    raw = _csv_str(row, key)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _csv_float(row: dict[str, str | None], key: str) -> float | None:
    raw = _csv_str(row, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_pdmx_sqlite(
    csv_path: Path,
    db_path: Path,
    *,
    include_license_conflicts: bool = False,
) -> int:
    """Ingest `PDMX.csv` into an indexed SQLite database for fast offline search."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PDMX.csv not found at {csv_path}")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = db_path.with_suffix(".sqlite.tmp")
    tmp_db.unlink(missing_ok=True)

    count = 0
    csv.field_size_limit(max(csv.field_size_limit(), 1024 * 1024))
    try:
        with open_sqlite(tmp_db) as con:
            con.execute("PRAGMA journal_mode=OFF")
            con.execute("PRAGMA synchronous=OFF")
            con.execute(
                """
                CREATE TABLE pdmx_scores (
                    mxl_rel TEXT,
                    mid_rel TEXT,
                    title TEXT NOT NULL,
                    composer TEXT,
                    genres TEXT,
                    tracks TEXT,
                    tags TEXT,
                    n_tracks INTEGER,
                    rating REAL,
                    n_ratings INTEGER,
                    is_best_unique INTEGER NOT NULL,
                    search_text TEXT NOT NULL
                )
                """
            )
            insert_sql = """
                INSERT INTO pdmx_scores (
                    mxl_rel, mid_rel, title, composer, genres, tracks, tags,
                    n_tracks, rating, n_ratings, is_best_unique, search_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            batch: list[tuple[Any, ...]] = []
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                for row in csv.DictReader(fh):
                    mxl_rel = _csv_str(row, "mxl")
                    mid_rel = _csv_str(row, "mid")
                    if mxl_rel is None and mid_rel is None:
                        continue
                    no_conflict = (_csv_str(row, "subset:no_license_conflict") or "").lower() in {
                        "true",
                        "1",
                    }
                    if not include_license_conflicts and not no_conflict:
                        continue

                    title = _csv_str(row, "title") or _csv_str(row, "song_name") or "Untitled"
                    composer = _csv_str(row, "composer_name") or _csv_str(row, "artist_name")
                    genres = _csv_str(row, "genres")
                    tracks = _csv_str(row, "tracks")
                    tags = _csv_str(row, "tags")
                    is_best = int(
                        (_csv_str(row, "is_best_unique_arrangement") or "").lower() in {"true", "1"}
                    )
                    search_parts = [
                        _csv_str(row, k)
                        for k in (
                            "title",
                            "song_name",
                            "composer_name",
                            "artist_name",
                            "genres",
                            "tracks",
                            "tags",
                        )
                    ]
                    search_text = normalize_ascii(" ".join(p for p in search_parts if p))
                    batch.append(
                        (
                            mxl_rel,
                            mid_rel,
                            title,
                            composer,
                            genres,
                            tracks,
                            tags,
                            _csv_int(row, "n_tracks"),
                            _csv_float(row, "rating"),
                            _csv_int(row, "n_ratings"),
                            is_best,
                            search_text,
                        )
                    )
                    if len(batch) >= 5000:
                        con.executemany(insert_sql, batch)
                        count += len(batch)
                        batch.clear()
                if batch:
                    con.executemany(insert_sql, batch)
                    count += len(batch)
        tmp_db.replace(db_path)
    finally:
        tmp_db.unlink(missing_ok=True)
    return count


def _extract_full_archive(archive_path: Path, dest_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")


def sync_pdmx(
    cache_dir: Path = Path(".transcribe_cache"),
    *,
    pdmx_dir: Path | None = None,
    download_csv: bool = True,
    download_mxl: bool = False,
    download_mid: bool = False,
    extract_archives: bool = False,
    include_license_conflicts: bool = False,
    force: bool = False,
) -> PdmxStatus:
    """Download/register PDMX metadata & optional score archives and build `pdmx.sqlite`."""
    target_dir = Path(pdmx_dir) if pdmx_dir is not None else (Path(cache_dir) / "pdmx")
    target_dir.mkdir(parents=True, exist_ok=True)
    if pdmx_dir is not None:
        save_catalog_dir(cache_dir, "pdmx", "pdmx_dir", target_dir)

    csv_path = target_dir / "PDMX.csv"
    if download_csv and (force or not csv_path.exists()):
        download_https_stream(PDMX_CSV_URL, csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PDMX.csv not found in {target_dir}. Run without --no-download or supply PDMX.csv."
        )

    for enabled, name, url in (
        (download_mxl, "mxl.tar.gz", PDMX_MXL_TAR_URL),
        (download_mid, "mid.tar.gz", PDMX_MID_TAR_URL),
    ):
        if not enabled:
            continue
        tar_path = target_dir / name
        if force or not tar_path.exists():
            download_https_stream(url, tar_path)
        if extract_archives and tar_path.exists():
            _extract_full_archive(tar_path, target_dir)

    build_pdmx_sqlite(
        csv_path,
        target_dir / "pdmx.sqlite",
        include_license_conflicts=include_license_conflicts,
    )
    return get_pdmx_status(cache_dir)


def search_pdmx_catalog(
    query: str,
    *,
    cache_dir: Path = Path(".transcribe_cache"),
    allowed_exts: frozenset[str],
    instrument: Instrument | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Query the local SQLite PDMX catalog if synced; returns `[]` immediately otherwise."""
    pdmx_dir = resolve_pdmx_dir(cache_dir)
    db_path = pdmx_dir / "pdmx.sqlite"
    if not db_path.exists():
        return []

    tokens = [tok for tok in normalize_ascii(query).split() if tok]
    if not tokens:
        return []

    want_mxl = ".mxl" in allowed_exts
    want_mid = ".mid" in allowed_exts
    if not (want_mxl or want_mid):
        return []

    mxl_ready = _pdmx_archive_available(pdmx_dir, "mxl")
    mid_ready = _pdmx_archive_available(pdmx_dir, "mid")

    where_clauses = ["INSTR(search_text, ?) > 0"] * len(tokens)
    inst_term = instrument.value if instrument is not None else None
    try:
        with open_sqlite(db_path, read_only=True) as con:
            cols = {str(r[1]) for r in con.execute("PRAGMA table_info('pdmx_scores')").fetchall()}
            tracks_sel = "tracks" if "tracks" in cols else "NULL AS tracks"
            tags_sel = "tags" if "tags" in cols else "NULL AS tags"
            inst_order = ""
            params: list[Any] = [*tokens]
            if inst_term is not None:
                t_expr = (
                    "LOWER(COALESCE(tracks, '') || ' ' || COALESCE(tags, '') || ' ' || search_text)"
                    if ("tracks" in cols or "tags" in cols)
                    else "search_text"
                )
                inst_order = f"CASE WHEN INSTR({t_expr}, ?) > 0 THEN 1 ELSE 0 END DESC,"
                params.append(inst_term)
            params.append(limit)
            sql = f"""
                SELECT
                    mxl_rel,
                    mid_rel,
                    title,
                    composer,
                    genres,
                    n_tracks,
                    rating,
                    is_best_unique,
                    {tracks_sel},
                    {tags_sel}
                FROM pdmx_scores
                WHERE {" AND ".join(where_clauses)}
                ORDER BY
                    {inst_order}
                    is_best_unique DESC,
                    rating DESC NULLS LAST,
                    n_ratings DESC NULLS LAST
                LIMIT ?
            """
            rows = con.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        logger.debug("PDMX SQLite query failed: %s", exc)
        return []

    inst_keywords = (
        ("guitar", "gtr")
        if instrument == Instrument.GUITAR
        else ((instrument.value,) if instrument is not None else ())
    )

    records: list[dict[str, Any]] = []
    for mxl_rel, mid_rel, title, composer, genres, n_tracks, rating, is_best_unique, tracks, tags in rows:
        candidates: list[tuple[str, str | None, bool]] = []
        if want_mxl and mxl_rel:
            candidates.append((".mxl", str(mxl_rel), mxl_ready))
        if want_mid and mid_rel:
            candidates.append((".mid", str(mid_rel), mid_ready))

        inst_meta_lower = f"{tracks or ''} {tags or ''}".lower()
        has_inst_match = bool(inst_keywords and any(kw in inst_meta_lower for kw in inst_keywords))

        for ext, rel_path, archive_ready in candidates:
            if ext == ".mxl" and not mxl_ready and mid_ready and want_mid and mid_rel:
                continue
            note_bits = []
            if tracks:
                note_bits.append(str(tracks))
            if genres:
                note_bits.append(str(genres))
            if n_tracks:
                note_bits.append(f"{n_tracks} parts")
            if rating is not None:
                note_bits.append(f"★{float(rating):.1f}")
            if not archive_ready:
                flag = "--with-mxl" if ext == ".mxl" else "--with-mid"
                note_bits.append(f"needs `pdmx sync {flag}`")

            bonus = 0.3 if is_best_unique else 0.0
            if rating is not None:
                bonus += min(0.5, float(rating) / 10.0)
            if has_inst_match:
                bonus += 3.0
            if not archive_ready:
                bonus -= 2.5

            clean_rel = str(rel_path).removeprefix("./")
            records.append(
                {
                    "title": str(title),
                    "composer": str(composer) if composer else None,
                    "format_ext": ext,
                    "source": "PDMX",
                    "filename": f"{Path(clean_rel).stem}{ext}",
                    "edition_note": " · ".join(note_bits) if note_bits else None,
                    "source_bonus": bonus,
                    "pdmx_dir": str(pdmx_dir),
                    "pdmx_member": clean_rel,
                }
            )
    return records


def extract_pdmx_member(pdmx_dir: Path | str, member_rel: str, dest: Path) -> Path:
    """Safely extract a single `.mxl` or `.mid` score from a PDMX dir or `.tar.gz` archive."""
    base_resolved, clean_rel, direct_file = safe_resolve_member_path(pdmx_dir, member_rel)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if direct_file.exists():
        shutil.copy2(direct_file, dest)
        return dest

    top_folder = clean_rel.split("/", 1)[0]
    tar_path = base_resolved / f"{top_folder}.tar.gz"
    if not tar_path.exists():
        flag = "--with-mxl" if top_folder == "mxl" else "--with-mid"
        raise FileNotFoundError(
            f"PDMX score '{clean_rel}' is in catalog metadata, but '{tar_path.name}' is not downloaded yet. "
            f"Run: transcribe-music pdmx sync {flag}"
        )

    data = read_bounded_tar_member(tar_path, clean_rel)
    direct_file.parent.mkdir(parents=True, exist_ok=True)
    direct_file.write_bytes(data)
    dest.write_bytes(data)
    return dest
