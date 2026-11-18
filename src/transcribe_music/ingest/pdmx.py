"""Optional local PDMX (Public Domain MusicXML, Zenodo 15571083) DuckDB catalog & archive extractor."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tarfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import requests

logger = logging.getLogger(__name__)

PDMX_ZENODO_RECORD_ID = "15571083"
PDMX_CSV_URL = f"https://zenodo.org/api/records/{PDMX_ZENODO_RECORD_ID}/files/PDMX.csv/content"
PDMX_MXL_TAR_URL = f"https://zenodo.org/api/records/{PDMX_ZENODO_RECORD_ID}/files/mxl.tar.gz/content"
PDMX_MID_TAR_URL = f"https://zenodo.org/api/records/{PDMX_ZENODO_RECORD_ID}/files/mid.tar.gz/content"

_USER_AGENT = "transcribe-music/0.2 (+https://github.com/transcribe-music)"


@dataclass(frozen=True)
class PdmxStatus:
    """Status summary of local PDMX catalog and score archives."""

    pdmx_dir: Path
    csv_exists: bool
    db_exists: bool
    score_count: int
    mxl_available: bool
    mid_available: bool


def _config_file(cache_dir: Path) -> Path:
    return Path(cache_dir) / "pdmx" / "config.json"


def resolve_pdmx_dir(cache_dir: Path = Path(".transcribe_cache")) -> Path:
    """Resolve the active PDMX directory (custom configured dir or `<cache_dir>/pdmx`)."""
    default_dir = Path(cache_dir) / "pdmx"
    cfg_path = _config_file(cache_dir)
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            custom = data.get("pdmx_dir")
            if isinstance(custom, str) and custom.strip():
                return Path(custom)
        except (OSError, ValueError):
            pass
    return default_dir


def _save_pdmx_dir_config(cache_dir: Path, pdmx_dir: Path) -> None:
    cfg_path = _config_file(cache_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"pdmx_dir": str(pdmx_dir.resolve())}), encoding="utf-8")


def _pdmx_archive_available(pdmx_dir: Path, kind: str) -> bool:
    return (pdmx_dir / f"{kind}.tar.gz").exists() or (pdmx_dir / kind).is_dir()


def get_pdmx_status(cache_dir: Path = Path(".transcribe_cache")) -> PdmxStatus:
    """Inspect local PDMX CSV, DuckDB index, and MXL/MIDI archive availability."""
    pdmx_dir = resolve_pdmx_dir(cache_dir)
    csv_path = pdmx_dir / "PDMX.csv"
    db_path = pdmx_dir / "pdmx.duckdb"
    count = 0
    if db_path.exists():
        try:
            with duckdb.connect(str(db_path), read_only=True) as con:
                row = con.execute("SELECT COUNT(*) FROM pdmx_scores").fetchone()
                count = int(row[0]) if row else 0
        except duckdb.Error:
            count = 0
    return PdmxStatus(
        pdmx_dir=pdmx_dir,
        csv_exists=csv_path.exists(),
        db_exists=db_path.exists(),
        score_count=count,
        mxl_available=_pdmx_archive_available(pdmx_dir, "mxl"),
        mid_available=_pdmx_archive_available(pdmx_dir, "mid"),
    )


def _normalize_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(ascii_only.lower().split())


def build_pdmx_duckdb(
    csv_path: Path,
    db_path: Path,
    *,
    include_license_conflicts: bool = False,
) -> int:
    """Ingest `PDMX.csv` into an indexed DuckDB database for fast offline search."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PDMX.csv not found at {csv_path}")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = db_path.with_suffix(".duckdb.tmp")
    if tmp_db.exists():
        tmp_db.unlink()

    conflict_filter = "" if include_license_conflicts else "AND TRY_CAST(\"subset:no_license_conflict\" AS BOOLEAN) IS TRUE"

    with duckdb.connect(str(tmp_db)) as con:
        con.execute(
            f"""
            CREATE TABLE pdmx_scores AS
            SELECT
                NULLIF(TRIM(mxl), '') AS mxl_rel,
                NULLIF(TRIM(mid), '') AS mid_rel,
                NULLIF(TRIM(pdf), '') AS pdf_rel,
                COALESCE(NULLIF(TRIM(title), ''), NULLIF(TRIM(song_name), ''), 'Untitled') AS title,
                NULLIF(TRIM(song_name), '') AS song_name,
                COALESCE(NULLIF(TRIM(composer_name), ''), NULLIF(TRIM(artist_name), '')) AS composer,
                NULLIF(TRIM(genres), '') AS genres,
                TRY_CAST(n_tracks AS INTEGER) AS n_tracks,
                TRY_CAST(complexity AS DOUBLE) AS complexity,
                TRY_CAST(rating AS DOUBLE) AS rating,
                TRY_CAST(n_ratings AS INTEGER) AS n_ratings,
                COALESCE(TRY_CAST(is_best_unique_arrangement AS BOOLEAN), FALSE) AS is_best_unique,
                COALESCE(TRY_CAST("subset:no_license_conflict" AS BOOLEAN), FALSE) AS no_license_conflict,
                LOWER(
                    STRIP_ACCENTS(
                        CONCAT_WS(
                            ' ',
                            COALESCE(title, ''),
                            COALESCE(song_name, ''),
                            COALESCE(composer_name, ''),
                            COALESCE(artist_name, ''),
                            COALESCE(genres, ''),
                            COALESCE(tags, '')
                        )
                    )
                ) AS search_text
            FROM read_csv(
                ?,
                header = TRUE,
                nullstr = ['NA', ''],
                ignore_errors = TRUE
            )
            WHERE (mxl IS NOT NULL OR mid IS NOT NULL)
            {conflict_filter}
            """,
            [str(csv_path)],
        )
        row = con.execute("SELECT COUNT(*) FROM pdmx_scores").fetchone()
        count = int(row[0]) if row else 0

    tmp_db.replace(db_path)
    return count


def _download_stream(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f"{dest.suffix}.tmp")
    with requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        stream=True,
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    return dest


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
    """Download/register PDMX metadata & optional score archives and build `pdmx.duckdb`."""
    target_dir = Path(pdmx_dir) if pdmx_dir is not None else (Path(cache_dir) / "pdmx")
    target_dir.mkdir(parents=True, exist_ok=True)
    if pdmx_dir is not None:
        _save_pdmx_dir_config(cache_dir, target_dir)

    csv_path = target_dir / "PDMX.csv"
    if download_csv and (force or not csv_path.exists()):
        _download_stream(PDMX_CSV_URL, csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PDMX.csv not found in {target_dir}. Run without --no-download or supply PDMX.csv."
        )

    if download_mxl:
        mxl_tar = target_dir / "mxl.tar.gz"
        if force or not mxl_tar.exists():
            _download_stream(PDMX_MXL_TAR_URL, mxl_tar)
        if extract_archives and mxl_tar.exists():
            _extract_full_archive(mxl_tar, target_dir)

    if download_mid:
        mid_tar = target_dir / "mid.tar.gz"
        if force or not mid_tar.exists():
            _download_stream(PDMX_MID_TAR_URL, mid_tar)
        if extract_archives and mid_tar.exists():
            _extract_full_archive(mid_tar, target_dir)

    db_path = target_dir / "pdmx.duckdb"
    build_pdmx_duckdb(
        csv_path,
        db_path,
        include_license_conflicts=include_license_conflicts,
    )
    return get_pdmx_status(cache_dir)


def search_pdmx_catalog(
    query: str,
    *,
    cache_dir: Path = Path(".transcribe_cache"),
    allowed_exts: frozenset[str],
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Query the local DuckDB PDMX catalog if synced; returns `[]` immediately otherwise."""
    pdmx_dir = resolve_pdmx_dir(cache_dir)
    db_path = pdmx_dir / "pdmx.duckdb"
    if not db_path.exists():
        return []

    tokens = [tok for tok in _normalize_ascii(query).split() if tok]
    if not tokens:
        return []

    want_mxl = ".mxl" in allowed_exts
    want_mid = ".mid" in allowed_exts
    if not (want_mxl or want_mid):
        return []

    mxl_ready = _pdmx_archive_available(pdmx_dir, "mxl")
    mid_ready = _pdmx_archive_available(pdmx_dir, "mid")

    where_clauses = ["search_text LIKE '%' || ? || '%'"] * len(tokens)
    sql = f"""
        SELECT
            mxl_rel,
            mid_rel,
            title,
            composer,
            genres,
            n_tracks,
            rating,
            is_best_unique
        FROM pdmx_scores
        WHERE {" AND ".join(where_clauses)}
        ORDER BY
            is_best_unique DESC,
            rating DESC NULLS LAST,
            n_ratings DESC NULLS LAST
        LIMIT ?
    """
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            rows = con.execute(sql, [*tokens, limit]).fetchall()
    except duckdb.Error as exc:
        logger.debug("PDMX DuckDB query failed: %s", exc)
        return []

    records: list[dict[str, Any]] = []
    for mxl_rel, mid_rel, title, composer, genres, n_tracks, rating, is_best_unique in rows:
        candidates: list[tuple[str, str | None, bool]] = []
        if want_mxl and mxl_rel:
            candidates.append((".mxl", str(mxl_rel), mxl_ready))
        if want_mid and mid_rel:
            candidates.append((".mid", str(mid_rel), mid_ready))

        for ext, rel_path, archive_ready in candidates:
            if ext == ".mxl" and not mxl_ready and mid_ready and want_mid and mid_rel:
                continue
            note_bits = []
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
            if not archive_ready:
                bonus -= 2.5

            clean_rel = str(rel_path).lstrip("./")
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
    """Extract a single `.mxl` or `.mid` score from an extracted PDMX dir or `.tar.gz` archive."""
    base_dir = Path(pdmx_dir)
    clean_rel = member_rel.lstrip("./")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    direct_file = base_dir / clean_rel
    if direct_file.exists():
        shutil.copy2(direct_file, dest)
        return dest

    top_folder = clean_rel.split("/", 1)[0]
    tar_path = base_dir / f"{top_folder}.tar.gz"
    if not tar_path.exists():
        flag = "--with-mxl" if top_folder == "mxl" else "--with-mid"
        raise FileNotFoundError(
            f"PDMX score '{clean_rel}' is in catalog metadata, but '{tar_path.name}' is not downloaded yet. "
            f"Run: transcribe-music pdmx sync {flag}"
        )

    # Fast path: extract via system `tar` CLI (libarchive C decoder) and cache locally
    tar_bin = shutil.which("tar")
    if tar_bin is not None:
        for candidate_name in (clean_rel, f"./{clean_rel}"):
            proc = subprocess.run(
                [tar_bin, "-xzf", str(tar_path), "-O", candidate_name],
                check=False,
                capture_output=True,
                timeout=120,
            )
            if proc.returncode == 0 and proc.stdout:
                direct_file.parent.mkdir(parents=True, exist_ok=True)
                direct_file.write_bytes(proc.stdout)
                dest.write_bytes(proc.stdout)
                return dest

    with tarfile.open(tar_path, "r:gz") as tf:
        for candidate_name in (clean_rel, f"./{clean_rel}"):
            try:
                member = tf.getmember(candidate_name)
            except KeyError:
                continue
            extracted = tf.extractfile(member)
            if extracted is not None:
                data = extracted.read()
                direct_file.parent.mkdir(parents=True, exist_ok=True)
                direct_file.write_bytes(data)
                dest.write_bytes(data)
                return dest

    raise FileNotFoundError(f"Member '{clean_rel}' not found inside {tar_path}")
