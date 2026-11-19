"""Optional local/Zenodo DadaGP & Guitar Pro SQLite catalog and archive extractor."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tarfile
import urllib.error
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from transcribe_music.ingest._io import (
    atomic_write_bytes,
    copy_file_safe,
    download_https_stream,
    http_get_json,
    is_zip_symlink,
    normalize_ascii,
    read_bounded_tar_member,
    read_bounded_zip_member,
    resolve_catalog_dir,
    safe_resolve_member_path,
    sanitize_relative_member,
    save_catalog_dir,
    tokenize_ascii_query,
)
from transcribe_music.storage import atomic_sqlite, count_sqlite_rows, open_sqlite

logger = logging.getLogger(__name__)

GP_CATALOG_EXTENSIONS: frozenset[str] = frozenset(
    {".gp3", ".gp4", ".gp5", ".gp", ".gpx", ".tab", ".mxl", ".musicxml"}
)
_IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {"songs", "tabs", "scores", "guitarpro", "gp", "data", "files"}
)


@dataclass(frozen=True)
class GpCatalogStatus:
    """Status summary of the local Guitar Pro / DadaGP catalog."""

    gp_dir: Path
    db_exists: bool
    score_count: int
    archive_available: bool


def resolve_gp_catalog_dir(cache_dir: Path = Path(".transcribe_cache")) -> Path:
    """Resolve active Guitar Pro catalog dir (custom configured dir or `<cache_dir>/dadagp`)."""
    return resolve_catalog_dir(cache_dir, "dadagp", "gp_dir")


def _has_gp_sources(gp_dir: Path) -> bool:
    if not gp_dir.exists():
        return False
    for p in gp_dir.iterdir():
        name_l = p.name.lower()
        if p.is_file() and (
            name_l.endswith((".tar.gz", ".tgz", ".zip")) or p.suffix.lower() in GP_CATALOG_EXTENSIONS
        ):
            return True
        if p.is_dir():
            return True
    return False


def get_gp_catalog_status(cache_dir: Path = Path(".transcribe_cache")) -> GpCatalogStatus:
    """Inspect local Guitar Pro / DadaGP SQLite index and archive availability."""
    gp_dir = resolve_gp_catalog_dir(cache_dir)
    db_path = gp_dir / "gp_catalog.sqlite"
    return GpCatalogStatus(
        gp_dir=gp_dir,
        db_exists=db_path.exists(),
        score_count=count_sqlite_rows(db_path, "gp_scores"),
        archive_available=_has_gp_sources(gp_dir),
    )


def _infer_entry_metadata(rel_path: str) -> tuple[str, str | None, str | None]:
    """Derive (title, artist, genre_or_group) from a relative Guitar Pro path."""
    p = Path(rel_path)
    stem = p.stem.replace("_", " ").strip()
    meaningful_dirs = [
        part.replace("_", " ").strip()
        for part in p.parts[:-1]
        if part
        and part != "."
        and len(part) > 1
        and part.lower() not in _IGNORED_DIR_NAMES
        and not part.lower().startswith("dadagp")
    ]

    artist: str | None = None
    genre: str | None = None
    title = stem

    if " - " in stem:
        left, right = (seg.strip() for seg in stem.split(" - ", 1))
        if left and right:
            artist = left
            title = right

    if artist is None and meaningful_dirs:
        artist = meaningful_dirs[-1]
        if len(meaningful_dirs) >= 2:
            genre = meaningful_dirs[0]
    elif len(meaningful_dirs) >= 1 and meaningful_dirs[-1].lower() != (artist or "").lower():
        genre = meaningful_dirs[-1]

    return title or p.stem, artist, genre


def _safe_archive_member_name(name: str) -> str | None:
    try:
        return sanitize_relative_member(name.replace("\\", "/"))
    except ValueError:
        return None


def _make_gp_row(
    parts: tuple[str, ...],
    rel_member: str,
    ext: str,
) -> tuple[str, str, str | None, str | None, str, str]:
    title, artist, genre = _infer_entry_metadata(rel_member)
    search_text = normalize_ascii(f"{title} {artist or ''} {genre or ''} {Path(rel_member).stem}")
    return (json.dumps(parts), ext, artist, genre, title, search_text)


def _collect_gp_entries(target_dir: Path) -> list[tuple[str, str, str | None, str | None, str, str]]:
    """Scan local files, `.tar.gz`, and `.zip` archives inside `target_dir` for GP scores."""
    rows: list[tuple[str, str, str | None, str | None, str, str]] = []
    base_resolved = target_dir.resolve()

    for path in sorted(target_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(base_resolved):
            continue
        rel = path.relative_to(target_dir).as_posix()
        ext = path.suffix.lower()
        name_l = path.name.lower()

        if ext in GP_CATALOG_EXTENSIONS:
            rows.append(_make_gp_row(("file", rel), rel, ext))
        elif name_l.endswith((".tar.gz", ".tgz")):
            try:
                with tarfile.open(path, "r:gz") as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        if (clean_m := _safe_archive_member_name(member.name)) is None:
                            continue
                        m_ext = Path(clean_m).suffix.lower()
                        if m_ext in GP_CATALOG_EXTENSIONS:
                            rows.append(_make_gp_row(("tar", rel, clean_m), clean_m, m_ext))
            except (tarfile.TarError, OSError) as exc:
                logger.debug("Skipping unreadable tar archive %s: %s", path, exc)
        elif ext == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir() or is_zip_symlink(info):
                            continue
                        if (clean_m := _safe_archive_member_name(info.filename)) is None:
                            continue
                        m_ext = Path(clean_m).suffix.lower()
                        if m_ext in GP_CATALOG_EXTENSIONS:
                            rows.append(_make_gp_row(("zip", rel, clean_m), clean_m, m_ext))
            except (zipfile.BadZipFile, OSError) as exc:
                logger.debug("Skipping unreadable zip archive %s: %s", path, exc)

    return rows


def _is_zenodo_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "zenodo.org" or host.endswith(".zenodo.org")


def _download_zenodo_record_archives(
    record_id: str,
    target_dir: Path,
    *,
    token: str | None,
    force: bool,
) -> None:
    clean_id = record_id.strip()
    if not clean_id.isdigit():
        raise ValueError(f"Zenodo record ID must be numeric, got: {record_id!r}")

    api_url = f"https://zenodo.org/api/records/{clean_id}"
    auth_headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        data = http_get_json(api_url, headers=auth_headers, timeout=20.0)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                f"Zenodo record {clean_id} requires approved researcher access (HTTP {exc.code}). "
                "Set ZENODO_TOKEN after access approval, or index local tabs with --gp-dir."
            ) from exc
        raise
    files = data.get("files", []) if isinstance(data, dict) else []

    downloaded = 0
    for entry in files:
        key = str(entry.get("key") or entry.get("filename") or "").strip()
        if not key or not key.lower().endswith((".tar.gz", ".tgz", ".zip")):
            continue
        links = entry.get("links") or {}
        dl_url = str(links.get("self") or links.get("download") or "").strip()
        if not dl_url:
            continue
        _, clean_key, dest = safe_resolve_member_path(target_dir, key)
        if force or not dest.exists():
            dl_headers = auth_headers if _is_zenodo_host(dl_url) else None
            download_https_stream(dl_url, dest, headers=dl_headers)
        downloaded += 1
        logger.debug("Synced Zenodo GP archive: %s", clean_key)

    if downloaded == 0:
        raise RuntimeError(
            f"No .tar.gz or .zip archives found in Zenodo record {clean_id}. "
            "Supply local Guitar Pro files or archives via --gp-dir."
        )


def build_gp_catalog_sqlite(gp_dir: Path, db_path: Path) -> int:
    """Scan `gp_dir` and build an indexed `gp_catalog.sqlite` database."""
    gp_dir = Path(gp_dir)
    if not gp_dir.exists():
        raise FileNotFoundError(f"Guitar Pro catalog directory not found: {gp_dir}")

    entries = _collect_gp_entries(gp_dir)
    with atomic_sqlite(Path(db_path)) as con:
        con.execute(
            """
            CREATE TABLE gp_scores (
                member_ref TEXT PRIMARY KEY,
                format_ext TEXT NOT NULL,
                artist TEXT,
                genre TEXT,
                title TEXT NOT NULL,
                search_text TEXT NOT NULL
            )
            """
        )
        if entries:
            con.executemany(
                "INSERT OR REPLACE INTO gp_scores (member_ref, format_ext, artist, genre, title, search_text) VALUES (?, ?, ?, ?, ?, ?)",
                entries,
            )
        row = con.execute("SELECT COUNT(*) FROM gp_scores").fetchone()
        return int(row[0]) if row else 0


def sync_gp_catalog(
    cache_dir: Path = Path(".transcribe_cache"),
    *,
    gp_dir: Path | None = None,
    zenodo_record: str | None = None,
    zenodo_token: str | None = None,
    force: bool = False,
) -> GpCatalogStatus:
    """Index a local Guitar Pro / DadaGP folder/archive (or download a Zenodo record) into SQLite."""
    target_dir = Path(gp_dir) if gp_dir is not None else (Path(cache_dir) / "dadagp")
    target_dir.mkdir(parents=True, exist_ok=True)
    if gp_dir is not None:
        save_catalog_dir(cache_dir, "dadagp", "gp_dir", target_dir)

    if zenodo_record:
        resolved_token = (zenodo_token or os.environ.get("ZENODO_TOKEN") or "").strip() or None
        _download_zenodo_record_archives(
            zenodo_record,
            target_dir,
            token=resolved_token,
            force=force,
        )

    build_gp_catalog_sqlite(target_dir, target_dir / "gp_catalog.sqlite")
    return get_gp_catalog_status(cache_dir)


def search_gp_catalog(
    query: str,
    *,
    cache_dir: Path = Path(".transcribe_cache"),
    allowed_exts: frozenset[str],
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Query the local SQLite DadaGP / Guitar Pro catalog if synced; returns `[]` otherwise."""
    gp_dir = resolve_gp_catalog_dir(cache_dir)
    db_path = gp_dir / "gp_catalog.sqlite"
    if not db_path.exists():
        return []

    tokens = tokenize_ascii_query(query)
    if not tokens:
        return []

    target_exts = sorted(allowed_exts & GP_CATALOG_EXTENSIONS)
    if not target_exts:
        return []

    where_clauses = ["INSTR(search_text, ?) > 0"] * len(tokens)
    ext_placeholders = ", ".join("?" for _ in target_exts)
    sql = f"""
        SELECT member_ref, format_ext, artist, genre, title
        FROM gp_scores
        WHERE {" AND ".join(where_clauses)}
          AND format_ext IN ({ext_placeholders})
        LIMIT ?
    """
    try:
        with open_sqlite(db_path, read_only=True) as con:
            rows = con.execute(sql, [*tokens, *target_exts, limit]).fetchall()
    except sqlite3.Error as exc:
        logger.debug("GP SQLite query failed: %s", exc)
        return []

    records: list[dict[str, Any]] = []
    for member_ref, fmt_ext, artist, genre, title in rows:
        note_bits = ["Guitar TAB"]
        if genre:
            note_bits.append(str(genre))
        member_part = _decode_member_ref(str(member_ref))[-1]
        records.append(
            {
                "title": str(title),
                "composer": str(artist) if artist else None,
                "format_ext": str(fmt_ext),
                "source": "DadaGP",
                "filename": f"{Path(member_part).stem}{fmt_ext}",
                "edition_note": " · ".join(note_bits),
                "gp_dir": str(gp_dir),
                "gp_member": str(member_ref),
            }
        )
    return records


def _decode_member_ref(member_ref: str) -> list[str]:
    if member_ref.startswith("["):
        try:
            parsed = json.loads(member_ref)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Guitar Pro member reference: {member_ref!r}") from exc
        if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
            return parsed
        raise ValueError(f"Invalid Guitar Pro member reference: {member_ref!r}")
    return member_ref.split("::")


def extract_gp_catalog_member(gp_dir: Path | str, member_ref: str, dest: Path) -> Path:
    """Safely extract a Guitar Pro / tab file from a catalog dir, `.tar.gz`, or `.zip`."""
    dest = Path(dest)
    parts = _decode_member_ref(member_ref)
    kind = parts[0]
    if kind == "file" and len(parts) == 2:
        _, _, src = safe_resolve_member_path(gp_dir, parts[1])
        if not src.exists():
            raise FileNotFoundError(f"Guitar Pro file not found in catalog: {src}")
        return copy_file_safe(src, dest)

    if kind in {"tar", "zip"} and len(parts) == 3:
        _, clean_archive, archive_path = safe_resolve_member_path(gp_dir, parts[1])
        _, clean_member, _ = safe_resolve_member_path(gp_dir, parts[2])
        if not archive_path.exists():
            raise FileNotFoundError(f"Guitar Pro archive '{clean_archive}' not found in {gp_dir}")

        if kind == "tar":
            data = read_bounded_tar_member(archive_path, clean_member)
        else:
            targets = {clean_member, f"./{clean_member}"}
            data = read_bounded_zip_member(archive_path, lambda name: name in targets)
            if data is None:
                raise FileNotFoundError(f"Member '{clean_member}' not found inside {archive_path}")
        return atomic_write_bytes(dest, data)

    raise ValueError(f"Invalid Guitar Pro catalog member reference: {member_ref!r}")
