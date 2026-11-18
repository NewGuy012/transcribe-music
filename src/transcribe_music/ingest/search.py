"""Search and download digital symbolic music scores across fast curated repositories."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import music21
import requests

from transcribe_music.ingest.digital import DIGITAL_EXTENSIONS
from transcribe_music.ingest.pdmx import (
    USER_AGENT,
    extract_pdmx_member,
    normalize_ascii,
    search_pdmx_catalog,
    validate_https_url,
)
from transcribe_music.models import ScoreSearchResult

logger = logging.getLogger(__name__)

SUPPORTED_SEARCH_EXTENSIONS: frozenset[str] = DIGITAL_EXTENSIONS
MUSICXML_SEARCH_EXTENSIONS: frozenset[str] = frozenset({".musicxml", ".mxl", ".xml"})

_FORMAT_PRIORITY: dict[str, float] = {
    ".musicxml": 1.00,
    ".mxl": 0.98,
    ".xml": 0.95,
    ".krn": 0.72,
    ".mei": 0.72,
    ".abc": 0.72,
    ".capx": 0.68,
    ".cap": 0.66,
    ".mscz": 0.65,
    ".mscx": 0.65,
    ".ly": 0.64,
    ".gp": 0.60,
    ".gpx": 0.60,
    ".gp5": 0.60,
    ".gp4": 0.60,
    ".gp3": 0.60,
    ".mid": 0.55,
    ".midi": 0.55,
}

_SOURCE_BIAS: dict[str, float] = {
    "OpenScore Orchestra": 1.1,
    "OpenScore Lieder": 1.0,
    "OpenScore StringQuartets": 1.0,
    "When-in-Rome": 0.95,
    "PDMX": 0.9,
    "music21 Corpus": 0.6,
}

_DEFAULT_INDEX_TTL_SEC = 86_400
_REQUEST_TIMEOUT_SEC = 7.0
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

_OPENSCORE_LIEDER_INDEX_URL = (
    "https://raw.githubusercontent.com/OpenScore/Lieder/main/data/corpus_conversion.json"
)
_OPENSCORE_LIEDER_RAW_PREFIX = "https://raw.githubusercontent.com/OpenScore/Lieder/main/"

_OPENSCORE_SQ_TSV_URL = (
    "https://raw.githubusercontent.com/OpenScore/StringQuartets/main/data/scores.tsv"
)
_OPENSCORE_SQ_RAW_PREFIX = "https://raw.githubusercontent.com/OpenScore/StringQuartets/main/scores/"

_OPENSCORE_ORCH_TREE_URL = (
    "https://api.github.com/repos/MarkGotham/Hauptstimme/git/trees/main?recursive=1"
)
_OPENSCORE_ORCH_RAW_PREFIX = "https://raw.githubusercontent.com/MarkGotham/Hauptstimme/main/"

_WHEN_IN_ROME_TREE_URL = (
    "https://api.github.com/repos/MarkGotham/When-in-Rome/git/trees/master?recursive=1"
)
_WHEN_IN_ROME_RAW_PREFIX = "https://raw.githubusercontent.com/MarkGotham/When-in-Rome/master/"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_MUSICAL_ALIASES: dict[str, str] = {
    "sonate": "sonata",
    "sinfonie": "symphony",
    "symphonie": "symphony",
    "konzert": "concerto",
    "klavier": "piano",
    "violoncello": "cello",
    "prlude": "prelude",
    "gymnopdie": "gymnopedie",
    "frdric": "frederic",
    "streichquartett": "string quartet",
}


def _normalize_text(text: str) -> str:
    tokens = _NON_ALNUM_RE.sub(" ", normalize_ascii(text)).split()
    return " ".join(_MUSICAL_ALIASES.get(tok, tok) for tok in tokens)


def _format_composer_name(raw: str) -> str:
    cleaned = raw.replace("_", " ").strip()
    if "," in cleaned:
        last, first = (p.strip() for p in cleaned.split(",", 1))
        if last and first:
            return f"{first} {last}"
    return cleaned


def _safe_filename(name: str, fallback_ext: str = ".musicxml") -> str:
    base = Path(name).name.strip() or f"score{fallback_ext}"
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("._") or f"score{fallback_ext}"
    if Path(cleaned).suffix.lower() not in SUPPORTED_SEARCH_EXTENSIONS:
        cleaned = f"{cleaned}{fallback_ext}"
    return cleaned


def _encode_url_path(path: str) -> str:
    return quote(path.lstrip("/"), safe="/")


def _github_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get_json(
    url: str,
    timeout: float = _REQUEST_TIMEOUT_SEC,
    headers: dict[str, str] | None = None,
) -> Any:
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    resp = requests.get(validate_https_url(url), headers=req_headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _http_get_text(url: str, timeout: float = _REQUEST_TIMEOUT_SEC) -> str:
    resp = requests.get(
        validate_https_url(url), headers={"User-Agent": USER_AGENT}, timeout=timeout
    )
    resp.raise_for_status()
    return resp.content.decode("utf-8", errors="replace")


def _fetch_github_tree(tree_url: str) -> list[dict[str, Any]]:
    data = _http_get_json(tree_url, headers=_github_headers(), timeout=12.0)
    tree = data.get("tree", []) if isinstance(data, dict) else []
    return [node for node in tree if isinstance(node, dict) and node.get("type") == "blob"]


def _load_or_fetch_index(
    cache_dir: Path,
    name: str,
    fetcher: Callable[[], list[dict[str, Any]]],
    *,
    allow_network: bool,
    ttl_sec: int = _DEFAULT_INDEX_TTL_SEC,
) -> list[dict[str, Any]]:
    index_dir = Path(cache_dir) / "search_index"
    cache_path = index_dir / f"{name}.json"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if not allow_network or age <= ttl_sec:
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    return loaded
            except (OSError, ValueError):
                pass

    if not allow_network:
        return []

    records = fetcher()
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(records), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        pass
    return records


def _fetch_openscore_orchestra_index() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in _fetch_github_tree(_OPENSCORE_ORCH_TREE_URL):
        rel_path = str(node.get("path", ""))
        if not rel_path.startswith("data/") or not rel_path.lower().endswith(".mxl"):
            continue
        if rel_path.lower().endswith("_melody.mxl"):
            continue
        parts = Path(rel_path).parts
        composer = _format_composer_name(parts[1]) if len(parts) >= 3 else None
        work = parts[2].replace("_", " ").strip() if len(parts) >= 3 else Path(rel_path).stem
        movement = parts[3].replace("_", " ").strip() if len(parts) >= 5 else ""
        records.append(
            {
                "title": f"{work} ({movement})" if movement else work,
                "composer": composer,
                "format_ext": ".mxl",
                "source": "OpenScore Orchestra",
                "filename": _safe_filename(Path(rel_path).name, fallback_ext=".mxl"),
                "download_url": f"{_OPENSCORE_ORCH_RAW_PREFIX}{_encode_url_path(rel_path)}",
            }
        )
    return records


def _fetch_when_in_rome_index() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in _fetch_github_tree(_WHEN_IN_ROME_TREE_URL):
        rel_path = str(node.get("path", ""))
        if not rel_path.startswith("Corpus/") or not rel_path.lower().endswith(".mxl"):
            continue
        parts = Path(rel_path).parts
        if len(parts) < 5 or parts[1] in {"Textbooks", "OpenScore-LiederCorpus"}:
            continue
        composer = _format_composer_name(parts[2])
        work = parts[3].replace("_", " ").strip()
        movement = parts[4].replace("_", " ").strip() if len(parts) >= 6 else ""
        records.append(
            {
                "title": f"{work} ({movement})" if movement else work,
                "composer": composer,
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": _safe_filename(f"{work}_{movement or 'score'}.mxl", fallback_ext=".mxl"),
                "edition_note": parts[1].replace("_", " ").strip(),
                "download_url": f"{_WHEN_IN_ROME_RAW_PREFIX}{_encode_url_path(rel_path)}",
            }
        )
    return records


def _fetch_openscore_lieder_index() -> list[dict[str, Any]]:
    data = _http_get_json(_OPENSCORE_LIEDER_INDEX_URL)
    if not isinstance(data, list):
        return []
    records: list[dict[str, Any]] = []
    for entry in data:
        out_rel = str(entry.get("out", "")).strip()
        if not out_rel:
            continue
        clean_rel = out_rel.removeprefix("../").lstrip("/")
        ext = Path(clean_rel).suffix.lower()
        if ext not in SUPPORTED_SEARCH_EXTENSIONS:
            continue
        parts = Path(clean_rel).parts
        composer = _format_composer_name(parts[1]) if len(parts) >= 2 else None
        collection = parts[2].replace("_", " ").strip() if len(parts) >= 3 and parts[2] != "_" else ""
        song_title = parts[3].replace("_", " ").strip() if len(parts) >= 4 else Path(clean_rel).stem
        records.append(
            {
                "title": f"{song_title} ({collection})" if collection else song_title,
                "composer": composer,
                "format_ext": ext,
                "source": "OpenScore Lieder",
                "filename": _safe_filename(f"{song_title}_{Path(clean_rel).name}", fallback_ext=ext),
                "download_url": f"{_OPENSCORE_LIEDER_RAW_PREFIX}{_encode_url_path(clean_rel)}",
            }
        )
    return records


def _fetch_openscore_sq_index() -> list[dict[str, Any]]:
    text = _http_get_text(_OPENSCORE_SQ_TSV_URL)
    records: list[dict[str, Any]] = []
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        score_id = (row.get("id") or "").strip()
        rel_dir = (row.get("path") or "").strip().strip("/")
        name = (row.get("name") or "").strip()
        if not score_id or not rel_dir:
            continue
        title = name or rel_dir.rsplit("/", 1)[-1].replace("_", " ")
        mxl_rel = f"{rel_dir}/sq{score_id}.mxl"
        records.append(
            {
                "title": title,
                "composer": _format_composer_name(rel_dir.split("/", 1)[0]),
                "format_ext": ".mxl",
                "source": "OpenScore StringQuartets",
                "filename": _safe_filename(f"{title}_sq{score_id}.mxl", fallback_ext=".mxl"),
                "download_url": f"{_OPENSCORE_SQ_RAW_PREFIX}{_encode_url_path(mxl_rel)}",
            }
        )
    return records


def _search_music21_corpus(query: str) -> list[dict[str, Any]]:
    tokens = _normalize_text(query).split()
    if not tokens:
        return []

    seen_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    for tok in sorted(tokens, key=len, reverse=True)[:2]:
        try:
            entries = music21.corpus.search(tok)
        except Exception as exc:  # noqa: BLE001 — non-fatal local corpus index fallback
            logger.debug("music21 corpus search failed for %r: %s", tok, exc)
            continue
        for entry in entries:
            rel_path = str(entry.sourcePath)
            if rel_path in seen_paths:
                continue
            ext = Path(rel_path).suffix.lower()
            if ext not in SUPPORTED_SEARCH_EXTENSIONS:
                continue
            seen_paths.add(rel_path)

            md = entry.metadata
            raw_title = (
                (getattr(md, "title", None) or "").strip()
                or (getattr(md, "movementName", None) or "").strip()
                or (getattr(md, "opusNumber", None) or "").strip()
            )
            composer = (getattr(md, "composer", None) or "").strip() or None
            path_label = rel_path.removesuffix(ext).replace("/", " - ").replace("_", " ")
            if composer is None and "/" in rel_path:
                composer = rel_path.split("/", 1)[0].replace("_", " ").title()
            records.append(
                {
                    "title": raw_title or path_label,
                    "composer": composer,
                    "format_ext": ext,
                    "source": "music21 Corpus",
                    "filename": _safe_filename(Path(rel_path).name, fallback_ext=ext),
                    "local_corpus_path": rel_path,
                }
            )
    return records


def _score_candidate(record: dict[str, Any], query: str) -> float:
    q_norm = _normalize_text(query)
    q_tokens = q_norm.split()
    if not q_tokens:
        return 0.0

    title_norm = _normalize_text(str(record.get("title", "")))
    composer_norm = _normalize_text(str(record.get("composer") or ""))
    filename_norm = _normalize_text(str(record.get("filename", "")))
    ext = str(record.get("format_ext", "")).lower()
    source = str(record.get("source", ""))

    haystack = f"{title_norm} {composer_norm} {filename_norm}".strip()
    if not haystack:
        return 0.0

    hay_words = set(haystack.split())
    title_words = set(title_norm.split())
    composer_words = set(composer_norm.split())

    matched = 0
    exact_word_hits = 0
    title_hits = 0
    composer_hits = 0
    for tok in q_tokens:
        if tok in hay_words:
            matched += 1
            exact_word_hits += 1
        elif tok in haystack:
            matched += 1
        if tok in title_words or tok in title_norm:
            title_hits += 1
        if tok in composer_words:
            composer_hits += 1

    min_required = len(q_tokens) if len(q_tokens) <= 2 else math.ceil(len(q_tokens) * 0.6)
    if matched < min_required:
        return 0.0

    coverage = matched / len(q_tokens)
    exact_ratio = exact_word_hits / len(q_tokens)
    score = 14.0 * coverage + 3.0 * exact_ratio + 4.0 * (composer_hits > 0)

    if matched == len(q_tokens):
        score += 8.0
        if ext in MUSICXML_SEARCH_EXTENSIONS:
            score += 15.0
    if q_norm in title_norm:
        score += 6.0
    elif q_norm in haystack:
        score += 3.0

    score += 2.0 * (title_hits / max(1, len(title_words)))
    score += 18.0 * _FORMAT_PRIORITY.get(ext, 0.2)
    score += _SOURCE_BIAS.get(source, 0.0)
    score += float(record.get("source_bonus", 0.0) or 0.0)
    return round(score, 4)


def _normalize_allowed_formats(
    formats: Iterable[str] | None,
    *,
    musicxml_only: bool,
) -> frozenset[str]:
    if formats is not None:
        cleaned = {f".{f.strip().lstrip('.').lower()}" for f in formats if f.strip()}
        valid = frozenset(cleaned & SUPPORTED_SEARCH_EXTENSIONS)
        if not valid:
            supported = ", ".join(sorted(SUPPORTED_SEARCH_EXTENSIONS))
            raise ValueError(f"No supported score formats in {list(formats)!r}. Supported: {supported}")
        return frozenset(valid & MUSICXML_SEARCH_EXTENSIONS) if musicxml_only else valid
    return MUSICXML_SEARCH_EXTENSIONS if musicxml_only else SUPPORTED_SEARCH_EXTENSIONS


def search_digital_scores(
    query: str,
    *,
    limit: int = 5,
    musicxml_only: bool = False,
    formats: Iterable[str] | None = None,
    include_pdmx: bool = True,
    include_remote: bool = True,
    include_local: bool = True,
    cache_dir: Path = Path(".transcribe_cache"),
) -> list[ScoreSearchResult]:
    """Search fast symbolic catalogs (local PDMX/music21 + cached OpenScore/When-in-Rome)."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    allowed_exts = _normalize_allowed_formats(formats, musicxml_only=musicxml_only)

    def _safe_index(name: str, fetcher: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        try:
            return _load_or_fetch_index(cache_dir, name, fetcher, allow_network=include_remote)
        except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
            logger.debug("Skipping search index %s: %s", name, exc)
            return []

    tasks: list[Callable[[], list[dict[str, Any]]]] = [
        lambda: _safe_index("openscore_orchestra", _fetch_openscore_orchestra_index),
        lambda: _safe_index("openscore_lieder", _fetch_openscore_lieder_index),
        lambda: _safe_index("openscore_stringquartets", _fetch_openscore_sq_index),
        lambda: _safe_index("when_in_rome", _fetch_when_in_rome_index),
    ]
    if include_local:
        tasks.append(lambda: _search_music21_corpus(clean_query))
    if include_pdmx:
        tasks.append(
            lambda: search_pdmx_catalog(clean_query, cache_dir=cache_dir, allowed_exts=allowed_exts)
        )

    raw_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for fut in as_completed(pool.submit(fn) for fn in tasks):
            raw_records.extend(fut.result())

    scored_candidates: list[ScoreSearchResult] = []
    for rec in raw_records:
        ext = str(rec.get("format_ext", "")).lower()
        if ext not in allowed_exts:
            continue
        score = _score_candidate(rec, clean_query)
        if score <= 0.0:
            continue
        scored_candidates.append(
            ScoreSearchResult(
                title=str(rec["title"]),
                composer=rec.get("composer"),
                format_ext=ext,
                source=str(rec["source"]),
                filename=str(rec["filename"]),
                download_url=rec.get("download_url"),
                page_url=rec.get("page_url"),
                edition_note=rec.get("edition_note"),
                local_corpus_path=rec.get("local_corpus_path"),
                pdmx_dir=rec.get("pdmx_dir"),
                pdmx_member=rec.get("pdmx_member"),
                score=score,
            )
        )

    scored_candidates.sort(
        key=lambda r: (r.score, _FORMAT_PRIORITY.get(r.format_ext, 0.0)),
        reverse=True,
    )

    ranked: list[ScoreSearchResult] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in scored_candidates:
        dedup_key = (
            _normalize_text(item.title),
            _normalize_text(item.composer or ""),
            item.format_ext,
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        ranked.append(item)
        if len(ranked) >= limit:
            break
    return ranked


def _read_bounded_response(resp: Any, max_bytes: int = _MAX_DOWNLOAD_BYTES) -> bytes:
    headers = getattr(resp, "headers", None) or {}
    content_len = headers.get("content-length")
    if content_len and str(content_len).isdigit() and int(content_len) > max_bytes:
        raise ValueError(f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit.")

    if callable(getattr(resp, "iter_content", None)):
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65_536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit.")
            chunks.append(chunk)
        return b"".join(chunks)

    payload = getattr(resp, "content", b"")
    if len(payload) > max_bytes:
        raise ValueError(f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit.")
    return bytes(payload)


def download_score_result(
    result: ScoreSearchResult,
    *,
    output_path: Path | None = None,
    download_dir: Path = Path("downloads"),
) -> Path:
    """Download (or copy/extract locally) a selected ScoreSearchResult."""
    safe_name = _safe_filename(result.filename, fallback_ext=result.format_ext)
    if output_path is not None:
        out_path = Path(output_path)
        if (out_path.exists() and out_path.is_dir()) or str(output_path).endswith(("/", "\\")):
            dest = out_path / safe_name
        else:
            dest = out_path
    else:
        dest = Path(download_dir) / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if result.local_corpus_path:
        src_path = music21.corpus.getWork(result.local_corpus_path)
        shutil.copy2(src_path, dest)
        return dest

    if result.pdmx_dir and result.pdmx_member:
        return extract_pdmx_member(result.pdmx_dir, result.pdmx_member, dest)

    if not result.download_url:
        raise ValueError(f"Search result has no downloadable URL or local corpus path: {result.title}")

    download_url = validate_https_url(result.download_url)
    resp = requests.get(
        download_url,
        headers={"User-Agent": USER_AGENT},
        stream=True,
        timeout=30.0,
    )
    try:
        resp.raise_for_status()
        final_url = getattr(resp, "url", None)
        if isinstance(final_url, str) and final_url:
            validate_https_url(final_url)
        payload = _read_bounded_response(resp)
    finally:
        close_fn = getattr(resp, "close", None)
        if callable(close_fn):
            close_fn()

    if not payload:
        raise RuntimeError(f"Downloaded file was empty: {download_url}")

    tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp")
    tmp_dest.write_bytes(payload)
    tmp_dest.replace(dest)
    return dest
