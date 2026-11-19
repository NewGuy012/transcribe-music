"""Search and download digital symbolic music scores across fast curated repositories."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import time
import urllib.error
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import music21

from transcribe_music.ingest._io import (
    MAX_MEMBER_BYTES,
    atomic_write_text,
    copy_file_safe,
    download_https_stream,
    http_get_json,
    http_get_text,
    normalize_ascii,
)
from transcribe_music.ingest.digital import (
    DIGITAL_EXTENSIONS,
    GUITAR_TAB_EXTENSIONS,
    MUSICXML_EXTENSIONS,
)
from transcribe_music.ingest.gp_catalog import extract_gp_catalog_member, search_gp_catalog
from transcribe_music.ingest.pdmx import extract_pdmx_member, search_pdmx_catalog
from transcribe_music.models import Instrument, ScoreSearchResult

logger = logging.getLogger(__name__)

SUPPORTED_SEARCH_EXTENSIONS: frozenset[str] = DIGITAL_EXTENSIONS
MUSICXML_SEARCH_EXTENSIONS: frozenset[str] = MUSICXML_EXTENSIONS

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
    ".tab": 0.58,
    ".mid": 0.55,
    ".midi": 0.55,
}

_SOURCE_BIAS: dict[str, float] = {
    "OpenScore Orchestra": 1.1,
    "GP Classical Guitar": 1.05,
    "OpenScore Lieder": 1.0,
    "OpenScore StringQuartets": 1.0,
    "DadaGP": 1.0,
    "When-in-Rome": 0.95,
    "PDMX": 0.9,
    "music21 Corpus": 0.6,
}

_DEFAULT_INDEX_TTL_SEC = 86_400
_REQUEST_TIMEOUT_SEC = 7.0
_MAX_DOWNLOAD_BYTES = MAX_MEMBER_BYTES

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

_GP_CLASSICAL_GUITAR_TREE_URL = (
    "https://api.github.com/repos/otnemrasordep/gp-classical-guitar/git/trees/master?recursive=1"
)
_GP_CLASSICAL_GUITAR_RAW_PREFIX = (
    "https://raw.githubusercontent.com/otnemrasordep/gp-classical-guitar/master/"
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ALPHA_DIGIT_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")
_CATALOG_TOKEN_RE = re.compile(r"^(?:k|kv|bwv|op|hwv|rv|hob|d|s|w|mwv|woo|sz)[0-9]+$")
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
    raw_tokens = _NON_ALNUM_RE.sub(" ", normalize_ascii(text)).split()
    out_tokens: list[str] = []
    for tok in raw_tokens:
        if _CATALOG_TOKEN_RE.match(tok):
            out_tokens.append(tok)
        else:
            for sub in _ALPHA_DIGIT_SPLIT_RE.split(tok):
                if sub:
                    out_tokens.append(_MUSICAL_ALIASES.get(sub, sub))
    return " ".join(out_tokens)


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
    headers = {"Accept": "application/vnd.github+json"}
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_github_tree(tree_url: str) -> list[dict[str, Any]]:
    data = http_get_json(tree_url, headers=_github_headers(), timeout=12.0)
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
    cache_path = Path(cache_dir) / "search_index" / f"{name}.json"
    stale_records: list[dict[str, Any]] | None = None

    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                if not allow_network or (time.time() - cache_path.stat().st_mtime) <= ttl_sec:
                    return loaded
                stale_records = loaded
        except (OSError, ValueError):
            pass

    if not allow_network:
        return []

    try:
        records = fetcher()
    except (urllib.error.URLError, ValueError, OSError, RuntimeError) as exc:
        if stale_records is not None:
            logger.debug("Using stale search index %s after fetch error: %s", name, exc)
            return stale_records
        raise
    try:
        atomic_write_text(cache_path, json.dumps(records))
    except OSError:
        pass
    return records


def _fetch_github_records(
    tree_url: str,
    mapper: Callable[[str], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in _fetch_github_tree(tree_url):
        rel_path = str(node.get("path", ""))
        if rec := mapper(rel_path):
            records.append(rec)
    return records


def _fetch_openscore_orchestra_index() -> list[dict[str, Any]]:
    def _map(rel_path: str) -> dict[str, Any] | None:
        if not rel_path.startswith("data/") or not rel_path.lower().endswith(".mxl"):
            return None
        if rel_path.lower().endswith("_melody.mxl"):
            return None
        parts = Path(rel_path).parts
        composer = _format_composer_name(parts[1]) if len(parts) >= 3 else None
        work = parts[2].replace("_", " ").strip() if len(parts) >= 3 else Path(rel_path).stem
        movement = parts[3].replace("_", " ").strip() if len(parts) >= 5 else ""
        return {
            "title": f"{work} ({movement})" if movement else work,
            "composer": composer,
            "format_ext": ".mxl",
            "source": "OpenScore Orchestra",
            "filename": _safe_filename(Path(rel_path).name, fallback_ext=".mxl"),
            "download_url": f"{_OPENSCORE_ORCH_RAW_PREFIX}{_encode_url_path(rel_path)}",
        }

    return _fetch_github_records(_OPENSCORE_ORCH_TREE_URL, _map)


def _fetch_when_in_rome_index() -> list[dict[str, Any]]:
    def _map(rel_path: str) -> dict[str, Any] | None:
        if not rel_path.startswith("Corpus/") or not rel_path.lower().endswith(".mxl"):
            return None
        parts = Path(rel_path).parts
        if len(parts) < 5 or parts[1] in {"Textbooks", "OpenScore-LiederCorpus"}:
            return None
        work = parts[3].replace("_", " ").strip()
        movement = parts[4].replace("_", " ").strip() if len(parts) >= 6 else ""
        return {
            "title": f"{work} ({movement})" if movement else work,
            "composer": _format_composer_name(parts[2]),
            "format_ext": ".mxl",
            "source": "When-in-Rome",
            "filename": _safe_filename(f"{work}_{movement or 'score'}.mxl", fallback_ext=".mxl"),
            "edition_note": parts[1].replace("_", " ").strip(),
            "download_url": f"{_WHEN_IN_ROME_RAW_PREFIX}{_encode_url_path(rel_path)}",
        }

    return _fetch_github_records(_WHEN_IN_ROME_TREE_URL, _map)


def _fetch_openscore_lieder_index() -> list[dict[str, Any]]:
    data = http_get_json(_OPENSCORE_LIEDER_INDEX_URL, timeout=_REQUEST_TIMEOUT_SEC)
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
    text = http_get_text(_OPENSCORE_SQ_TSV_URL, timeout=_REQUEST_TIMEOUT_SEC)
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


def _fetch_gp_classical_guitar_index() -> list[dict[str, Any]]:
    def _map(rel_path: str) -> dict[str, Any] | None:
        ext = Path(rel_path).suffix.lower()
        if ext not in SUPPORTED_SEARCH_EXTENSIONS:
            return None
        parts = Path(rel_path).parts
        if any(p.lower() == "dada" for p in parts[:-1]):
            return None
        clean_dirs = [
            p.replace("_", " ").strip()
            for p in parts[:-1]
            if p.lower() not in {"data", "scores", "midi", "mid", "gp", "gp5", "gpx", "files"}
        ]
        composer = _format_composer_name(clean_dirs[0]) if clean_dirs else None
        stem_title = Path(rel_path).stem.replace("_", " ").strip()
        work_group = " - ".join(clean_dirs[1:]) if len(clean_dirs) > 1 else ""
        return {
            "title": f"{stem_title} ({work_group})" if work_group else stem_title,
            "composer": composer,
            "format_ext": ext,
            "source": "GP Classical Guitar",
            "filename": _safe_filename(Path(rel_path).name, fallback_ext=ext),
            "edition_note": "Classical Guitar MIDI" if ext in {".mid", ".midi"} else "Classical Guitar TAB",
            "download_url": f"{_GP_CLASSICAL_GUITAR_RAW_PREFIX}{_encode_url_path(rel_path)}",
        }

    return _fetch_github_records(_GP_CLASSICAL_GUITAR_TREE_URL, _map)


def _search_music21_corpus(query: str) -> list[dict[str, Any]]:
    tokens = _normalize_text(query).split()
    if not tokens:
        return []

    seen_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    for tok in sorted(tokens, key=len, reverse=True)[:3]:
        try:
            entries = music21.corpus.search(tok)
        except (music21.exceptions21.Music21Exception, OSError, ValueError) as exc:
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


def _is_guitar_intent(q_tokens: list[str], instrument: Instrument | None) -> bool:
    return instrument == Instrument.GUITAR or "guitar" in q_tokens or "tab" in q_tokens


def _effective_format_priority(ext: str, guitar_intent: bool, default: float = 0.0) -> float:
    prio = _FORMAT_PRIORITY.get(ext, default)
    return max(prio, 0.96) if (guitar_intent and ext in GUITAR_TAB_EXTENSIONS) else prio


def _score_candidate(
    record: dict[str, Any],
    q_norm: str,
    q_tokens: list[str],
    *,
    instrument: Instrument | None = None,
    guitar_intent: bool = False,
) -> float:
    if not q_tokens:
        return 0.0

    title_norm = _normalize_text(str(record.get("title", "")))
    composer_norm = _normalize_text(str(record.get("composer") or ""))
    filename_norm = _normalize_text(str(record.get("filename", "")))
    edition_norm = _normalize_text(str(record.get("edition_note") or ""))
    corpus_path_norm = _normalize_text(str(record.get("local_corpus_path") or ""))
    ext = str(record.get("format_ext", "")).lower()
    source = str(record.get("source", ""))

    haystack = (
        f"{title_norm} {composer_norm} {filename_norm} {edition_norm} {corpus_path_norm}".strip()
    )
    if not haystack:
        return 0.0

    hay_words = set(haystack.split())
    for w in list(hay_words):
        if _CATALOG_TOKEN_RE.match(w):
            hay_words.update(_ALPHA_DIGIT_SPLIT_RE.split(w))
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

    is_preferred_fmt = ext in MUSICXML_SEARCH_EXTENSIONS or (
        guitar_intent and ext in GUITAR_TAB_EXTENSIONS
    )
    fmt_prio = _effective_format_priority(ext, guitar_intent, default=0.2)

    coverage = matched / len(q_tokens)
    exact_ratio = exact_word_hits / len(q_tokens)
    score = 14.0 * coverage + 3.0 * exact_ratio + 4.0 * (composer_hits > 0)

    if matched == len(q_tokens):
        score += 8.0
        if is_preferred_fmt:
            score += 15.0
    if q_norm in title_norm:
        score += 6.0
    elif q_norm in haystack:
        score += 3.0

    score += 2.0 * (title_hits / max(1, len(title_words)))
    score += 18.0 * fmt_prio
    score += _SOURCE_BIAS.get(source, 0.0)
    score += float(record.get("source_bonus", 0.0) or 0.0)

    if guitar_intent:
        if source in {"GP Classical Guitar", "DadaGP"} or "guitar" in hay_words or "tab" in hay_words:
            score += 4.0
        if source == "OpenScore StringQuartets" and "quartet" not in q_tokens:
            score -= 4.0
    elif instrument is not None and instrument.value in hay_words:
        score += 2.0

    return round(score, 4)


def _normalize_allowed_formats(
    formats: Iterable[str] | None,
    *,
    musicxml_only: bool,
) -> frozenset[str]:
    supported = SUPPORTED_SEARCH_EXTENSIONS
    if formats is not None:
        cleaned = {f".{f.strip().lstrip('.').lower()}" for f in formats if f.strip()}
        valid = frozenset(cleaned & supported)
        if not valid:
            supported_str = ", ".join(sorted(supported))
            raise ValueError(
                f"No supported score formats in {list(formats)!r}. Supported: {supported_str}"
            )
        return frozenset(valid & MUSICXML_SEARCH_EXTENSIONS) if musicxml_only else valid
    return MUSICXML_SEARCH_EXTENSIONS if musicxml_only else supported


def search_digital_scores(
    query: str,
    *,
    limit: int = 5,
    instrument: Instrument | None = None,
    musicxml_only: bool = False,
    formats: Iterable[str] | None = None,
    include_pdmx: bool = True,
    include_unsynced_pdmx: bool = False,
    include_gp_catalog: bool = True,
    include_remote: bool = True,
    include_local: bool = True,
    cache_dir: Path = Path(".transcribe_cache"),
) -> list[ScoreSearchResult]:
    """Search fast symbolic catalogs (local PDMX/DadaGP/music21 + cached GitHub corpora)."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    allowed_exts = _normalize_allowed_formats(formats, musicxml_only=musicxml_only)
    q_norm = _normalize_text(clean_query)
    q_tokens = q_norm.split()
    guitar_intent = _is_guitar_intent(q_tokens, instrument)

    def _safe_index(name: str, fetcher: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        try:
            return _load_or_fetch_index(cache_dir, name, fetcher, allow_network=include_remote)
        except (urllib.error.URLError, ValueError, OSError, RuntimeError) as exc:
            logger.debug("Skipping search index %s: %s", name, exc)
            return []

    tasks: list[Callable[[], list[dict[str, Any]]]] = [
        lambda: _safe_index("openscore_orchestra", _fetch_openscore_orchestra_index),
        lambda: _safe_index("openscore_lieder", _fetch_openscore_lieder_index),
        lambda: _safe_index("openscore_stringquartets", _fetch_openscore_sq_index),
        lambda: _safe_index("when_in_rome", _fetch_when_in_rome_index),
        lambda: _safe_index("gp_classical_guitar", _fetch_gp_classical_guitar_index),
    ]
    if include_local:
        tasks.append(lambda: _search_music21_corpus(clean_query))
    if include_pdmx:
        tasks.append(
            lambda: search_pdmx_catalog(
                clean_query,
                cache_dir=cache_dir,
                allowed_exts=allowed_exts,
                instrument=instrument,
                include_unsynced_pdmx=include_unsynced_pdmx,
            )
        )
    if include_gp_catalog:
        tasks.append(
            lambda: search_gp_catalog(
                clean_query,
                cache_dir=cache_dir,
                allowed_exts=allowed_exts,
            )
        )

    raw_records: list[dict[str, Any]] = []
    if include_remote and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            for fut in as_completed(pool.submit(fn) for fn in tasks):
                try:
                    raw_records.extend(fut.result())
                except Exception as exc:  # noqa: BLE001 — isolate failing search backend
                    logger.debug("Search backend failed safely: %s", exc)
    else:
        for fn in tasks:
            try:
                raw_records.extend(fn())
            except Exception as exc:  # noqa: BLE001 — isolate failing search backend
                logger.debug("Search backend failed safely: %s", exc)

    scored_candidates: list[ScoreSearchResult] = []
    for rec in raw_records:
        ext = str(rec.get("format_ext", "")).lower()
        if ext not in allowed_exts:
            continue
        score = _score_candidate(
            rec, q_norm, q_tokens, instrument=instrument, guitar_intent=guitar_intent
        )
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
                edition_note=rec.get("edition_note"),
                local_corpus_path=rec.get("local_corpus_path"),
                pdmx_dir=rec.get("pdmx_dir"),
                pdmx_member=rec.get("pdmx_member"),
                gp_dir=rec.get("gp_dir"),
                gp_member=rec.get("gp_member"),
                score=score,
            )
        )

    scored_candidates.sort(
        key=lambda r: (r.score, _effective_format_priority(r.format_ext, guitar_intent)),
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
        return copy_file_safe(Path(src_path), dest)

    if result.pdmx_dir and result.pdmx_member:
        return extract_pdmx_member(result.pdmx_dir, result.pdmx_member, dest)

    if result.gp_dir and result.gp_member:
        return extract_gp_catalog_member(result.gp_dir, result.gp_member, dest)

    if not result.download_url:
        raise ValueError(f"Search result has no downloadable URL or local corpus path: {result.title}")

    return download_https_stream(
        result.download_url,
        dest,
        timeout=30.0,
        max_bytes=_MAX_DOWNLOAD_BYTES,
        require_non_empty=True,
    )
