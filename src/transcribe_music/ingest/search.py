"""Search and download digital symbolic music scores across open repositories."""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import re
import shutil
import time
import unicodedata
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import music21
import requests

from transcribe_music.ingest.digital import DIGITAL_EXTENSIONS
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
    ".mscz": 0.65,
    ".mscx": 0.65,
    ".gp5": 0.60,
    ".gp4": 0.60,
    ".gp3": 0.60,
    ".mid": 0.55,
    ".midi": 0.55,
}

_SOURCE_BIAS: dict[str, float] = {
    "MuseTrainer": 1.1,
    "OpenScore Lieder": 1.0,
    "OpenScore StringQuartets": 1.0,
    "music21 Corpus": 0.6,
    "Humdrum GitHub": 0.5,
    "Archive.org": 0.4,
}

_USER_AGENT = "transcribe-music/0.2 (+https://github.com/transcribe-music)"
_DEFAULT_INDEX_TTL_SEC = 86_400
_REQUEST_TIMEOUT_SEC = 7.0

_OPENSCORE_LIEDER_INDEX_URL = (
    "https://raw.githubusercontent.com/OpenScore/Lieder/main/data/corpus_conversion.json"
)
_OPENSCORE_LIEDER_RAW_PREFIX = "https://raw.githubusercontent.com/OpenScore/Lieder/main/"

_OPENSCORE_SQ_TSV_URL = (
    "https://raw.githubusercontent.com/OpenScore/StringQuartets/main/data/scores.tsv"
)
_OPENSCORE_SQ_RAW_PREFIX = "https://raw.githubusercontent.com/OpenScore/StringQuartets/main/scores/"

_MUSETRAINER_TREE_URL = (
    "https://api.github.com/repos/musetrainer/library/git/trees/master?recursive=1"
)
_MUSETRAINER_RAW_PREFIX = "https://raw.githubusercontent.com/musetrainer/library/master/"

_HUMDRUM_GITHUB_REPOS: tuple[tuple[str, str, str], ...] = (
    ("craigsapp/beethoven-piano-sonatas", "master", "Ludwig van Beethoven"),
    ("craigsapp/mozart-piano-sonatas", "master", "Wolfgang Amadeus Mozart"),
    ("craigsapp/bach-370-chorales", "master", "Johann Sebastian Bach"),
    ("craigsapp/joplin", "master", "Scott Joplin"),
)

_ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
_ARCHIVE_METADATA_URL = "https://archive.org/metadata/{identifier}"
_ARCHIVE_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

_KNOWN_COMPOSERS: tuple[str, ...] = (
    "Bach",
    "Beethoven",
    "Brahms",
    "Chopin",
    "Debussy",
    "Dvorak",
    "Handel",
    "Haydn",
    "Joplin",
    "Liszt",
    "Mendelssohn",
    "Mozart",
    "Pachelbel",
    "Rachmaninoff",
    "Ravel",
    "Satie",
    "Schubert",
    "Schumann",
    "Tchaikovsky",
    "Vivaldi",
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
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


def _strip_html(text: str) -> str:
    unescaped = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", unescaped)
    return " ".join(cleaned.split())


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", _strip_html(text))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    tokens = _NON_ALNUM_RE.sub(" ", ascii_only.lower()).split()
    return " ".join(_MUSICAL_ALIASES.get(tok, tok) for tok in tokens)


def _format_composer_name(raw: str) -> str:
    cleaned = raw.replace("_", " ").strip()
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        if parts[0] and parts[1]:
            return f"{parts[1]} {parts[0]}"
    return cleaned


def _safe_filename(name: str, fallback_ext: str = ".musicxml") -> str:
    base = Path(name).name.strip()
    if not base:
        base = f"score{fallback_ext}"
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("._")
    if not cleaned:
        cleaned = f"score{fallback_ext}"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in SUPPORTED_SEARCH_EXTENSIONS:
        cleaned = f"{cleaned}{fallback_ext}"
    return cleaned


def _encode_url_path(path: str) -> str:
    return quote(path.lstrip("/"), safe="/")


def _http_get_json(url: str, params: dict[str, Any] | None = None, timeout: float = _REQUEST_TIMEOUT_SEC) -> Any:
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _http_get_text(url: str, timeout: float = _REQUEST_TIMEOUT_SEC) -> str:
    resp = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.text


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


def _fetch_musetrainer_index() -> list[dict[str, Any]]:
    data = _http_get_json(_MUSETRAINER_TREE_URL)
    tree = data.get("tree", []) if isinstance(data, dict) else []
    records: list[dict[str, Any]] = []
    for node in tree:
        rel_path = str(node.get("path", ""))
        if not rel_path.startswith("scores/"):
            continue
        ext = Path(rel_path).suffix.lower()
        if ext not in SUPPORTED_SEARCH_EXTENSIONS:
            continue
        filename = Path(rel_path).name
        stem = Path(rel_path).stem
        title = " ".join(stem.replace("_", " ").split())
        composer: str | None = None
        norm_title = _normalize_text(title)
        for cand in _KNOWN_COMPOSERS:
            if cand.lower() in norm_title.split():
                composer = cand
                break
        records.append(
            {
                "title": title,
                "composer": composer,
                "format_ext": ext,
                "source": "MuseTrainer",
                "filename": filename,
                "download_url": f"{_MUSETRAINER_RAW_PREFIX}{_encode_url_path(rel_path)}",
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
        # Expected: scores/<Composer>/<Set>/<Song>/lc<id>.mxl
        composer = _format_composer_name(parts[1]) if len(parts) >= 2 else None
        collection = parts[2].replace("_", " ").strip() if len(parts) >= 3 and parts[2] != "_" else ""
        song_title = parts[3].replace("_", " ").strip() if len(parts) >= 4 else Path(clean_rel).stem
        full_title = f"{song_title} ({collection})" if collection else song_title
        filename = f"{song_title}_{Path(clean_rel).name}"
        records.append(
            {
                "title": full_title,
                "composer": composer,
                "format_ext": ext,
                "source": "OpenScore Lieder",
                "filename": _safe_filename(filename, fallback_ext=ext),
                "download_url": f"{_OPENSCORE_LIEDER_RAW_PREFIX}{_encode_url_path(clean_rel)}",
            }
        )
    return records


def _fetch_openscore_sq_index() -> list[dict[str, Any]]:
    text = _http_get_text(_OPENSCORE_SQ_TSV_URL)
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    records: list[dict[str, Any]] = []
    for row in reader:
        score_id = (row.get("id") or "").strip()
        rel_dir = (row.get("path") or "").strip().strip("/")
        name = (row.get("name") or "").strip()
        if not score_id or not rel_dir:
            continue
        composer = _format_composer_name(rel_dir.split("/", 1)[0])
        title = name or rel_dir.rsplit("/", 1)[-1].replace("_", " ")
        mxl_rel = f"{rel_dir}/sq{score_id}.mxl"
        records.append(
            {
                "title": title,
                "composer": composer,
                "format_ext": ".mxl",
                "source": "OpenScore StringQuartets",
                "filename": _safe_filename(f"{title}_sq{score_id}.mxl", fallback_ext=".mxl"),
                "download_url": f"{_OPENSCORE_SQ_RAW_PREFIX}{_encode_url_path(mxl_rel)}",
            }
        )
    return records


def _fetch_humdrum_github_index() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for repo, branch, composer in _HUMDRUM_GITHUB_REPOS:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/index.hmd"
        try:
            text = _http_get_text(url)
        except Exception as exc:  # noqa: BLE001 — skip unreachable Humdrum sub-repo
            logger.debug("Skipping Humdrum index %s: %s", repo, exc)
            continue

        group_titles: dict[str, str] = {}
        for line in text.splitlines():
            if not line or line.startswith(("!", "*")):
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            file_col, sort_col, available_col, _pdf_col, desc_col = cols[:5]
            desc = _strip_html(desc_col)
            if file_col.startswith("@"):
                prefix = sort_col.split("-", 1)[0]
                if prefix and desc:
                    group_titles[prefix] = desc
                continue
            if available_col.strip().upper() == "N":
                continue
            ext = Path(file_col).suffix.lower()
            if ext not in SUPPORTED_SEARCH_EXTENSIONS:
                continue
            prefix = sort_col.split("-", 1)[0]
            parent_title = group_titles.get(prefix, "")
            title = f"{parent_title} - {desc}" if parent_title and desc else (desc or parent_title or Path(file_col).stem)
            records.append(
                {
                    "title": title,
                    "composer": composer,
                    "format_ext": ext,
                    "source": "Humdrum GitHub",
                    "filename": _safe_filename(f"{Path(repo).name}_{Path(file_col).name}", fallback_ext=ext),
                    "download_url": f"https://raw.githubusercontent.com/{repo}/{branch}/{_encode_url_path(file_col)}",
                }
            )
    return records


def _search_music21_corpus(query: str) -> list[dict[str, Any]]:
    tokens = _normalize_text(query).split()
    if not tokens:
        return []

    seen_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    # Query the primary/longest tokens in music21's indexed corpus for fast recall
    search_tokens = sorted(tokens, key=len, reverse=True)[:2]
    for tok in search_tokens:
        try:
            entries = music21.corpus.search(tok)
        except Exception as exc:  # noqa: BLE001 — fallback cleanly if corpus index fails
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
            title = raw_title or path_label
            if composer is None and "/" in rel_path:
                composer = rel_path.split("/", 1)[0].replace("_", " ").title()
            records.append(
                {
                    "title": title,
                    "composer": composer,
                    "format_ext": ext,
                    "source": "music21 Corpus",
                    "filename": _safe_filename(Path(rel_path).name, fallback_ext=ext),
                    "local_corpus_path": rel_path,
                }
            )
    return records


def _pick_best_archive_file(
    files: list[dict[str, Any]],
    allowed_exts: frozenset[str],
) -> tuple[str, str] | None:
    best: tuple[float, str, str] | None = None
    for f in files:
        fname = str(f.get("name", "")).strip()
        if not fname:
            continue
        ext = Path(fname).suffix.lower()
        if ext not in allowed_exts:
            continue
        # Skip Archive.org internal XML metadata files (*_meta.xml, *_files.xml, *_djvu.xml, *_scandata.xml)
        lower_name = fname.lower()
        if ext == ".xml" and lower_name.endswith(
            ("_meta.xml", "_files.xml", "_djvu.xml", "_scandata.xml", "_archive.xml")
        ):
            continue
        prio = _FORMAT_PRIORITY.get(ext, 0.1)
        if best is None or prio > best[0]:
            best = (prio, fname, ext)
    if best is None:
        return None
    return best[1], best[2]


def _resolve_archive_item(
    doc: dict[str, Any],
    allowed_exts: frozenset[str],
) -> dict[str, Any] | None:
    identifier = str(doc.get("identifier", "")).strip()
    if not identifier:
        return None
    try:
        meta = _http_get_json(_ARCHIVE_METADATA_URL.format(identifier=quote(identifier)), timeout=5.0)
    except Exception:  # noqa: BLE001 — skip slow/failing Archive metadata calls
        return None

    files = meta.get("files", []) if isinstance(meta, dict) else []
    picked = _pick_best_archive_file(files, allowed_exts)
    if picked is None:
        return None
    fname, ext = picked

    raw_title = doc.get("title")
    title = (raw_title[0] if isinstance(raw_title, list) else raw_title) or identifier
    raw_creator = doc.get("creator")
    composer = (raw_creator[0] if isinstance(raw_creator, list) else raw_creator) or None

    return {
        "title": str(title).strip(),
        "composer": str(composer).strip() if composer else None,
        "format_ext": ext,
        "source": "Archive.org",
        "filename": _safe_filename(fname, fallback_ext=ext),
        "download_url": _ARCHIVE_DOWNLOAD_URL.format(
            identifier=quote(identifier),
            filename=quote(fname, safe="/"),
        ),
        "archive_identifier": identifier,
    }


def _search_archive_org(
    query: str,
    allowed_exts: frozenset[str],
    *,
    musicxml_only: bool,
) -> list[dict[str, Any]]:
    tokens = _normalize_text(query).split()
    if not tokens:
        return []

    term_expr = " AND ".join(tokens)
    if musicxml_only:
        fmt_clause = 'format:"MusicXML"'
    else:
        fmt_clause = '(format:"MusicXML" OR format:"MIDI")'
    q_expr = f"({term_expr}) AND {fmt_clause}"

    try:
        data = _http_get_json(
            _ARCHIVE_SEARCH_URL,
            params={
                "q": q_expr,
                "fl[]": ["identifier", "title", "creator", "format"],
                "rows": 8,
                "output": "json",
            },
            timeout=6.0,
        )
    except Exception:  # noqa: BLE001 — keep search working if Archive.org is unavailable
        return []

    docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []
    if not docs:
        return []

    # Prioritize docs explicitly advertising MusicXML before resolving metadata
    docs.sort(
        key=lambda d: "MusicXML" in (d.get("format") or []),
        reverse=True,
    )
    top_docs = docs[:6]

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_resolve_archive_item, doc, allowed_exts) for doc in top_docs]
        for fut in as_completed(futures):
            resolved = fut.result()
            if resolved is not None:
                records.append(resolved)
    return records


def _score_candidate(record: dict[str, Any], query: str) -> float:
    q_norm = _normalize_text(query)
    q_tokens = q_norm.split()
    if not q_tokens:
        return 0.0

    title = str(record.get("title", ""))
    composer = str(record.get("composer") or "")
    filename = str(record.get("filename", ""))
    ext = str(record.get("format_ext", "")).lower()
    source = str(record.get("source", ""))

    title_norm = _normalize_text(title)
    composer_norm = _normalize_text(composer)
    haystack = f"{title_norm} {composer_norm} {_normalize_text(filename)}".strip()
    if not haystack:
        return 0.0

    hay_words = set(haystack.split())
    title_words = set(title_norm.split())

    matched = 0
    exact_word_hits = 0
    title_hits = 0
    for tok in q_tokens:
        if tok in hay_words:
            matched += 1
            exact_word_hits += 1
        elif tok in haystack:
            matched += 1
        if tok in title_words or tok in title_norm:
            title_hits += 1

    min_required = len(q_tokens) if len(q_tokens) <= 2 else math.ceil(len(q_tokens) * 0.6)
    if matched < min_required:
        return 0.0

    coverage = matched / len(q_tokens)
    exact_ratio = exact_word_hits / len(q_tokens)
    score = 10.0 * coverage + 3.0 * exact_ratio

    if matched == len(q_tokens):
        score += 4.0
        if ext in MUSICXML_SEARCH_EXTENSIONS:
            score += 15.0
    if q_norm in title_norm:
        score += 6.0
    elif q_norm in haystack:
        score += 3.0

    score += 2.0 * (title_hits / max(1, len(title_words)))
    # Strongly prioritize MusicXML (.musicxml/.mxl/.xml) over other digital formats
    score += 18.0 * _FORMAT_PRIORITY.get(ext, 0.2)
    score += _SOURCE_BIAS.get(source, 0.0)
    if ".omr" in filename.lower() or str(record.get("archive_identifier", "")).endswith(".omr"):
        score -= 1.2
    return round(score, 4)


def _normalize_allowed_formats(
    formats: Iterable[str] | None,
    *,
    musicxml_only: bool,
) -> frozenset[str]:
    if formats is not None:
        cleaned = {
            f".{f.strip().lstrip('.').lower()}"
            for f in formats
            if f.strip()
        }
        valid = frozenset(cleaned & SUPPORTED_SEARCH_EXTENSIONS)
        if not valid:
            supported = ", ".join(sorted(SUPPORTED_SEARCH_EXTENSIONS))
            raise ValueError(f"No supported score formats in {list(formats)!r}. Supported: {supported}")
        if musicxml_only:
            valid = frozenset(valid & MUSICXML_SEARCH_EXTENSIONS)
        return valid
    return MUSICXML_SEARCH_EXTENSIONS if musicxml_only else SUPPORTED_SEARCH_EXTENSIONS


def search_digital_scores(
    query: str,
    *,
    limit: int = 5,
    musicxml_only: bool = False,
    formats: Iterable[str] | None = None,
    include_remote: bool = True,
    include_local: bool = True,
    cache_dir: Path = Path(".transcribe_cache"),
) -> list[ScoreSearchResult]:
    """Search local and open online score catalogs, ranking MusicXML formats first."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    allowed_exts = _normalize_allowed_formats(formats, musicxml_only=musicxml_only)
    raw_records: list[dict[str, Any]] = []

    def _safe_index(name: str, fetcher: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        try:
            return _load_or_fetch_index(cache_dir, name, fetcher, allow_network=include_remote)
        except Exception:  # noqa: BLE001 — resilient multi-source aggregation
            return []

    tasks: list[Callable[[], list[dict[str, Any]]]] = [
        lambda: _safe_index("musetrainer_library", _fetch_musetrainer_index),
        lambda: _safe_index("openscore_lieder", _fetch_openscore_lieder_index),
        lambda: _safe_index("openscore_stringquartets", _fetch_openscore_sq_index),
        lambda: _safe_index("humdrum_github", _fetch_humdrum_github_index),
    ]
    if include_local:
        tasks.append(lambda: _search_music21_corpus(clean_query))
    if include_remote:
        tasks.append(
            lambda: _search_archive_org(clean_query, allowed_exts, musicxml_only=musicxml_only)
        )

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(fn) for fn in tasks]
        for fut in as_completed(futures):
            raw_records.extend(fut.result())

    ranked: list[ScoreSearchResult] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for rec in raw_records:
        ext = str(rec.get("format_ext", "")).lower()
        if ext not in allowed_exts:
            continue
        score = _score_candidate(rec, clean_query)
        if score <= 0.0:
            continue

        dedup_key = (
            _normalize_text(str(rec.get("title", ""))),
            _normalize_text(str(rec.get("composer") or "")),
            ext,
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        ranked.append(
            ScoreSearchResult(
                title=str(rec["title"]),
                composer=rec.get("composer"),
                format_ext=ext,
                source=str(rec["source"]),
                filename=str(rec["filename"]),
                download_url=rec.get("download_url"),
                local_corpus_path=rec.get("local_corpus_path"),
                archive_identifier=rec.get("archive_identifier"),
                score=score,
            )
        )

    ranked.sort(
        key=lambda r: (
            r.score,
            _FORMAT_PRIORITY.get(r.format_ext, 0.0),
            r.source != "Archive.org",
        ),
        reverse=True,
    )
    return ranked[:limit]


def download_score_result(
    result: ScoreSearchResult,
    *,
    output_path: Path | None = None,
    download_dir: Path = Path("downloads"),
) -> Path:
    """Download (or copy from local music21 corpus) a selected ScoreSearchResult."""
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

    download_url = result.download_url
    if not download_url and result.archive_identifier:
        resolved = _resolve_archive_item(
            {"identifier": result.archive_identifier, "title": result.title, "creator": result.composer},
            SUPPORTED_SEARCH_EXTENSIONS,
        )
        if resolved is not None:
            download_url = resolved.get("download_url")

    if not download_url:
        raise ValueError(f"Search result has no downloadable URL or local corpus path: {result.title}")

    resp = requests.get(download_url, headers={"User-Agent": _USER_AGENT}, timeout=30.0)
    resp.raise_for_status()
    if not resp.content:
        raise RuntimeError(f"Downloaded file was empty: {download_url}")

    tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp")
    tmp_dest.write_bytes(resp.content)
    tmp_dest.replace(dest)
    return dest
