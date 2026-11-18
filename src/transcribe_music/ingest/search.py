"""Search and download digital symbolic music scores across open repositories."""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import math
import os
import re
import shutil
import time
import unicodedata
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import music21
import requests

from transcribe_music.ingest.digital import DIGITAL_EXTENSIONS
from transcribe_music.ingest.imslp import search_imslp_works
from transcribe_music.models import ScoreSearchResult

logger = logging.getLogger(__name__)

SUPPORTED_SEARCH_EXTENSIONS: frozenset[str] = DIGITAL_EXTENSIONS
PDF_SEARCH_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
ALL_SEARCH_EXTENSIONS: frozenset[str] = SUPPORTED_SEARCH_EXTENSIONS | PDF_SEARCH_EXTENSIONS
MUSICXML_SEARCH_EXTENSIONS: frozenset[str] = frozenset({".musicxml", ".mxl", ".xml"})


class ManualDownloadRequiredError(RuntimeError):
    """Raised when a source gates direct downloads behind browser verification."""

    def __init__(self, result: ScoreSearchResult, url: str, reason: str) -> None:
        self.result = result
        self.url = url
        self.reason = reason
        super().__init__(f"{reason} Open in browser: {url}")

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
    ".pdf": 0.15,
}

_SOURCE_BIAS: dict[str, float] = {
    "OpenScore Orchestra": 1.1,
    "OpenScore Lieder": 1.0,
    "OpenScore StringQuartets": 1.0,
    "When-in-Rome": 0.95,
    "Mutopia Project": 0.75,
    "music21 Corpus": 0.6,
    "Humdrum GitHub": 0.55,
    "IMSLP": 0.45,
    "Archive.org": 0.4,
}

_USER_AGENT = "transcribe-music/0.2 (+https://github.com/transcribe-music)"
_DEFAULT_INDEX_TTL_SEC = 86_400
_HUMDRUM_INDEX_TTL_SEC = 7 * 86_400
_REQUEST_TIMEOUT_SEC = 7.0

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

_HUMDRUM_LIST_URL = (
    "https://raw.githubusercontent.com/humdrum-tools/humdrum-data/master/.lists/LIST.txt"
)
# Curated classical Humdrum repositories with composer metadata and fallback branch/group labels.
_HUMDRUM_CLASSICAL_META: dict[str, tuple[str | None, str]] = {
    "musedata/vivaldi": ("Antonio Vivaldi", "Concertos"),
    "musedata/humdrum-bach-brandenburg": ("Johann Sebastian Bach", "Brandenburg Concertos"),
    "musedata/humdrum-corelli": ("Arcangelo Corelli", "Corelli"),
    "musedata/humdrum-haydn-symphonies": ("Joseph Haydn", "Symphonies"),
    "musedata/humdrum-haydn-quartets": ("Joseph Haydn", "String Quartets"),
    "musedata/humdrum-mozart-quartets": ("Wolfgang Amadeus Mozart", "String Quartets"),
    "craigsapp/beethoven-piano-sonatas": ("Ludwig van Beethoven", "Piano Sonatas"),
    "craigsapp/beethoven-string-quartets": ("Ludwig van Beethoven", "String Quartets"),
    "craigsapp/mozart-piano-sonatas": ("Wolfgang Amadeus Mozart", "Piano Sonatas"),
    "craigsapp/haydn-piano-sonatas": ("Joseph Haydn", "Piano Sonatas"),
    "craigsapp/bach-370-chorales": ("Johann Sebastian Bach", "370 Chorales"),
    "craigsapp/art-of-the-fugue": ("Johann Sebastian Bach", "Art of the Fugue"),
    "craigsapp/bach-musical-offering": ("Johann Sebastian Bach", "Musical Offering"),
    "craigsapp/chopin-mazurkas": ("Frederic Chopin", "Mazurkas"),
    "craigsapp/chopin-preludes": ("Frederic Chopin", "Preludes"),
    "craigsapp/scarlatti-keyboard-sonatas": ("Domenico Scarlatti", "Keyboard Sonatas"),
    "craigsapp/hummel-preludes": ("Johann Nepomuk Hummel", "Preludes"),
    "craigsapp/joplin": ("Scott Joplin", "Ragtime"),
    "humdrum-tools/bach-wtc": ("Johann Sebastian Bach", "Well-Tempered Clavier"),
    "humdrum-tools/inventions": ("Johann Sebastian Bach", "Inventions"),
    "humdrum-tools/schubert-songs": ("Franz Schubert", "Songs"),
    "pl-wnifc/humdrum-chopin-first-editions": ("Frederic Chopin", "First Editions"),
    "pl-wnifc/humdrum-polish-scores": (None, "Polish Heritage Scores"),
}

_MUTOPIA_SEARCH_URL = "https://www.mutopiaproject.org/cgibin/make-table.cgi"
_MUTOPIA_BASE_URL = "https://www.mutopiaproject.org/cgibin/"

_ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
_ARCHIVE_METADATA_URL = "https://archive.org/metadata/{identifier}"
_ARCHIVE_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

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
    if suffix not in ALL_SEARCH_EXTENSIONS:
        cleaned = f"{cleaned}{fallback_ext}"
    return cleaned


def _encode_url_path(path: str) -> str:
    return quote(path.lstrip("/"), safe="/")


def _github_headers() -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"}
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = _REQUEST_TIMEOUT_SEC,
    headers: dict[str, str] | None = None,
) -> Any:
    req_headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    resp = requests.get(
        url,
        params=params,
        headers=req_headers,
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
    tree = _fetch_github_tree(_OPENSCORE_ORCH_TREE_URL)
    records: list[dict[str, Any]] = []
    for node in tree:
        rel_path = str(node.get("path", ""))
        if not rel_path.startswith("data/") or not rel_path.lower().endswith(".mxl"):
            continue
        if rel_path.lower().endswith("_melody.mxl"):
            continue
        parts = Path(rel_path).parts
        # Expected: data/<Composer>/<Work>/<Movement>/<file>.mxl
        composer = _format_composer_name(parts[1]) if len(parts) >= 3 else None
        work = parts[2].replace("_", " ").strip() if len(parts) >= 3 else Path(rel_path).stem
        movement = parts[3].replace("_", " ").strip() if len(parts) >= 5 else ""
        title = f"{work} ({movement})" if movement else work
        filename = Path(rel_path).name
        records.append(
            {
                "title": title,
                "composer": composer,
                "format_ext": ".mxl",
                "source": "OpenScore Orchestra",
                "filename": _safe_filename(filename, fallback_ext=".mxl"),
                "download_url": f"{_OPENSCORE_ORCH_RAW_PREFIX}{_encode_url_path(rel_path)}",
            }
        )
    return records


def _fetch_when_in_rome_index() -> list[dict[str, Any]]:
    tree = _fetch_github_tree(_WHEN_IN_ROME_TREE_URL)
    records: list[dict[str, Any]] = []
    for node in tree:
        rel_path = str(node.get("path", ""))
        if not rel_path.startswith("Corpus/") or not rel_path.lower().endswith(".mxl"):
            continue
        parts = Path(rel_path).parts
        # Expected: Corpus/<Category>/<Composer>/<Work>/<Movement>/score.mxl
        if len(parts) < 5:
            continue
        category = parts[1]
        if category in {"Textbooks", "OpenScore-LiederCorpus"}:
            continue
        composer = _format_composer_name(parts[2])
        work = parts[3].replace("_", " ").strip()
        movement = parts[4].replace("_", " ").strip() if len(parts) >= 6 else ""
        title = f"{work} ({movement})" if movement else work
        cat_label = category.replace("_", " ").strip()
        filename = f"{work}_{movement or 'score'}.mxl"
        records.append(
            {
                "title": title,
                "composer": composer,
                "format_ext": ".mxl",
                "source": "When-in-Rome",
                "filename": _safe_filename(filename, fallback_ext=".mxl"),
                "edition_note": cat_label,
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


def _parse_humdrum_list_repos(text: str) -> list[str]:
    repos: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith(("!", "*")):
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        owner = cols[0].strip()
        repo = cols[1].strip()
        if not owner or not repo:
            continue
        slug = f"{owner}/{repo}"
        if slug not in seen:
            seen.add(slug)
            repos.append(slug)
    return repos


def _parse_hmd_index_records(text: str, repo: str, composer: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_titles: dict[str, str] = {}
    repo_stem = Path(repo).name
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
        title = (
            f"{parent_title} - {desc}"
            if parent_title and desc
            else (desc or parent_title or Path(file_col).stem)
        )
        records.append(
            {
                "title": title,
                "composer": composer,
                "format_ext": ext,
                "source": "Humdrum GitHub",
                "filename": _safe_filename(f"{repo_stem}_{Path(file_col).name}", fallback_ext=ext),
                "download_url": f"https://raw.githubusercontent.com/{repo}/HEAD/{_encode_url_path(file_col)}",
            }
        )
    return records


def _fetch_single_humdrum_repo(repo: str) -> list[dict[str, Any]]:
    composer, group_label = _HUMDRUM_CLASSICAL_META.get(repo, (None, Path(repo).name))
    hmd_url = f"https://raw.githubusercontent.com/{repo}/HEAD/index.hmd"
    try:
        text = _http_get_text(hmd_url)
        return _parse_hmd_index_records(text, repo, composer)
    except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
        logger.debug("No index.hmd for %s: %s", repo, exc)

    if repo not in _HUMDRUM_CLASSICAL_META:
        return []

    tree_url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    try:
        tree = _fetch_github_tree(tree_url)
    except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
        logger.debug("Skipping Humdrum repo %s: %s", repo, exc)
        return []

    records: list[dict[str, Any]] = []
    repo_stem = Path(repo).name
    for node in tree:
        rel_path = str(node.get("path", ""))
        ext = Path(rel_path).suffix.lower()
        if ext not in SUPPORTED_SEARCH_EXTENSIONS:
            continue
        parts = Path(rel_path).parts
        sub_label = ""
        for part in parts[:-1]:
            if part.lower() not in {"kern", "krn", "data", "scores"}:
                sub_label = part.replace("_", " ").replace("-", " ").strip()
                break
        stem_label = " ".join(Path(rel_path).stem.replace("_", " ").replace("-", " ").split())
        title = f"{group_label} - {sub_label} ({stem_label})" if sub_label else f"{group_label} - {stem_label}"
        records.append(
            {
                "title": title,
                "composer": composer,
                "format_ext": ext,
                "source": "Humdrum GitHub",
                "filename": _safe_filename(f"{repo_stem}_{Path(rel_path).name}", fallback_ext=ext),
                "download_url": f"https://raw.githubusercontent.com/{repo}/HEAD/{_encode_url_path(rel_path)}",
            }
        )
    return records


def _fetch_humdrum_github_index() -> list[dict[str, Any]]:
    candidate_repos = list(_HUMDRUM_CLASSICAL_META.keys())
    try:
        list_text = _http_get_text(_HUMDRUM_LIST_URL)
        discovered = _parse_humdrum_list_repos(list_text)
        seen = set(candidate_repos)
        for slug in discovered:
            if slug in seen:
                continue
            owner = slug.split("/", 1)[0]
            if owner in {"craigsapp", "musedata", "humdrum-tools", "pl-wnifc", "polyrhythm-project", "TassoInMusicProject"}:
                seen.add(slug)
                candidate_repos.append(slug)
    except (requests.RequestException, ValueError, OSError) as exc:
        logger.debug("Using fallback classical Humdrum list: %s", exc)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_single_humdrum_repo, repo) for repo in candidate_repos]
        for fut in as_completed(futures):
            records.extend(fut.result())
    return records


class _MutopiaTableParser(HTMLParser):
    """Parse Mutopia Project make-table.cgi result tables into structured entries."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict[str, Any]] = []
        self._in_result_table = False
        self._table_depth = 0
        self._in_row = False
        self._in_cell = False
        self._current_rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell_parts: list[str] = []
        self._current_links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "table":
            cls = attr_map.get("class") or ""
            if "result-table" in cls and not self._in_result_table:
                self._in_result_table = True
                self._table_depth = 1
                self._current_rows = []
                self._current_links = []
                return
            if self._in_result_table:
                self._table_depth += 1
        if not self._in_result_table:
            return
        if tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell_parts = []
        elif tag == "a" and self._in_cell:
            self._current_href = attr_map.get("href")
            self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_result_table:
            return
        if tag == "a" and self._current_href is not None:
            link_label = " ".join("".join(self._current_link_text).split())
            self._current_links.append((self._current_href, link_label))
            self._current_href = None
            self._current_link_text = []
        elif tag in {"td", "th"} and self._in_cell:
            cell_txt = " ".join("".join(self._current_cell_parts).replace("\xa0", " ").split())
            self._current_row.append(cell_txt)
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if self._current_row:
                self._current_rows.append(self._current_row)
            self._in_row = False
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_result_table = False
                self.entries.append(
                    {"rows": self._current_rows, "links": self._current_links}
                )

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell_parts.append(data)
            if self._current_href is not None:
                self._current_link_text.append(data)


def _clean_mutopia_composer(raw: str) -> str | None:
    text = raw.strip()
    if text.lower().startswith("by "):
        text = text[3:].strip()
    text = re.sub(r"\s*\([^)]*\d{3,4}[^)]*\)\s*$", "", text).strip()
    return text or None


def _search_mutopia(query: str, allowed_exts: frozenset[str]) -> list[dict[str, Any]]:
    try:
        resp = requests.get(
            _MUTOPIA_SEARCH_URL,
            params={"searchingfor": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        html_text = resp.content.decode("utf-8", errors="replace")
    except (requests.RequestException, ValueError, OSError):
        return []

    parser = _MutopiaTableParser()
    parser.feed(html_text)

    records: list[dict[str, Any]] = []
    for entry in parser.entries[:12]:
        rows: list[list[str]] = entry["rows"]
        links: list[tuple[str, str]] = entry["links"]
        if not rows or not rows[0]:
            continue

        title = rows[0][0].strip()
        composer = _clean_mutopia_composer(rows[0][1]) if len(rows[0]) > 1 else None
        opus = rows[0][2].strip() if len(rows[0]) > 2 and rows[0][2].strip() else ""
        if opus and opus.lower() not in title.lower():
            title = f"{title} ({opus})"

        instrumentation = rows[1][0].strip() if len(rows) > 1 and rows[1] else ""
        style = rows[1][2].strip() if len(rows) > 1 and len(rows[1]) > 2 else ""
        note_parts = [p for p in (instrumentation, style) if p]
        base_note = " · ".join(note_parts) if note_parts else None

        page_url: str | None = None
        mid_url: str | None = None
        mid_is_zip = False
        ly_url: str | None = None
        pdf_url: str | None = None

        for href, _label in links:
            abs_href = urljoin(_MUTOPIA_BASE_URL, href)
            lower = abs_href.lower()
            if "piece-info.cgi" in lower:
                page_url = abs_href
            elif lower.endswith(".mid") and mid_url is None:
                mid_url = abs_href
                mid_is_zip = False
            elif lower.endswith("-mids.zip") and mid_url is None:
                mid_url = abs_href
                mid_is_zip = True
            elif lower.endswith(".ly") and ly_url is None:
                ly_url = abs_href
            elif lower.endswith(("-a4.pdf", "-let.pdf")) and pdf_url is None:
                pdf_url = abs_href

        candidates = [
            (".mid", mid_url, f"{base_note} (MIDI archive)" if base_note and mid_is_zip else base_note),
            (".ly", ly_url, base_note),
            (".pdf", pdf_url, base_note),
        ]
        for ext, url, note in candidates:
            if not url or ext not in allowed_exts:
                continue
            url_name = Path(url.split("?", 1)[0]).name
            fname_base = url_name.removesuffix("-mids.zip") if mid_is_zip and ext == ".mid" else url_name
            records.append(
                {
                    "title": title,
                    "composer": composer,
                    "format_ext": ext,
                    "source": "Mutopia Project",
                    "filename": _safe_filename(fname_base, fallback_ext=ext),
                    "download_url": url,
                    "page_url": page_url,
                    "edition_note": note,
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


def _search_imslp(query: str, allowed_exts: frozenset[str]) -> list[dict[str, Any]]:
    records = search_imslp_works(
        query,
        allowed_exts,
        user_agent=_USER_AGENT,
        timeout=_REQUEST_TIMEOUT_SEC,
    )
    for rec in records:
        rec["filename"] = _safe_filename(str(rec["filename"]), fallback_ext=str(rec["format_ext"]))
    return records


def _pick_best_archive_file(
    files: list[dict[str, Any]],
    allowed_exts: frozenset[str],
    query_tokens: tuple[str, ...] = (),
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
        if ext == ".pdf":
            if any(tok in lower_name for tok in ("solo", "cello", "violin", "guitar", "piano")):
                prio += 0.04
            if any(tok in lower_name for tok in query_tokens):
                prio += 0.02
        if best is None or prio > best[0]:
            best = (prio, fname, ext)
    if best is None:
        return None
    return best[1], best[2]


def _resolve_archive_item(
    doc: dict[str, Any],
    allowed_exts: frozenset[str],
    query_tokens: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    identifier = str(doc.get("identifier", "")).strip()
    if not identifier:
        return None
    try:
        meta = _http_get_json(_ARCHIVE_METADATA_URL.format(identifier=quote(identifier)), timeout=5.0)
    except (requests.RequestException, ValueError):
        return None

    files = meta.get("files", []) if isinstance(meta, dict) else []
    picked = _pick_best_archive_file(files, allowed_exts, query_tokens=query_tokens)
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
        "page_url": f"https://archive.org/details/{quote(identifier)}",
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
    elif ".pdf" in allowed_exts:
        fmt_clause = '((format:"MusicXML" OR format:"MIDI") OR (collection:imslp AND format:"Text PDF"))'
    else:
        fmt_clause = '(format:"MusicXML" OR format:"MIDI")'
    q_expr = f"({term_expr}) AND {fmt_clause} AND NOT mediatype:audio AND NOT identifier:lp_*"

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
    except (requests.RequestException, ValueError):
        return []

    docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []
    if not docs:
        return []

    # Prioritize docs explicitly advertising MusicXML or MIDI before IMSLP PDFs
    docs.sort(
        key=lambda d: (
            "MusicXML" in (d.get("format") or []),
            "MIDI" in (d.get("format") or []),
        ),
        reverse=True,
    )
    top_docs = docs[:6]
    token_tuple = tuple(tokens)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_resolve_archive_item, doc, allowed_exts, token_tuple)
            for doc in top_docs
        ]
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
    # Strongly prioritize MusicXML (.musicxml/.mxl/.xml) over other digital formats and PDFs
    score += 18.0 * _FORMAT_PRIORITY.get(ext, 0.2)
    score += _SOURCE_BIAS.get(source, 0.0)
    score += float(record.get("source_bonus", 0.0) or 0.0)
    if ".omr" in filename.lower() or str(record.get("archive_identifier", "")).endswith(".omr"):
        score -= 1.2
    return round(score, 4)


def _normalize_allowed_formats(
    formats: Iterable[str] | None,
    *,
    musicxml_only: bool,
    include_pdf: bool = True,
) -> frozenset[str]:
    if formats is not None:
        cleaned = {
            f".{f.strip().lstrip('.').lower()}"
            for f in formats
            if f.strip()
        }
        valid = frozenset(cleaned & ALL_SEARCH_EXTENSIONS)
        if not valid:
            supported = ", ".join(sorted(ALL_SEARCH_EXTENSIONS))
            raise ValueError(f"No supported score formats in {list(formats)!r}. Supported: {supported}")
        if musicxml_only:
            valid = frozenset(valid & MUSICXML_SEARCH_EXTENSIONS)
        return valid
    if musicxml_only:
        return MUSICXML_SEARCH_EXTENSIONS
    return ALL_SEARCH_EXTENSIONS if include_pdf else SUPPORTED_SEARCH_EXTENSIONS


def search_digital_scores(
    query: str,
    *,
    limit: int = 5,
    musicxml_only: bool = False,
    formats: Iterable[str] | None = None,
    include_pdf: bool = True,
    include_remote: bool = True,
    include_local: bool = True,
    cache_dir: Path = Path(".transcribe_cache"),
) -> list[ScoreSearchResult]:
    """Search local and open online score catalogs, ranking MusicXML > digital > solo PDFs."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    allowed_exts = _normalize_allowed_formats(
        formats,
        musicxml_only=musicxml_only,
        include_pdf=include_pdf,
    )
    raw_records: list[dict[str, Any]] = []

    def _safe_index(
        name: str,
        fetcher: Callable[[], list[dict[str, Any]]],
        *,
        ttl_sec: int = _DEFAULT_INDEX_TTL_SEC,
    ) -> list[dict[str, Any]]:
        try:
            return _load_or_fetch_index(
                cache_dir,
                name,
                fetcher,
                allow_network=include_remote,
                ttl_sec=ttl_sec,
            )
        except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
            logger.debug("Skipping search index %s: %s", name, exc)
            return []

    tasks: list[Callable[[], list[dict[str, Any]]]] = [
        lambda: _safe_index("openscore_orchestra", _fetch_openscore_orchestra_index),
        lambda: _safe_index("openscore_lieder", _fetch_openscore_lieder_index),
        lambda: _safe_index("openscore_stringquartets", _fetch_openscore_sq_index),
        lambda: _safe_index("when_in_rome", _fetch_when_in_rome_index),
        lambda: _safe_index("humdrum_github", _fetch_humdrum_github_index, ttl_sec=_HUMDRUM_INDEX_TTL_SEC),
    ]
    if include_local:
        tasks.append(lambda: _search_music21_corpus(clean_query))
    if include_remote:
        tasks.append(lambda: _search_mutopia(clean_query, allowed_exts))
        tasks.append(lambda: _search_imslp(clean_query, allowed_exts))
        tasks.append(
            lambda: _search_archive_org(clean_query, allowed_exts, musicxml_only=musicxml_only)
        )

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(fn) for fn in tasks]
        for fut in as_completed(futures):
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
                archive_identifier=rec.get("archive_identifier"),
                score=score,
            )
        )

    scored_candidates.sort(
        key=lambda r: (
            r.score,
            _FORMAT_PRIORITY.get(r.format_ext, 0.0),
            r.source not in {"Archive.org", "IMSLP"},
        ),
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
            ALL_SEARCH_EXTENSIONS,
        )
        if resolved is not None:
            download_url = resolved.get("download_url")

    if not download_url:
        raise ValueError(f"Search result has no downloadable URL or local corpus path: {result.title}")

    resp = requests.get(download_url, headers={"User-Agent": _USER_AGENT}, timeout=30.0)
    resp.raise_for_status()
    if not resp.content:
        raise RuntimeError(f"Downloaded file was empty: {download_url}")

    payload = resp.content
    final_url_lower = str(getattr(resp, "url", download_url)).lower()
    headers_map = getattr(resp, "headers", None) or {}
    content_type = str(headers_map.get("content-type", "")).lower()
    head_bytes = payload.lstrip()[:32].lower()
    is_html_gate = (
        "friendlytest" in final_url_lower
        or "friendlyredirect" in final_url_lower
        or (
            result.format_ext not in {".xml", ".musicxml"}
            and (
                content_type.startswith("text/html")
                or head_bytes.startswith((b"<!doctype html", b"<html"))
            )
        )
    )
    if is_html_gate:
        raise ManualDownloadRequiredError(
            result=result,
            url=result.page_url or download_url,
            reason=f"{result.source} requires browser verification before downloading this file.",
        )
    if result.format_ext in {".mid", ".midi"} and payload[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            midi_members = sorted(
                name
                for name in zf.namelist()
                if Path(name).suffix.lower() in {".mid", ".midi"} and not name.endswith("/")
            )
            if not midi_members:
                raise RuntimeError(f"MIDI archive contained no .mid files: {download_url}")
            for extra in midi_members[1:]:
                extra_dest = dest.parent / _safe_filename(Path(extra).name, fallback_ext=".mid")
                extra_dest.write_bytes(zf.read(extra))
            payload = zf.read(midi_members[0])

    tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp")
    tmp_dest.write_bytes(payload)
    tmp_dest.replace(dest)
    return dest
