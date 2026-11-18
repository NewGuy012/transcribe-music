"""IMSLP MediaWiki API search, wikitext edition parsing, and solo-part ranking."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

IMSLP_API_URL = "https://imslp.org/api.php"
IMSLP_WIKI_PREFIX = "https://imslp.org/wiki/"

_DIGITAL_EXTS = frozenset(
    {".mxl", ".musicxml", ".xml", ".mid", ".midi", ".ly", ".mscz", ".mscx", ".krn", ".cap", ".capx"}
)
_SOLO_HINTS = ("solo", "complete part", "cello", "violoncello", "violin", "guitar", "piano")
_ENSEMBLE_PART_HINTS = (
    "oboe",
    "flute",
    "clarinet",
    "bassoon",
    "horn",
    "trumpet",
    "trombone",
    "timpani",
    "viola",
    "violin 1",
    "violin 2",
    "violins",
    "basses",
    "cello-bass",
    "cellobass",
    "conductor",
    "full score",
    "partitur",
    "complete score",
)
_CADENZA_HINTS = ("cadenza", "kadenz")


def _extract_balanced_blocks(text: str, marker: str) -> list[str]:
    """Extract top-level MediaWiki template bodies starting with `marker` (respecting nested `{{...}}`)."""
    blocks: list[str] = []
    pos = 0
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            break
        depth = 0
        end = idx
        i = idx
        while i < len(text) - 1:
            two = text[i : i + 2]
            if two == "{{":
                depth += 1
                i += 2
                continue
            if two == "}}":
                depth -= 1
                i += 2
                if depth <= 0:
                    end = i
                    break
                continue
            i += 1
        if end > idx:
            blocks.append(text[idx + len(marker) : end - 2])
            pos = end
        else:
            break
    return blocks


def _split_top_level_fields(body: str) -> dict[str, str]:
    """Split `|Key=Value` fields at brace/bracket depth 0 inside an IMSLP template block."""
    fields: dict[str, str] = {}
    parts: list[str] = []
    buf: list[str] = []
    brace_depth = 0
    bracket_depth = 0
    i = 0
    while i < len(body):
        two = body[i : i + 2]
        if two == "{{":
            brace_depth += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}":
            brace_depth = max(0, brace_depth - 1)
            buf.append(two)
            i += 2
            continue
        if two == "[[":
            bracket_depth += 1
            buf.append(two)
            i += 2
            continue
        if two == "]]":
            bracket_depth = max(0, bracket_depth - 1)
            buf.append(two)
            i += 2
            continue
        ch = body[i]
        if ch == "|" and brace_depth == 0 and bracket_depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))

    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        clean_key = key.strip()
        if clean_key:
            fields[clean_key] = val.strip()
    return fields


def _clean_template_value(raw: str) -> str:
    """Flatten MediaWiki template tags like `{{LinkEd|Kurt|Soldan|1891|1946}}` into readable text."""
    text = raw.strip()
    if not text:
        return ""
    text = re.sub(
        r"\{\{Link(?:Ed|Arr|Comp)\|([^|{}]+)\|([^|{}]+)(?:\|[^{}]*)?\}\}",
        r"\1 \2",
        text,
    )
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return " ".join(text.split())


def _parse_page_title(page_title: str) -> tuple[str, str | None]:
    """Split `'Work Title (Last, First)'` into `(work_title, 'First Last')`."""
    match = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", page_title.strip())
    if not match:
        return page_title.strip(), None
    work_title = match.group(1).strip()
    raw_comp = match.group(2).strip()
    if "," in raw_comp:
        last, first = [p.strip() for p in raw_comp.split(",", 1)]
        composer = f"{first} {last}".strip()
    else:
        composer = raw_comp
    return work_title, composer or None


def _score_imslp_edition(
    filename: str,
    description: str,
    image_type: str,
    query_tokens: list[str],
) -> tuple[float, str]:
    """Rank Solo Part + Typeset editions above scanned full scores or cadenzas."""
    hay = f"{filename} {description} {image_type}".lower()
    ext = Path(filename).suffix.lower()

    bonus = 0.0
    notes: list[str] = []
    if description:
        notes.append(description)

    is_typeset = "typeset" in image_type.lower() or "typeset" in hay
    is_manuscript = "manuscript" in image_type.lower() or "manuscript" in hay
    is_cadenza = any(tok in hay for tok in _CADENZA_HINTS)
    has_solo_hint = any(tok in hay for tok in _SOLO_HINTS)
    has_ensemble_hint = any(tok in hay for tok in _ENSEMBLE_PART_HINTS)

    if ext in _DIGITAL_EXTS:
        bonus += 4.0
    if is_typeset:
        bonus += 2.2
        if "typeset" not in description.lower():
            notes.append("Typeset")
    elif image_type:
        notes.append(image_type)

    if has_solo_hint and not has_ensemble_hint:
        bonus += 2.5
    elif has_solo_hint and ("solo" in hay or "cello" in hay):
        bonus += 2.0
    elif has_ensemble_hint:
        bonus -= 1.2

    if is_manuscript:
        bonus -= 1.8
    if is_cadenza and "cadenza" not in query_tokens:
        bonus -= 2.0

    for tok in query_tokens:
        if tok in {"solo", "cello", "violin", "guitar", "piano"} and tok in hay:
            bonus += 1.0

    return bonus, " · ".join(notes[:3])


def _resolve_imslp_image_urls(
    filenames: list[str],
    *,
    user_agent: str,
    timeout: float,
) -> dict[str, str]:
    """Batch-resolve MediaWiki `File:<name>` pages to direct `/images/...` URLs."""
    if not filenames:
        return {}
    url_map: dict[str, str] = {}
    batch = filenames[:25]
    titles_param = "|".join(f"File:{name}" for name in batch)
    try:
        resp = requests.get(
            IMSLP_API_URL,
            params={
                "action": "query",
                "titles": titles_param,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            },
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return url_map

    pages = data.get("query", {}).get("pages", {})
    for pdata in pages.values():
        if not isinstance(pdata, dict):
            continue
        title = str(pdata.get("title", ""))
        canonical_name = title.removeprefix("File:").strip()
        info_list = pdata.get("imageinfo") or []
        if canonical_name and info_list and isinstance(info_list[0], dict):
            raw_url = str(info_list[0].get("url", "")).strip()
            if raw_url.startswith("//"):
                raw_url = f"https:{raw_url}"
            if raw_url:
                url_map[canonical_name.replace(" ", "_")] = raw_url
                url_map[canonical_name] = raw_url
    return url_map


def search_imslp_works(
    query: str,
    allowed_exts: frozenset[str],
    *,
    user_agent: str,
    timeout: float = 7.0,
    max_works: int = 4,
    max_results_per_work: int = 8,
) -> list[dict[str, Any]]:
    """Search IMSLP via MediaWiki API and extract ranked edition/file entries."""
    clean_query = query.strip()
    if not clean_query:
        return []

    try:
        search_resp = requests.get(
            IMSLP_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "srlimit": 6,
                "format": "json",
            },
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.debug("IMSLP search failed: %s", exc)
        return []

    hits = search_data.get("query", {}).get("search", [])
    titles = [str(h.get("title", "")).strip() for h in hits if h.get("title")]
    if not titles:
        return []

    try:
        pages_resp = requests.get(
            IMSLP_API_URL,
            params={
                "action": "query",
                "titles": "|".join(titles[:max_works]),
                "prop": "revisions",
                "rvprop": "content",
                "redirects": 1,
                "format": "json",
            },
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
        )
        pages_resp.raise_for_status()
        pages_data = pages_resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.debug("IMSLP page fetch failed: %s", exc)
        return []

    query_tokens = [tok for tok in re.findall(r"[a-z0-9]+", clean_query.lower()) if tok]
    candidates: list[dict[str, Any]] = []
    pages = pages_data.get("query", {}).get("pages", {})

    for pdata in pages.values():
        if not isinstance(pdata, dict) or "missing" in pdata:
            continue
        page_title = str(pdata.get("title", "")).strip()
        revs = pdata.get("revisions") or []
        if not page_title or not revs:
            continue
        wikitext = str(revs[0].get("*", ""))
        if not wikitext:
            continue

        work_title, composer = _parse_page_title(page_title)
        work_candidates: list[dict[str, Any]] = []
        file_blocks = _extract_balanced_blocks(wikitext, "{{#fte:imslpfile")

        for block_body in file_blocks:
            fields = _split_top_level_fields(block_body)
            image_type = _clean_template_value(fields.get("Image Type", ""))
            editor = _clean_template_value(fields.get("Editor", ""))
            arranger = _clean_template_value(fields.get("Arranger", ""))
            credit = editor or arranger

            for idx in range(1, 25):
                fname = fields.get(f"File Name {idx}", "").strip()
                if not fname:
                    continue
                ext = Path(fname).suffix.lower()
                if ext not in allowed_exts:
                    continue
                desc = _clean_template_value(fields.get(f"File Description {idx}", ""))
                bonus, note = _score_imslp_edition(fname, desc, image_type, query_tokens)
                if credit and credit.lower() not in note.lower():
                    note = f"{note} · {credit}" if note else credit

                page_file_url = f"{IMSLP_WIKI_PREFIX}File:{quote(fname.replace(' ', '_'))}"
                tag_bits = [bit for bit in (desc, credit) if bit]
                label = f"{work_title} ({', '.join(tag_bits)})" if tag_bits else work_title
                work_candidates.append(
                    {
                        "title": label,
                        "composer": composer,
                        "format_ext": ext,
                        "source": "IMSLP",
                        "filename": fname,
                        "page_url": page_file_url,
                        "edition_note": note or None,
                        "source_bonus": bonus,
                        "_imslp_raw_name": fname,
                    }
                )

        work_candidates.sort(key=lambda r: float(r.get("source_bonus", 0.0)), reverse=True)
        candidates.extend(work_candidates[:max_results_per_work])

    if not candidates:
        return []

    raw_names = [str(c["_imslp_raw_name"]) for c in candidates]
    resolved_urls = _resolve_imslp_image_urls(raw_names, user_agent=user_agent, timeout=timeout)
    for rec in candidates:
        raw_name = str(rec.pop("_imslp_raw_name"))
        rec["download_url"] = (
            resolved_urls.get(raw_name)
            or resolved_urls.get(raw_name.replace(" ", "_"))
            or rec["page_url"]
        )
    return candidates
