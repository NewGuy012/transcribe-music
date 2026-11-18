"""Shared hardened HTTP, bounded archive, path-safety, and catalog config helpers."""

from __future__ import annotations

import json
import os
import stat
import tarfile
import tempfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

USER_AGENT = "transcribe-music/0.2 (+https://github.com/transcribe-music)"
MAX_MEMBER_BYTES = 50 * 1024 * 1024
_MAX_REDIRECTS = 5


def validate_https_url(url: str) -> str:
    """Require an HTTPS URL with a network host before or after redirects."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"Only HTTPS download URLs are permitted, got: {url!r}")
    return url


class _SafeHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate HTTPS on every redirect hop and strip auth headers across hosts."""

    def __init__(self) -> None:
        super().__init__()
        self._redirects = 0

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if self._redirects >= _MAX_REDIRECTS:
            raise urllib.error.URLError(f"Exceeded {_MAX_REDIRECTS} HTTP redirects")
        self._redirects += 1
        validate_https_url(newurl)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        old_host = urlparse(req.full_url).netloc.lower()
        new_host = urlparse(new_req.full_url).netloc.lower()
        if old_host != new_host:
            for hdr_map in (new_req.headers, new_req.unredirected_hdrs):
                for key in [k for k in hdr_map if k.lower() == "authorization"]:
                    del hdr_map[key]
        return new_req


def open_https_response(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    """Open a validated HTTPS URL enforcing HTTPS on every redirect hop."""
    clean_url = validate_https_url(url)
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(clean_url, headers=req_headers)
    opener = urllib.request.build_opener(_SafeHttpsRedirectHandler())
    resp = opener.open(req, timeout=timeout)
    final_url = resp.geturl() if hasattr(resp, "geturl") else getattr(resp, "url", None)
    if isinstance(final_url, str) and final_url:
        validate_https_url(final_url)
    return resp


def _check_content_length(headers: Any, max_bytes: int | None) -> None:
    if max_bytes is None or headers is None:
        return
    raw = headers.get("Content-Length") or headers.get("content-length")
    if raw and str(raw).strip().isdigit() and int(str(raw).strip()) > max_bytes:
        raise ValueError(f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit.")


def http_get_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 12.0,
    max_bytes: int = MAX_MEMBER_BYTES,
) -> bytes:
    """Fetch bounded bytes from a validated HTTPS endpoint."""
    with open_https_response(url, headers=headers, timeout=timeout) as resp:
        _check_content_length(getattr(resp, "headers", None), max_bytes)
        chunks: list[bytes] = []
        total = 0
        while chunk := resp.read(65_536):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit.")
            chunks.append(chunk)
        return b"".join(chunks)


def http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 12.0,
) -> Any:
    merged = {"Accept": "application/json"}
    if headers:
        merged.update(headers)
    return json.loads(http_get_bytes(url, headers=merged, timeout=timeout).decode("utf-8"))


def http_get_text(url: str, *, timeout: float = 12.0) -> str:
    return http_get_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def download_https_stream(
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    max_bytes: int | None = None,
    require_non_empty: bool = False,
) -> Path:
    """Stream an HTTPS download atomically into `dest`, enforcing HTTPS redirects."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with open_https_response(url, headers=headers, timeout=timeout) as resp:
            _check_content_length(getattr(resp, "headers", None), max_bytes)
            with tempfile.NamedTemporaryFile(
                dir=dest.parent,
                prefix=f".{dest.name}.",
                suffix=".tmp",
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                total = 0
                while chunk := resp.read(262_144):
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise ValueError(
                            f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit."
                        )
                    fh.write(chunk)
            if require_non_empty and total == 0:
                raise RuntimeError(f"Downloaded file was empty: {url}")
        os.replace(tmp_path, dest)
        tmp_path = None
        return dest
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(dest: Path, content: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
        os.replace(tmp_path, dest)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def normalize_ascii(text: str) -> str:
    """Fold Unicode accents to ASCII and normalize whitespace/casing."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(ascii_only.lower().split())


def resolve_catalog_dir(cache_dir: Path, kind: str, config_key: str) -> Path:
    default_dir = Path(cache_dir) / kind
    cfg_path = default_dir / "config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            custom = data.get(config_key)
            if isinstance(custom, str) and custom.strip():
                return Path(custom)
        except (OSError, ValueError):
            pass
    return default_dir


def save_catalog_dir(cache_dir: Path, kind: str, config_key: str, target_dir: Path) -> None:
    cfg_path = Path(cache_dir) / kind / "config.json"
    atomic_write_text(cfg_path, json.dumps({config_key: str(target_dir.resolve())}))


def safe_resolve_member_path(base_dir: Path | str, member_rel: str) -> tuple[Path, str, Path]:
    """Validate `member_rel` stays strictly inside `base_dir` and return (base, clean_rel, target)."""
    base_resolved = Path(base_dir).resolve()
    clean_rel = member_rel.removeprefix("./")
    rel_parts = Path(clean_rel).parts
    if (
        not clean_rel
        or "\x00" in clean_rel
        or "\\" in clean_rel
        or Path(clean_rel).is_absolute()
        or ".." in rel_parts
    ):
        raise ValueError(f"Unsafe archive member path outside catalog directory: {member_rel!r}")
    target = (base_resolved / clean_rel).resolve()
    if not target.is_relative_to(base_resolved) or target == base_resolved:
        raise ValueError(f"Unsafe archive member path outside catalog directory: {member_rel!r}")
    return base_resolved, clean_rel, target


def read_bounded_tar_member(
    tar_path: Path,
    clean_rel: str,
    *,
    max_bytes: int = MAX_MEMBER_BYTES,
) -> bytes:
    """Safely extract a single regular file entry from a `.tar.gz` archive within size bounds."""
    with tarfile.open(tar_path, "r:gz") as tf:
        for candidate_name in (clean_rel, f"./{clean_rel}"):
            try:
                member = tf.getmember(candidate_name)
            except KeyError:
                continue
            if not member.isfile():
                raise ValueError(f"Archive entry '{clean_rel}' is not a regular file.")
            if member.size > max_bytes:
                raise ValueError(f"Archive entry '{clean_rel}' exceeds size limit.")
            extracted = tf.extractfile(member)
            if extracted is not None:
                data = extracted.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ValueError(f"Archive entry '{clean_rel}' exceeds size limit.")
                return data
    raise FileNotFoundError(f"Member '{clean_rel}' not found inside {tar_path}")


def is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def read_bounded_zip_member(
    zip_path: Path,
    matcher: Callable[[str], bool],
    *,
    max_bytes: int = MAX_MEMBER_BYTES,
) -> bytes | None:
    """Safely extract the first regular non-symlink ZIP entry matching `matcher`."""
    if not zipfile.is_zipfile(zip_path):
        return None
    with zipfile.ZipFile(zip_path, "r") as zf:
        target = next(
            (
                info
                for info in zf.infolist()
                if not info.is_dir() and matcher(info.filename.replace("\\", "/"))
            ),
            None,
        )
        if target is None:
            return None
        if is_zip_symlink(target):
            raise ValueError(f"Archive entry '{target.filename}' is a symlink, not a regular file.")
        if target.file_size > max_bytes:
            raise ValueError(f"{zip_path.name} entry '{target.filename}' exceeds {max_bytes} byte limit.")
        with zf.open(target, "r") as fh:
            data = fh.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"{zip_path.name} entry '{target.filename}' exceeds {max_bytes} byte limit.")
        return data
