"""Shared hardened HTTP, bounded archive, path-safety, and atomic I/O helpers."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

USER_AGENT = "transcribe-music/0.2 (+https://github.com/transcribe-music)"
MAX_MEMBER_BYTES = 50 * 1024 * 1024
_MAX_REDIRECTS = 5


def _host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower(), parsed.port or 443


def validate_https_url(url: str) -> str:
    """Require a public HTTPS URL without whitespace, credentials, or local/private hosts."""
    if not url or any(ord(ch) < 32 or ch.isspace() for ch in url):
        raise ValueError(f"Only clean HTTPS download URLs are permitted, got: {url!r}")
    parsed = urlparse(url)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid port in download URL: {url!r}") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"Only HTTPS download URLs are permitted, got: {url!r}")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError(f"Local download host is not permitted: {host!r}")
    try:
        if not ipaddress.ip_address(host).is_global:
            raise ValueError(f"Non-public IP download host is not permitted: {host!r}")
    except ValueError as exc:
        if "not permitted" in str(exc):
            raise
    return url


class _SafeHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate HTTPS on every redirect hop and strip auth headers across origins."""

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
        if _host_port(req.full_url) != _host_port(new_req.full_url):
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


@contextmanager
def atomic_write_path(dest: Path) -> Iterator[Path]:
    """Yield a closed same-directory temp path and atomically replace `dest` on clean exit."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,
        prefix=f".{dest.name}.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path = Path(fh.name)
    try:
        yield tmp_path
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_bytes(dest: Path, data: bytes) -> Path:
    with atomic_write_path(dest) as tmp_path:
        tmp_path.write_bytes(data)
    return Path(dest)


def atomic_write_text(dest: Path, content: str) -> None:
    with atomic_write_path(dest) as tmp_path:
        tmp_path.write_text(content, encoding="utf-8")


def copy_file_safe(src: Path, dst: Path) -> Path:
    """Copy a regular file to `dst`, skipping same-file overwrites."""
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


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
    with open_https_response(url, headers=headers, timeout=timeout) as resp:
        _check_content_length(getattr(resp, "headers", None), max_bytes)
        with atomic_write_path(dest) as tmp_path:
            total = 0
            with tmp_path.open("wb") as fh:
                while chunk := resp.read(262_144):
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise ValueError(
                            f"Download exceeds {max_bytes // (1024 * 1024)} MiB safety limit."
                        )
                    fh.write(chunk)
            if require_non_empty and total == 0:
                raise RuntimeError(f"Downloaded file was empty: {url}")
    return dest


def normalize_ascii(text: str) -> str:
    """Fold Unicode accents to ASCII and normalize whitespace/casing."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(ascii_only.lower().split())


def tokenize_ascii_query(text: str) -> list[str]:
    """Fold accents and split a search query into alphanumeric tokens."""
    return re.sub(r"[^a-z0-9]+", " ", normalize_ascii(text)).split()


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


def sanitize_relative_member(member_rel: str) -> str:
    """Validate and normalize a relative archive member path (rejecting traversal/absolutes)."""
    clean_rel = member_rel
    while clean_rel.startswith("./"):
        clean_rel = clean_rel[2:]
    rel_parts = Path(clean_rel).parts
    if (
        not clean_rel
        or clean_rel == "."
        or "\x00" in clean_rel
        or "\\" in clean_rel
        or Path(clean_rel).is_absolute()
        or ".." in rel_parts
    ):
        raise ValueError(f"Unsafe archive member path outside catalog directory: {member_rel!r}")
    return clean_rel


def safe_resolve_member_path(base_dir: Path | str, member_rel: str) -> tuple[Path, str, Path]:
    """Validate `member_rel` stays strictly inside `base_dir` and return (base, clean_rel, target)."""
    base_resolved = Path(base_dir).resolve()
    clean_rel = sanitize_relative_member(member_rel)
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


def check_zip_safety(zip_path: Path, *, max_bytes: int | None = None) -> None:
    """Reject ZIP files containing symlink entries or uncompressed totals exceeding `max_bytes`."""
    if not zipfile.is_zipfile(zip_path):
        return
    limit = MAX_MEMBER_BYTES if max_bytes is None else max_bytes
    total = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if is_zip_symlink(info):
                raise ValueError(f"Archive entry '{info.filename}' is a symlink, not a regular file.")
            if info.file_size > limit or (total := total + info.file_size) > limit:
                raise ValueError(f"{zip_path.name} exceeds {limit} byte uncompressed limit.")


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
