"""YouTube audio-only downloader and optional range slicer using yt-dlp and ffmpeg."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from transcribe_music.ingest.audio import get_ffmpeg_executable, normalize_audio_to_wav
from transcribe_music.models import TimeRange

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_youtube_id(url: str) -> str:
    """Extract canonical 11-character YouTube video ID for stable cache keys."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")

    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
        if _YOUTUBE_ID_RE.match(candidate):
            return candidate

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
            if _YOUTUBE_ID_RE.match(vid):
                return vid
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"} and _YOUTUBE_ID_RE.match(parts[1]):
            return parts[1]

    # Fallback: use normalized URL if non-standard link
    return url.strip()


def download_youtube_slice(
    url: str,
    time_range: TimeRange | None,
    output_wav: Path,
    sample_rate: int = 44100,
) -> Path:
    """Download YouTube audio (whole video by default, or [start_sec, end_sec] slice) as WAV."""
    import yt_dlp

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    raw_stem = output_wav.parent / "yt_slice_raw"
    for stale in output_wav.parent.glob("yt_slice_raw.*"):
        if stale != output_wav:
            stale.unlink(missing_ok=True)
    ffmpeg_path = get_ffmpeg_executable()

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "ffmpeg_location": str(Path(ffmpeg_path).parent),
        "outtmpl": f"{raw_stem}.%(ext)s",
    }

    has_slice = time_range is not None and (
        time_range.start_sec > 0.0 or time_range.end_sec is not None
    )
    if has_slice:
        assert time_range is not None
        end_cut = time_range.end_sec if time_range.end_sec is not None else float("inf")
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(time_range.start_sec, end_cut)]
        )
        ydl_opts["force_keyframes_at_cuts"] = False

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded_candidates = sorted(output_wav.parent.glob("yt_slice_raw.*"))
    if not downloaded_candidates:
        raise RuntimeError(f"yt-dlp did not produce an audio file for {url}")

    raw_audio = downloaded_candidates[0]
    try:
        normalize_start = (
            0.0 if (time_range is not None and time_range.duration_sec is not None) else None
        )
        normalize_end = time_range.duration_sec if time_range is not None else None
        normalize_audio_to_wav(
            raw_audio,
            output_wav,
            sample_rate=sample_rate,
            start_sec=normalize_start,
            end_sec=normalize_end,
        )
    finally:
        for candidate in downloaded_candidates:
            if candidate != output_wav and candidate.exists():
                candidate.unlink(missing_ok=True)

    return output_wav
