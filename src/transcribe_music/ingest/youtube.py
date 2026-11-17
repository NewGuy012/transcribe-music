"""YouTube audio-only range downloader and slicer using yt-dlp and ffmpeg."""

from __future__ import annotations

import re
from pathlib import Path
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
    time_range: TimeRange,
    output_wav: Path,
    sample_rate: int = 44100,
) -> Path:
    """Download only the requested [start_sec, end_sec] audio slice from YouTube as WAV."""
    import yt_dlp

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    raw_stem = output_wav.parent / "yt_slice_raw"
    ffmpeg_path = get_ffmpeg_executable()

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "ffmpeg_location": str(Path(ffmpeg_path).parent),
        "outtmpl": f"{raw_stem}.%(ext)s",
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(time_range.start_sec, time_range.end_sec)]
        ),
        "force_keyframes_at_cuts": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded_candidates = sorted(output_wav.parent.glob("yt_slice_raw.*"))
    if not downloaded_candidates:
        raise RuntimeError(f"yt-dlp did not produce an audio file for {url}")

    raw_audio = downloaded_candidates[0]
    try:
        normalize_audio_to_wav(
            raw_audio,
            output_wav,
            sample_rate=sample_rate,
            start_sec=0.0,
            end_sec=time_range.duration_sec,
        )
    finally:
        for candidate in downloaded_candidates:
            if candidate != output_wav and candidate.exists():
                candidate.unlink(missing_ok=True)

    return output_wav
