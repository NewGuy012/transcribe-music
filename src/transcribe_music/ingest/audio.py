"""Audio decoding, normalization, and portable FFmpeg discovery."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf


@lru_cache(maxsize=1)
def get_ffmpeg_executable() -> str:
    """Return system ffmpeg or auto-fetch static ARM64/x86_64 binary via static-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    from static_ffmpeg import run as static_run

    ffmpeg_path, _ = static_run.get_or_fetch_platform_executables_else_raise()
    return ffmpeg_path


def normalize_audio_to_wav(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 44100,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> Path:
    """Decode any audio/video file (optionally trimmed) into 16-bit PCM WAV."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Fast zero-subprocess path for uncompressed WAV/FLAC when no resampling/trim is needed
    if (
        start_sec is None
        and end_sec is None
        and input_path.suffix.lower() in {".wav", ".flac"}
    ):
        info = sf.info(str(input_path))
        if info.samplerate == sample_rate and info.channels in (1, 2):
            data, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if 1e-4 < peak < 0.5:
                data = data * (0.85 / peak)
            sf.write(str(output_wav), data, sr, subtype="PCM_16")
            return output_wav

    ffmpeg = get_ffmpeg_executable()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec is not None:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    if end_sec is not None:
        cmd.extend(["-to", f"{end_sec:.3f}"])
    cmd.extend(
        [
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            str(output_wav),
        ]
    )
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio decode failed: {proc.stderr.strip()}")
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced empty WAV for {input_path}")
    return output_wav


def load_mono_audio(wav_path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load WAV as mono float32 array resampled cleanly if needed."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    return mono, sr
