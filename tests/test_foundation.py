"""Tests for domain models, timestamps, DuckDB catalog, ingestion, and MusicXML contract."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Self

import music21
import numpy as np
import pymupdf
import pytest
import soundfile as sf

from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.audio import normalize_audio_to_wav
from transcribe_music.ingest.pdf import render_pdf_pages
from transcribe_music.ingest.youtube import download_youtube_slice, extract_youtube_id
from transcribe_music.models import (
    Instrument,
    SourceType,
    TimeRange,
    TranscriptionConfig,
    TranscriptionResult,
    parse_timestamp,
)
from transcribe_music.storage.catalog import CatalogStore, compute_job_hash


def test_parse_timestamp_formats() -> None:
    assert parse_timestamp(42) == 42.0
    assert parse_timestamp("75.5") == 75.5
    assert parse_timestamp("1:05") == 65.0
    assert parse_timestamp("01:02:03.5") == 3723.5

    tr = TimeRange.from_inputs("0:42", "2:15")
    assert tr.start_sec == 42.0
    assert tr.end_sec == 135.0
    assert tr.duration_sec == 93.0

    tr_full = TimeRange.from_inputs()
    assert tr_full.start_sec == 0.0
    assert tr_full.end_sec is None
    assert tr_full.duration_sec is None

    tr_start_only = TimeRange.from_inputs("0:42", None)
    assert tr_start_only.start_sec == 42.0
    assert tr_start_only.end_sec is None
    assert tr_start_only.duration_sec is None

    tr_end_only = TimeRange.from_inputs(None, "2:15")
    assert tr_end_only.start_sec == 0.0
    assert tr_end_only.end_sec == 135.0
    assert tr_end_only.duration_sec == 135.0

    with pytest.raises(ValueError):
        TimeRange.from_inputs("2:00", "1:00")

    with pytest.raises(ValueError):
        TimeRange.from_inputs(None, "0:00")


def test_extract_youtube_id() -> None:
    assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_download_youtube_slice_full_and_partial_ranges(monkeypatch, tmp_path: Path) -> None:
    import yt_dlp

    captured_opts: list[dict] = []
    sr = 44100
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    sample_wave = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    class _FakeYoutubeDL:
        def __init__(self, opts: dict) -> None:
            self.opts = opts
            captured_opts.append(opts)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def download(self, urls: list[str]) -> None:
            raw_path = Path(self.opts["outtmpl"].replace("%(ext)s", "wav"))
            sf.write(raw_path, sample_wave, sr)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)

    full_wav = tmp_path / "full" / "out.wav"
    download_youtube_slice("https://youtu.be/dQw4w9WgXcQ", None, full_wav, sample_rate=sr)
    assert full_wav.exists()
    assert "download_ranges" not in captured_opts[-1]
    assert "force_keyframes_at_cuts" not in captured_opts[-1]
    full_data, _ = sf.read(full_wav)
    assert len(full_data) == len(sample_wave)

    zero_to_end_wav = tmp_path / "zero_end" / "out.wav"
    download_youtube_slice(
        "https://youtu.be/dQw4w9WgXcQ",
        TimeRange(start_sec=0.0, end_sec=None),
        zero_to_end_wav,
        sample_rate=sr,
    )
    assert "download_ranges" not in captured_opts[-1]

    start_only_wav = tmp_path / "start_only" / "out.wav"
    download_youtube_slice(
        "https://youtu.be/dQw4w9WgXcQ",
        TimeRange.from_inputs("0:01", None),
        start_only_wav,
        sample_rate=sr,
    )
    assert "download_ranges" in captured_opts[-1]
    assert captured_opts[-1]["download_ranges"].ranges == [(1.0, float("inf"))]

    sliced_wav = tmp_path / "sliced" / "out.wav"
    download_youtube_slice(
        "https://youtu.be/dQw4w9WgXcQ",
        TimeRange.from_inputs("0:00", "0:01"),
        sliced_wav,
        sample_rate=sr,
    )
    assert "download_ranges" in captured_opts[-1]
    assert captured_opts[-1]["download_ranges"].ranges == [(0.0, 1.0)]
    sliced_data, _ = sf.read(sliced_wav)
    assert len(sliced_data) == sr


def test_export_musicxml_baseline_and_duckdb_catalog(tmp_path: Path) -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    for pitch_name in ["C4", "E4", "G4", "C5", "D5", "F5", "G5", "C6"]:
        part.append(music21.note.Note(pitch_name, quarterLength=1.0))
    score.append(part)

    out_xml = tmp_path / "out.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.VIOLIN, title="Test Scale")
    assert out_xml.exists()
    assert not warnings
    assert metrics.measure_count == 2
    assert metrics.note_count == 8

    tree = ET.parse(out_xml)
    assert tree.getroot().tag == "score-partwise"
    assert len(tree.findall(".//measure")) == 2
    assert all(n.find("duration") is not None for n in tree.findall(".//note"))

    cfg = TranscriptionConfig(instrument=Instrument.VIOLIN, cache_dir=tmp_path / ".cache")
    store = CatalogStore(cfg.cache_dir)
    job_hash = compute_job_hash(SourceType.AUDIO, "synthetic", cfg)
    res = TranscriptionResult(
        job_hash=job_hash,
        source_type=SourceType.AUDIO,
        source_ref="synthetic.wav",
        instrument=Instrument.VIOLIN,
        musicxml_path=out_xml,
        measure_count=metrics.measure_count,
        note_count=metrics.note_count,
        chord_count=metrics.chord_count,
        fingering_count=metrics.fingering_count,
        elapsed_sec=0.12,
    )
    store.record_job(res, cfg)

    cached = store.get_cached_job(job_hash)
    assert cached is not None
    assert cached.cached is True
    assert cached.note_count == 8
    assert len(store.list_recent_jobs()) == 1


def test_audio_normalization_and_pdf_rasterization(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    raw_wav = tmp_path / "tone.wav"
    sf.write(raw_wav, wave, sr)

    norm_wav = tmp_path / "norm.wav"
    normalize_audio_to_wav(raw_wav, norm_wav, sample_rate=44100)
    assert norm_wav.exists()
    data, out_sr = sf.read(norm_wav)
    assert out_sr == 44100
    assert len(data) == int(sr * 0.5)

    pdf_path = tmp_path / "sheet.pdf"
    doc = pymupdf.open()
    for label in ("Page 1", "Page 2"):
        doc.new_page().insert_text((72, 72), label)
    doc.save(str(pdf_path))
    doc.close()

    pages = render_pdf_pages(pdf_path, tmp_path / "pages", dpi=150)
    assert len(pages) == 2
    assert all(p.exists() and p.suffix == ".png" for p in pages)
