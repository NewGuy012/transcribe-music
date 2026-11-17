"""Tests for domain models, timestamps, DuckDB catalog, ingestion, and MusicXML contract."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import numpy as np
import pytest
import soundfile as sf
from reportlab.pdfgen import canvas

from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.audio import normalize_audio_to_wav
from transcribe_music.ingest.pdf import render_pdf_pages
from transcribe_music.ingest.youtube import extract_youtube_id
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

    with pytest.raises(ValueError):
        TimeRange.from_inputs("2:00", "1:00")


def test_extract_youtube_id() -> None:
    assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


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
    store.record_job(res, cfg, metrics.measures, metrics.notes, metrics.chords)

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
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 700, "Page 1")
    c.showPage()
    c.drawString(100, 700, "Page 2")
    c.showPage()
    c.save()

    pages = render_pdf_pages(pdf_path, tmp_path / "pages", dpi=150)
    assert len(pages) == 2
    assert all(p.exists() and p.suffix == ".png" for p in pages)
