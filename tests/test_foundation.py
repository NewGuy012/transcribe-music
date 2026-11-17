"""Tests for domain models, DuckDB catalog, PDF rasterization, and MusicXML validation."""

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import pymupdf
import pytest

from transcribe_music.export.musicxml import _extract_metrics_and_validate_xml, export_musicxml
from transcribe_music.ingest.pdf import render_pdf_pages
from transcribe_music.models import (
    Instrument,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.storage.catalog import CatalogStore, compute_job_hash


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
    job_hash = compute_job_hash(SourceType.DIGITAL, "synthetic", cfg)
    res = TranscriptionResult(
        job_hash=job_hash,
        source_type=SourceType.DIGITAL,
        source_ref="synthetic.musicxml",
        instrument=Instrument.VIOLIN,
        musicxml_path=out_xml,
        measure_count=metrics.measure_count,
        note_count=metrics.note_count,
        chord_count=metrics.chord_count,
        fingering_count=metrics.fingering_count,
        backend="musicxml",
        selected_part="1: Violin",
        elapsed_sec=0.05,
    )
    store.record_job(res, cfg)

    cached = store.get_cached_job(job_hash)
    assert cached is not None
    assert cached.cached is True
    assert cached.note_count == 8
    assert cached.backend == "musicxml"
    assert cached.selected_part == "1: Violin"
    assert len(store.list_recent_jobs()) == 1

    # Legacy DB schema without backend/selected_part columns migrates cleanly
    import duckdb

    legacy_dir = tmp_path / ".legacy_cache"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(legacy_dir / "catalog.duckdb")) as con:
        con.execute(
            """
            CREATE TABLE jobs (
                job_hash VARCHAR PRIMARY KEY,
                source_type VARCHAR NOT NULL,
                source_ref VARCHAR NOT NULL,
                start_sec DOUBLE,
                end_sec DOUBLE,
                instrument VARCHAR NOT NULL,
                config_json VARCHAR NOT NULL,
                musicxml_path VARCHAR NOT NULL,
                measure_count INTEGER NOT NULL,
                note_count INTEGER NOT NULL,
                chord_count INTEGER NOT NULL,
                fingering_count INTEGER NOT NULL,
                elapsed_sec DOUBLE NOT NULL,
                warnings_json VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
    migrated = CatalogStore(legacy_dir)
    assert migrated.list_recent_jobs() == []


def test_pdf_rasterization(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sheet.pdf"
    doc = pymupdf.open()
    for label in ("Page 1", "Page 2"):
        doc.new_page().insert_text((72, 72), label)
    doc.save(str(pdf_path))
    doc.close()

    pages = render_pdf_pages(pdf_path, tmp_path / "pages", dpi=150)
    assert len(pages) == 2
    assert all(p.exists() and p.suffix == ".png" for p in pages)


def test_export_musicxml_repairs_inexpressible_and_zero_durations(tmp_path: Path) -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    m1 = music21.stream.Measure(number=1)
    m1.append(music21.meter.TimeSignature("4/4"))
    m1.insert(0.0, music21.harmony.ChordSymbol("D"))
    grace = music21.note.Note("C#4", quarterLength=0.5).getGrace()
    m1.insert(0.0, grace)
    m1.insert(0.0, music21.chord.Chord(["D4", "F#4"], quarterLength=2.0))
    m1.insert(2.0, music21.chord.Chord(["D4", "F#4"], quarterLength=0.3333))
    m1.insert(2.3333, music21.chord.Chord(["E4", "G4"], quarterLength=0.4167))
    m1.insert(2.7499, music21.note.Rest(quarterLength=1e-4))
    m1.insert(2.75, music21.note.Note("G4", quarterLength=0.5))
    m1.insert(3.25, music21.note.Note("A4", quarterLength=0.75))
    part.append(m1)
    score.insert(0, part)

    out_xml = tmp_path / "repaired_measured.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.CELLO)
    assert not warnings
    assert metrics.measure_count == 1
    assert metrics.chord_count == 1
    root = ET.parse(out_xml).getroot()
    assert len(root.findall(".//note/grace")) == 1

    # Flat score with zero, sub-minimal, and oversized (> maxima) durations + leading rest gap
    flat_score = music21.stream.Score()
    flat_part = music21.stream.Part()
    flat_part.insert(8.0, music21.note.Note("D3", quarterLength=0.0))
    flat_part.insert(9.0, music21.note.Note("E3", quarterLength=0.001))
    flat_part.insert(10.0, music21.note.Note("F#3", quarterLength=20.0))
    flat_score.insert(0, flat_part)

    flat_xml = tmp_path / "repaired_flat.musicxml"
    flat_metrics, flat_warnings = export_musicxml(flat_score, flat_xml, Instrument.CELLO)
    assert not flat_warnings
    assert flat_metrics.measure_count >= 8
    assert flat_metrics.note_count >= 3

    # Verify Cello gets BassClef (never auto-guessed Treble) and zero duplicate measure rests
    flat_root = ET.parse(flat_xml).getroot()
    clef = flat_root.find(".//measure[@number='1']/attributes/clef")
    assert clef is not None
    assert clef.findtext("sign") == "F"
    assert clef.findtext("line") == "4"
    for m in flat_root.findall(".//measure"):
        assert len(m.findall("./note/rest[@measure='yes']")) <= 1


def test_xml_validator_rejects_duplicate_rests_and_overfull_measures(tmp_path: Path) -> None:
    dup_rest_xml = tmp_path / "dup_rest.musicxml"
    dup_rest_xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Cello</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><rest measure="yes"/><duration>4</duration></note>
      <note><rest measure="yes"/><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate whole-measure rests"):
        _extract_metrics_and_validate_xml(dup_rest_xml)

    overfull_xml = tmp_path / "overfull.musicxml"
    overfull_xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Cello</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration></note>
      <note><pitch><step>D</step><octave>3</octave></pitch><duration>2</duration></note>
    </measure>
  </part>
</score-partwise>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overflows active 4/4 meter"):
        _extract_metrics_and_validate_xml(overfull_xml)


def test_export_accepts_valid_multivoice_measure(tmp_path: Path) -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    m = music21.stream.Measure(number=1)
    m.insert(0, music21.meter.TimeSignature("4/4"))
    v1 = music21.stream.Voice(id="1")
    v1.insert(0.0, music21.note.Note("E4", quarterLength=4.0))
    v2 = music21.stream.Voice(id="2")
    v2.insert(0.0, music21.note.Note("C4", quarterLength=2.0))
    v2.insert(2.0, music21.note.Note("G4", quarterLength=2.0))
    m.insert(0, v1)
    m.insert(0, v2)
    m.insert(2.0, music21.harmony.ChordSymbol("G"))
    part.append(m)
    score.insert(0, part)

    out_xml = tmp_path / "multivoice.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.PIANO)
    assert not warnings
    assert metrics.measure_count == 1
    assert metrics.note_count == 3
    assert metrics.chord_count == 1
