"""Tests for SQLite catalog, PDF rasterization, MusicXML repair, and XML validation."""

from __future__ import annotations

import builtins
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import pytest
from conftest import write_pdf_file

from transcribe_music.export.musicxml import _postprocess_and_validate_xml, export_musicxml
from transcribe_music.models import (
    GuitarStaffMode,
    Instrument,
    SourceType,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcribe_music.score_utils import (
    GuitarTrackMeta,
    format_tuning_label,
    get_guitar_track_meta,
    set_guitar_track_meta,
)
from transcribe_music.storage import open_sqlite
from transcribe_music.storage.catalog import CatalogStore, compute_job_hash
from transcribe_music.transcribe.omr_pdf import render_pdf_pages


def test_export_musicxml_baseline_and_sqlite_catalog(tmp_path: Path) -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    for pitch_name in ["C4", "E4", "G4", "C5", "D5", "F5", "G5", "C6"]:
        part.append(music21.note.Note(pitch_name, quarterLength=1.0))
    score.append(part)

    out_xml = tmp_path / "out.musicxml"
    metrics, warnings = export_musicxml(score, out_xml, Instrument.VIOLIN, title="Test Scale")
    assert not warnings
    assert metrics.measure_count == 2
    assert metrics.note_count == 8

    tree = ET.parse(out_xml)
    assert tree.getroot().tag == "score-partwise"
    assert all(n.find("duration") is not None for n in tree.findall(".//note"))

    cfg = TranscriptionConfig(instrument=Instrument.VIOLIN, cache_dir=tmp_path / ".cache")
    store = CatalogStore(cfg.cache_dir)
    assert store.list_recent_jobs() == []
    assert not store.db_path.exists()

    job_hash = compute_job_hash(SourceType.DIGITAL, "synthetic", cfg)
    cached_artifact = store.artifact_dir(job_hash) / "out.musicxml"
    cached_artifact.write_bytes(out_xml.read_bytes())

    res = TranscriptionResult(
        job_hash=job_hash,
        source_type=SourceType.DIGITAL,
        source_ref="synthetic.musicxml",
        instrument=Instrument.VIOLIN,
        musicxml_path=cached_artifact,
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
    assert cached is not None and cached.cached is True
    assert cached.note_count == 8

    # Tampered DB path outside artifacts_dir must be ignored
    outside_xml = tmp_path / "outside.musicxml"
    outside_xml.write_bytes(out_xml.read_bytes())
    with open_sqlite(store.db_path) as con:
        con.execute("UPDATE jobs SET musicxml_path = ? WHERE job_hash = ?", [str(outside_xml), job_hash])
    assert store.get_cached_job(job_hash) is None

    assert store.clear_cache() == 1
    assert not store.artifacts_dir.exists()


def test_pdf_rasterization_and_missing_extra_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = write_pdf_file(tmp_path / "sheet.pdf")
    pages = render_pdf_pages(pdf_path, tmp_path / "pages", dpi=150)
    assert len(pages) == 2
    assert all(p.exists() and p.suffix == ".png" for p in pages)

    real_import = builtins.__import__

    def _block_pymupdf(name: str, *args: object, **kwargs: object) -> object:
        if name == "pymupdf":
            raise ImportError("No module named 'pymupdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pymupdf)
    with pytest.raises(RuntimeError, match="uv sync --extra pdf"):
        render_pdf_pages(pdf_path, tmp_path / "pages_err", dpi=150)


def test_export_repairs_durations_and_supports_multivoice(tmp_path: Path) -> None:
    score = music21.stream.Score()
    part = music21.stream.Part()
    m1 = music21.stream.Measure(number=1)
    m1.append(music21.meter.TimeSignature("4/4"))
    m1.insert(0.0, music21.harmony.ChordSymbol("D"))
    m1.insert(0.0, music21.note.Note("C#4", quarterLength=0.5).getGrace())
    m1.insert(0.0, music21.chord.Chord(["D4", "F#4"], quarterLength=2.0))
    m1.insert(2.0, music21.chord.Chord(["D4", "F#4"], quarterLength=0.3333))
    m1.insert(2.3333, music21.chord.Chord(["E4", "G4"], quarterLength=0.4167))
    m1.insert(2.7499, music21.note.Rest(quarterLength=1e-4))
    m1.insert(2.75, music21.note.Note("G4", quarterLength=0.5))
    m1.insert(3.25, music21.note.Note("A4", quarterLength=0.75))
    part.append(m1)
    score.insert(0, part)

    metrics, warnings = export_musicxml(score, tmp_path / "repaired.musicxml", Instrument.CELLO)
    assert not warnings
    assert metrics.measure_count == 1
    assert metrics.chord_count == 1
    assert len(ET.parse(tmp_path / "repaired.musicxml").findall(".//note/grace")) == 1

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
    flat_root = ET.parse(flat_xml).getroot()
    assert flat_root.findtext(".//measure[@number='1']/attributes/clef/sign") == "F"
    assert all(len(m.findall("./note/rest[@measure='yes']")) <= 1 for m in flat_root.findall(".//measure"))

    mv_score = music21.stream.Score()
    mv_part = music21.stream.Part()
    mv_m = music21.stream.Measure(number=1)
    mv_m.insert(0, music21.meter.TimeSignature("4/4"))
    v1 = music21.stream.Voice(id="1")
    v1.insert(0.0, music21.note.Note("E4", quarterLength=4.0))
    v2 = music21.stream.Voice(id="2")
    v2.insert(0.0, music21.note.Note("C4", quarterLength=2.0))
    v2.insert(2.0, music21.note.Note("G4", quarterLength=2.0))
    mv_m.insert(0, v1)
    mv_m.insert(0, v2)
    mv_m.insert(2.0, music21.harmony.ChordSymbol("G"))
    mv_part.append(mv_m)
    mv_score.insert(0, mv_part)

    mv_metrics, mv_warnings = export_musicxml(mv_score, tmp_path / "multivoice.musicxml", Instrument.PIANO)
    assert not mv_warnings
    assert mv_metrics.note_count == 3


@pytest.mark.parametrize(
    ("body", "error_match"),
    [
        (
            '<note><rest measure="yes"/><duration>4</duration></note>'
            + '<note><rest measure="yes"/><duration>4</duration></note>',
            "duplicate whole-measure rests",
        ),
        (
            "<note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration></note>"
            + "<note><pitch><step>D</step><octave>3</octave></pitch><duration>2</duration></note>",
            "overflows active 4/4 meter",
        ),
    ],
)
def test_xml_validator_rejects_malformed_measures(
    tmp_path: Path, body: str, error_match: str
) -> None:
    xml_path = tmp_path / "invalid.musicxml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Cello</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      {body}
    </measure>
  </part>
</score-partwise>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=error_match):
        _postprocess_and_validate_xml(xml_path)


def test_export_atomic_rollback_and_guitar_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "existing.musicxml"
    target.write_text("KEEP_INTACT", encoding="utf-8")
    monkeypatch.setattr(
        "transcribe_music.export.musicxml._postprocess_and_validate_xml",
        lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("boom")),
    )
    s = music21.stream.Score([music21.stream.Part([music21.note.Note("C4", quarterLength=4.0)])])
    with pytest.raises(ValueError, match="boom"):
        export_musicxml(s, target, Instrument.VIOLIN)
    assert target.read_text(encoding="utf-8") == "KEEP_INTACT"

    part = music21.stream.Part()
    meta = GuitarTrackMeta(open_string_midi=(64, 59, 55, 50, 45, 38), capo_fret=2)
    set_guitar_track_meta(part, meta)
    assert get_guitar_track_meta(copy.deepcopy(part)) == meta
    assert format_tuning_label(meta.open_string_midi) == "D2-A2-D3-G3-B3-E4"

    h_not = compute_job_hash(SourceType.DIGITAL, "abc", TranscriptionConfig(instrument=Instrument.GUITAR))
    h_tab = compute_job_hash(
        SourceType.DIGITAL,
        "abc",
        TranscriptionConfig(instrument=Instrument.GUITAR, guitar_staff=GuitarStaffMode.TAB),
    )
    h_recomp = compute_job_hash(
        SourceType.DIGITAL,
        "abc",
        TranscriptionConfig(instrument=Instrument.GUITAR, preserve_existing_tab=False),
    )
    assert len({h_not, h_tab, h_recomp}) == 3

    h_w2 = compute_job_hash(SourceType.DIGITAL, "abc", TranscriptionConfig(instrument=Instrument.PIANO))
    h_w1 = compute_job_hash(
        SourceType.DIGITAL,
        "abc",
        TranscriptionConfig(instrument=Instrument.PIANO, chord_window_beats=1.0),
    )
    h_noslash = compute_job_hash(
        SourceType.DIGITAL,
        "abc",
        TranscriptionConfig(instrument=Instrument.PIANO, include_slash_chords=False),
    )
    h_rn = compute_job_hash(
        SourceType.DIGITAL,
        "abc",
        TranscriptionConfig(instrument=Instrument.PIANO, roman_numerals=True),
    )
    assert len({h_w2, h_w1, h_noslash, h_rn}) == 4
    with pytest.raises(ValueError, match="chord_window_beats"):
        TranscriptionConfig(instrument=Instrument.PIANO, chord_window_beats=0.0)
