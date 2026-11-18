"""End-to-end integration tests across digital formats, PDF fallback, CLI, Verovio, and DuckDB."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import guitarpro
import guitarpro.models as gm
import music21
import pymupdf
import pytest
from typer.testing import CliRunner

from transcribe_music import (
    Instrument,
    TranscriptionConfig,
    convert_music,
    inspect_score,
    transcribe_pdf,
    transcribe_score,
)
from transcribe_music.cli import app
from transcribe_music.storage.catalog import CatalogStore


def _write_sample_concerto_mxl(path: Path) -> Path:
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title="Haydn D Major Sample")

    orch = music21.stream.Part(id="Strings")
    orch.partName = "Orchestra Strings"
    orch.insert(0, music21.instrument.StringInstrument())
    m1_o = music21.stream.Measure(number=1)
    m1_o.insert(0, music21.meter.TimeSignature("4/4"))
    m1_o.insert(0, music21.key.KeySignature(2))
    m1_o.insert(0, music21.chord.Chord(["D4", "F#4", "A4"], quarterLength=4.0))
    m2_o = music21.stream.Measure(number=2)
    m2_o.insert(0, music21.chord.Chord(["A3", "C#4", "E4", "G4"], quarterLength=4.0))
    orch.append(m1_o)
    orch.append(m2_o)

    cello = music21.stream.Part(id="SoloCello")
    cello.partName = "Solo Violoncello"
    cello.insert(0, music21.instrument.Violoncello())
    m1_c = music21.stream.Measure(number=1)
    m1_c.insert(0, music21.clef.BassClef())
    m1_c.insert(0, music21.meter.TimeSignature("4/4"))
    m1_c.insert(0, music21.key.KeySignature(2))
    m1_c.insert(0, music21.note.Rest(quarterLength=4.0))
    m2_c = music21.stream.Measure(number=2)
    m2_c.insert(0, music21.clef.TenorClef())
    for p in ("A3", "C#4", "E4", "A4"):
        m2_c.append(music21.note.Note(p, quarterLength=1.0))
    cello.append(m1_c)
    cello.append(m2_c)

    score.insert(0, orch)
    score.insert(0, cello)
    score.write("mxl", fp=str(path))
    return path


def _write_sample_gp5(path: Path) -> Path:
    song = gm.Song(title="Guitar Riff", tempo=120)
    tr = song.tracks[0]
    tr.name = "Lead Guitar"
    tr.channel.instrument = 25
    v = tr.measures[0].voices[0]
    for fret in (0, 3, 5, 7):
        b = gm.Beat(v, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
        b.notes.append(gm.Note(b, value=fret, string=6, type=gm.NoteType.normal))
        v.beats.append(b)
    guitarpro.write(song, str(path))
    return path


def test_end_to_end_mxl_concerto_verovio_and_duckdb_cache(tmp_path: Path) -> None:
    mxl_path = _write_sample_concerto_mxl(tmp_path / "haydn_sample.mxl")

    insp = inspect_score(mxl_path)
    assert len(insp.parts) == 2

    out_xml = tmp_path / "haydn_cello.musicxml"
    cfg = TranscriptionConfig(
        instrument=Instrument.CELLO,
        cache_dir=tmp_path / ".cache",
    )
    res1 = convert_music(mxl_path, Instrument.CELLO, output_path=out_xml, config=cfg)
    assert out_xml.exists()
    assert res1.cached is False
    assert res1.backend == "musicxml"
    assert res1.selected_part == "2: Solo Violoncello"
    assert res1.measure_count == 2
    assert res1.note_count == 4
    assert res1.chord_count == 2  # D in bar 1 from orchestra + A7 in bar 2!
    assert res1.fingering_count == 4

    root = ET.parse(out_xml).getroot()
    clefs = [c.findtext("sign") for c in root.findall(".//attributes/clef")]
    assert clefs == ["F", "C"]  # Bass clef in m.1, authored Tenor clef preserved in m.2!
    harmonies = [h.findtext("root/root-step") for h in root.findall(".//harmony")]
    assert harmonies == ["D", "A"]

    out_xml_2 = tmp_path / "haydn_cello_cached.musicxml"
    res2 = transcribe_score(mxl_path, Instrument.CELLO, output_path=out_xml_2, config=cfg)
    assert res2.cached is True
    assert res2.job_hash == res1.job_hash
    assert res2.selected_part == "2: Solo Violoncello"
    assert out_xml_2.exists()


def test_end_to_end_gp5_and_krn_conversion(tmp_path: Path) -> None:
    gp5_path = _write_sample_gp5(tmp_path / "riff.gp5")
    cfg_gtr = TranscriptionConfig(instrument=Instrument.GUITAR, use_cache=False)
    res_gtr = convert_music(gp5_path, Instrument.GUITAR, output_path=tmp_path / "riff.musicxml", config=cfg_gtr)
    assert res_gtr.backend == "pyguitarpro"
    assert res_gtr.note_count == 4
    assert res_gtr.fingering_count == 4
    root = ET.parse(res_gtr.musicxml_path).getroot()
    assert root.findtext(".//attributes/clef/clef-octave-change") == "-1"

    krn_path = tmp_path / "etude.krn"
    krn_path.write_text(
        "**kern\n*clefG2\n*M4/4\n4g\n4a\n4b\n4cc\n==\n*-\n",
        encoding="utf-8",
    )
    cfg_vln = TranscriptionConfig(instrument=Instrument.VIOLIN, use_cache=False)
    res_vln = convert_music(krn_path, Instrument.VIOLIN, output_path=tmp_path / "etude.musicxml", config=cfg_vln)
    assert res_vln.backend == "converter21_humdrum"
    assert res_vln.note_count == 4
    assert res_vln.fingering_count == 4


def test_end_to_end_pdf_fallback_omr_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "guitar_piece.pdf"
    doc = pymupdf.open()
    for label in ("Page 1", "Page 2"):
        doc.new_page().insert_text((72, 72), label)
    doc.save(str(pdf_path))
    doc.close()

    def _fake_homr_batch(image_paths: list[Path], _work_dir: Path) -> list[Path]:
        xml_outs: list[Path] = []
        for idx, image_path in enumerate(image_paths):
            s = music21.stream.Score()
            p = music21.stream.Part()
            p.insert(0, music21.clef.Treble8vbClef())
            pitches = ["C4", "E4", "G4", "C5"] if idx == 0 else ["G3", "B3", "D4", "G4"]
            for pitch in pitches:
                p.append(music21.note.Note(pitch, quarterLength=1.0))
            p.makeMeasures(inPlace=True)
            s.insert(0, p)
            xml_out = image_path.with_suffix(".musicxml")
            s.write("musicxml", fp=str(xml_out))
            xml_outs.append(xml_out)
        return xml_outs

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._run_homr_batch", _fake_homr_batch)

    out_xml = tmp_path / "guitar_pdf.musicxml"
    cfg = TranscriptionConfig(
        instrument=Instrument.GUITAR,
        cache_dir=tmp_path / ".cache",
        pdf_dpi=150,
    )
    res = transcribe_pdf(pdf_path, Instrument.GUITAR, output_path=out_xml, config=cfg)
    assert out_xml.exists()
    assert res.backend == "homr"
    assert res.measure_count == 2
    assert res.note_count == 8
    assert res.chord_count >= 1
    assert res.fingering_count == 8

    root = ET.parse(out_xml).getroot()
    assert len(root.findall(".//technical/string")) == 8
    assert len(root.findall(".//technical/fret")) == 8


def test_cli_convert_score_inspect_and_cache_commands(tmp_path: Path) -> None:
    mxl_path = _write_sample_concerto_mxl(tmp_path / "cli_concerto.mxl")
    cache_dir = tmp_path / ".cli_cache"
    runner = CliRunner()

    # inspect command prints part table
    insp_res = runner.invoke(app, ["inspect", str(mxl_path)])
    assert insp_res.exit_code == 0, insp_res.output
    assert "Solo Violoncello" in insp_res.output
    assert "Orchestra Strings" in insp_res.output

    # --no-cache leaves no DuckDB file behind
    nocache_out = tmp_path / "nocache.musicxml"
    nc_res = runner.invoke(
        app,
        [
            "score",
            str(mxl_path),
            "--instrument",
            "cello",
            "--no-cache",
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(nocache_out),
        ],
    )
    assert nc_res.exit_code == 0, nc_res.output
    assert "musicxml" in nc_res.output
    assert nocache_out.exists()
    assert not (cache_dir / "catalog.duckdb").exists()

    # Unified convert command with cache enabled
    conv_out = tmp_path / "cli_conv.musicxml"
    conv_res = runner.invoke(
        app,
        [
            "convert",
            str(mxl_path),
            "--instrument",
            "cello",
            "--part",
            "solo",
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(conv_out),
        ],
    )
    assert conv_res.exit_code == 0, conv_res.output
    assert conv_out.exists()

    list_res = runner.invoke(app, ["cache", "list", "--cache-dir", str(cache_dir)])
    assert list_res.exit_code == 0, list_res.output
    assert "cello" in list_res.output
    cached_rows = CatalogStore(cache_dir).list_recent_jobs()
    assert len(cached_rows) == 1
    assert cached_rows[0]["backend"] == "musicxml"
    assert cached_rows[0]["selected_part"] == "2: Solo Violoncello"

    clear_res = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir)])
    assert clear_res.exit_code == 0, clear_res.output
    assert "Cleared 1 job records" in clear_res.output
