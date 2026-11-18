"""End-to-end integration tests across digital formats, PDF fallback, CLI, and DuckDB cache."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import pytest
from conftest import (
    make_concerto_score,
    write_ascii_tab_file,
    write_gp5_file,
    write_gpif_file,
    write_pdf_file,
)
from typer.testing import CliRunner

from transcribe_music import (
    GuitarStaffMode,
    Instrument,
    TranscriptionConfig,
    convert_music,
    inspect_score,
)
from transcribe_music.cli import app
from transcribe_music.storage.catalog import CatalogStore


def test_end_to_end_mxl_concerto_and_canonical_cache(tmp_path: Path) -> None:
    mxl_path = tmp_path / "haydn_sample.mxl"
    make_concerto_score().write("mxl", fp=str(mxl_path))

    insp = inspect_score(mxl_path)
    assert len(insp.parts) == 3

    out_xml = tmp_path / "haydn_cello.musicxml"
    cfg = TranscriptionConfig(instrument=Instrument.CELLO, cache_dir=tmp_path / ".cache")
    res1 = convert_music(mxl_path, Instrument.CELLO, output_path=out_xml, config=cfg)
    assert res1.cached is False
    assert res1.backend == "musicxml"
    assert res1.selected_part == "2: Solo Violoncello"
    assert res1.note_count == 4
    assert res1.chord_count == 2
    assert res1.fingering_count == 4

    root = ET.parse(out_xml).getroot()
    assert [c.findtext("sign") for c in root.findall(".//attributes/clef")] == ["F", "C"]
    assert [h.findtext("root/root-step") for h in root.findall(".//harmony")] == ["D", "A"]

    # Deleting user output still serves intact canonical cache artifact
    out_xml.unlink()
    out_xml_2 = tmp_path / "haydn_cello_cached.musicxml"
    res2 = convert_music(mxl_path, Instrument.CELLO, output_path=out_xml_2, config=cfg)
    assert res2.cached is True
    assert res2.job_hash == res1.job_hash
    assert out_xml_2.exists()


def test_end_to_end_gp5_krn_and_pdf_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gp5_path = write_gp5_file(tmp_path / "riff.gp5", notes=((6, 0), (6, 3), (6, 5), (6, 7)))
    res_gtr = convert_music(
        gp5_path,
        Instrument.GUITAR,
        output_path=tmp_path / "riff.musicxml",
        config=TranscriptionConfig(instrument=Instrument.GUITAR, use_cache=False),
    )
    assert res_gtr.backend == "pyguitarpro"
    assert res_gtr.fingering_count == 4
    assert ET.parse(res_gtr.musicxml_path).findtext(".//clef-octave-change") == "-1"

    krn_path = tmp_path / "etude.krn"
    krn_path.write_text("**kern\n*clefG2\n*M4/4\n4g\n4a\n4b\n4cc\n==\n*-\n", encoding="utf-8")
    res_vln = convert_music(
        krn_path,
        Instrument.VIOLIN,
        output_path=tmp_path / "etude.musicxml",
        config=TranscriptionConfig(instrument=Instrument.VIOLIN, use_cache=False),
    )
    assert res_vln.backend == "converter21_humdrum"
    assert res_vln.fingering_count == 4

    pdf_path = write_pdf_file(tmp_path / "guitar_piece.pdf")

    def _fake_homr_batch(image_paths: list[Path], _work_dir: Path) -> list[Path]:
        outs: list[Path] = []
        for idx, image_path in enumerate(image_paths):
            s = music21.stream.Score()
            p = music21.stream.Part()
            p.insert(0, music21.clef.Treble8vbClef())
            for pitch in ("C4", "E4", "G4", "C5") if idx == 0 else ("G3", "B3", "D4", "G4"):
                p.append(music21.note.Note(pitch, quarterLength=1.0))
            p.makeMeasures(inPlace=True)
            s.insert(0, p)
            xml_out = image_path.with_suffix(".musicxml")
            s.write("musicxml", fp=str(xml_out))
            outs.append(xml_out)
        return outs

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._run_homr_batch", _fake_homr_batch)

    pdf_out = tmp_path / "guitar_pdf.musicxml"
    res_pdf = convert_music(
        pdf_path,
        Instrument.GUITAR,
        output_path=pdf_out,
        config=TranscriptionConfig(
            instrument=Instrument.GUITAR, cache_dir=tmp_path / ".cache", pdf_dpi=150
        ),
    )
    assert res_pdf.backend == "homr"
    assert res_pdf.note_count == 8
    assert len(ET.parse(pdf_out).findall(".//technical/fret")) == 8


def test_cli_convert_inspect_and_cache_commands(tmp_path: Path) -> None:
    mxl_path = tmp_path / "cli_concerto.mxl"
    make_concerto_score().write("mxl", fp=str(mxl_path))
    cache_dir = tmp_path / ".cli_cache"
    runner = CliRunner()

    insp_res = runner.invoke(app, ["inspect", str(mxl_path)])
    assert insp_res.exit_code == 0, insp_res.output
    assert "Solo Violoncello" in insp_res.output

    nocache_out = tmp_path / "nocache.musicxml"
    nc_res = runner.invoke(
        app,
        [
            "convert",
            str(mxl_path),
            "-i",
            "cello",
            "--no-cache",
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(nocache_out),
        ],
    )
    assert nc_res.exit_code == 0, nc_res.output
    assert nocache_out.exists()
    assert not (cache_dir / "catalog.sqlite").exists()

    conv_out = tmp_path / "cli_conv.musicxml"
    conv_res = runner.invoke(
        app,
        [
            "convert",
            str(mxl_path),
            "-i",
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

    list_res = runner.invoke(app, ["cache", "list", "--cache-dir", str(cache_dir)])
    assert list_res.exit_code == 0, list_res.output
    assert "cello" in list_res.output
    assert len(CatalogStore(cache_dir).list_recent_jobs()) == 1

    clear_res = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir)])
    assert clear_res.exit_code == 0, clear_res.output
    assert "Cleared 1 job records" in clear_res.output


def test_end_to_end_guitar_staff_modes_gpif_ascii_tab_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. GP5 with Drop-D tuning and Capo 2 exported in notation, tab, and both modes
    gp5_path = write_gp5_file(
        tmp_path / "dropd.gp5",
        notes=((6, 0), (6, 3), (6, 5), (6, 7)),
        tuning=(64, 59, 55, 50, 45, 38),
        capo=2,
    )
    cache_dir = tmp_path / ".gtr_cache"

    res_not = convert_music(
        gp5_path,
        Instrument.GUITAR,
        output_path=tmp_path / "dropd_not.musicxml",
        config=TranscriptionConfig(
            instrument=Instrument.GUITAR,
            guitar_staff=GuitarStaffMode.NOTATION,
            cache_dir=cache_dir,
        ),
    )
    res_tab = convert_music(
        gp5_path,
        Instrument.GUITAR,
        output_path=tmp_path / "dropd_tab.musicxml",
        config=TranscriptionConfig(
            instrument=Instrument.GUITAR,
            guitar_staff=GuitarStaffMode.TAB,
            cache_dir=cache_dir,
        ),
    )
    res_both = convert_music(
        gp5_path,
        Instrument.GUITAR,
        output_path=tmp_path / "dropd_both.musicxml",
        config=TranscriptionConfig(
            instrument=Instrument.GUITAR,
            guitar_staff=GuitarStaffMode.BOTH,
            cache_dir=cache_dir,
        ),
    )

    assert len({res_not.job_hash, res_tab.job_hash, res_both.job_hash}) == 3
    assert res_not.note_count == res_tab.note_count == res_both.note_count == 4
    assert res_not.fingering_count == res_both.fingering_count == 4

    # Preserved authored TAB keeps all 4 notes on string 6 (frets 0, 3, 5, 7)
    root_not = ET.parse(res_not.musicxml_path).getroot()
    assert [s.text for s in root_not.findall(".//technical/string")] == ["6", "6", "6", "6"]
    assert [f.text for f in root_not.findall(".//technical/fret")] == ["0", "3", "5", "7"]
    assert root_not.findtext(".//staff-details/capo") == "2"
    assert root_not.findtext(".//staff-details/staff-tuning[@line='1']/tuning-step") == "D"

    root_tab = ET.parse(res_tab.musicxml_path).getroot()
    assert root_tab.findtext(".//attributes/clef/sign") == "TAB"
    assert root_tab.find(".//attributes/transpose") is None
    # Note 1 is D2+capo2 = E2 sounding (octave 2 in TAB vs octave 3 in written notation)
    assert root_tab.findtext(".//measure[@number='1']/note[1]/pitch/octave") == "2"

    root_both = ET.parse(res_both.musicxml_path).getroot()
    assert root_both.findtext(".//attributes/staves") == "2"
    assert root_both.findtext(".//attributes/clef[@number='1']/sign") == "G"
    assert root_both.findtext(".//attributes/clef[@number='2']/sign") == "TAB"
    assert root_both.findtext(".//staff-details[@number='2']/capo") == "2"
    staff1_notes = [n for n in root_both.findall(".//note") if n.findtext("staff") == "1"]
    staff2_notes = [n for n in root_both.findall(".//note") if n.findtext("staff") == "2"]
    assert len(staff1_notes) == len(staff2_notes) == 4
    assert all(n.find("notations/technical/fingering") is not None for n in staff1_notes)
    assert all(n.find("notations/technical/string") is None for n in staff1_notes)
    assert all(n.find("notations/technical/string") is not None for n in staff2_notes)
    assert all(n.find("notations/technical/fingering") is None for n in staff2_notes)

    # 2. CLI inspect shows Tuning/Capo, and convert supports --guitar-staff both & --recompute-tab
    monkeypatch.setattr("transcribe_music.ingest.digital._find_musescore_executable", lambda: None)
    gp_path = write_gpif_file(tmp_path / "lead.gp")
    tab_path = write_ascii_tab_file(tmp_path / "riff.tab")

    runner = CliRunner()
    insp_res = runner.invoke(app, ["inspect", str(gp_path)])
    assert insp_res.exit_code == 0, insp_res.output
    assert "D2-A2-D3-G3-B3-E4" in insp_res.output

    cli_both = tmp_path / "cli_both.musicxml"
    conv_res = runner.invoke(
        app,
        [
            "convert",
            str(tab_path),
            "-i",
            "guitar",
            "--guitar-staff",
            "both",
            "--recompute-tab",
            "--no-cache",
            "-o",
            str(cli_both),
        ],
    )
    assert conv_res.exit_code == 0, conv_res.output
    assert ET.parse(cli_both).findtext(".//attributes/staves") == "2"
