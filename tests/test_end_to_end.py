"""End-to-end integration tests across digital formats, PDF fallback, CLI, and SQLite cache."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import music21
import pytest
from conftest import (
    make_concerto_score,
    write_ascii_tab_file,
    write_gp5_file,
    write_gpif_file,
    write_pdf_file,
)

from transcribe_music import (
    GuitarStaffMode,
    Instrument,
    TranscriptionConfig,
    convert_music,
    inspect_score,
)
from transcribe_music.cli import main
from transcribe_music.storage.catalog import CatalogStore


def test_end_to_end_mxl_concerto_cache_and_pdf_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mxl_path = tmp_path / "haydn_sample.mxl"
    make_concerto_score().write("mxl", fp=str(mxl_path))
    assert len(inspect_score(mxl_path).parts) == 3

    out_xml = tmp_path / "haydn_cello.musicxml"
    cfg = TranscriptionConfig(instrument=Instrument.CELLO, cache_dir=tmp_path / ".cache")
    res1 = convert_music(mxl_path, Instrument.CELLO, output_path=out_xml, config=cfg)
    assert res1.cached is False
    assert res1.backend == "musicxml"
    assert res1.selected_part == "2: Solo Violoncello"
    assert (res1.note_count, res1.chord_count, res1.fingering_count) == (4, 2, 4)

    root = ET.parse(out_xml).getroot()
    assert [c.findtext("sign") for c in root.findall(".//attributes/clef")] == ["F", "C"]
    assert [h.findtext("root/root-step") for h in root.findall(".//harmony")] == ["D", "A"]

    out_xml.unlink()
    out_xml_2 = tmp_path / "haydn_cello_cached.musicxml"
    res2 = convert_music(mxl_path, Instrument.CELLO, output_path=out_xml_2, config=cfg)
    assert res2.cached is True and res2.job_hash == res1.job_hash and out_xml_2.exists()

    pdf_path = write_pdf_file(tmp_path / "guitar_piece.pdf")

    def _fake_homr(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        for idx, img in enumerate(Path(arg) for arg in cmd if arg.endswith(".png")):
            s = music21.stream.Score()
            p = music21.stream.Part()
            p.insert(0, music21.clef.Treble8vbClef())
            for pitch in ("C4", "E4", "G4", "C5") if idx == 0 else ("G3", "B3", "D4", "G4"):
                p.append(music21.note.Note(pitch, quarterLength=1.0))
            p.makeMeasures(inPlace=True)
            s.insert(0, p)
            s.write("musicxml", fp=str(img.with_suffix(".musicxml")))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/homr" if name == "homr" else None)
    monkeypatch.setattr("subprocess.run", _fake_homr)

    pdf_out = tmp_path / "guitar_pdf.musicxml"
    res_pdf = convert_music(
        pdf_path,
        Instrument.GUITAR,
        output_path=pdf_out,
        config=TranscriptionConfig(instrument=Instrument.GUITAR, use_cache=False, pdf_dpi=150),
    )
    assert res_pdf.backend == "homr" and res_pdf.note_count == 8
    assert len(ET.parse(pdf_out).findall(".//technical/fret")) == 8


def test_end_to_end_guitar_staff_modes_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
            instrument=Instrument.GUITAR, guitar_staff=GuitarStaffMode.NOTATION, cache_dir=cache_dir
        ),
    )
    res_tab = convert_music(
        gp5_path,
        Instrument.GUITAR,
        output_path=tmp_path / "dropd_tab.musicxml",
        config=TranscriptionConfig(
            instrument=Instrument.GUITAR, guitar_staff=GuitarStaffMode.TAB, cache_dir=cache_dir
        ),
    )
    res_both = convert_music(
        gp5_path,
        Instrument.GUITAR,
        output_path=tmp_path / "dropd_both.musicxml",
        config=TranscriptionConfig(
            instrument=Instrument.GUITAR, guitar_staff=GuitarStaffMode.BOTH, cache_dir=cache_dir
        ),
    )

    assert len({res_not.job_hash, res_tab.job_hash, res_both.job_hash}) == 3
    assert res_not.note_count == res_tab.note_count == res_both.note_count == 4
    assert res_not.fingering_count == res_both.fingering_count == 4

    root_not = ET.parse(res_not.musicxml_path).getroot()
    assert [s.text for s in root_not.findall(".//technical/string")] == ["6", "6", "6", "6"]
    assert [f.text for f in root_not.findall(".//technical/fret")] == ["0", "3", "5", "7"]
    assert root_not.findtext(".//staff-details/capo") == "2"
    assert root_not.findtext(".//staff-details/staff-tuning[@line='1']/tuning-step") == "D"

    root_tab = ET.parse(res_tab.musicxml_path).getroot()
    assert root_tab.findtext(".//attributes/clef/sign") == "TAB"
    assert root_tab.find(".//attributes/transpose") is None
    assert root_tab.findtext(".//measure[@number='1']/note[1]/pitch/octave") == "2"

    root_both = ET.parse(res_both.musicxml_path).getroot()
    assert root_both.findtext(".//attributes/staves") == "2"
    assert root_both.findtext(".//staff-details[@number='2']/capo") == "2"
    staff1_notes = [n for n in root_both.findall(".//note") if n.findtext("staff") == "1"]
    staff2_notes = [n for n in root_both.findall(".//note") if n.findtext("staff") == "2"]
    assert len(staff1_notes) == len(staff2_notes) == 4
    assert all(n.find("notations/technical/fingering") is not None for n in staff1_notes)
    assert all(n.find("notations/technical/string") is None for n in staff1_notes)
    assert all(n.find("notations/technical/string") is not None for n in staff2_notes)
    assert all(n.find("notations/technical/fingering") is None for n in staff2_notes)

    monkeypatch.setattr("transcribe_music.ingest.digital._find_musescore_executable", lambda: None)
    gp_path = write_gpif_file(tmp_path / "lead.gp")
    tab_path = write_ascii_tab_file(tmp_path / "riff.tab")

    assert main(["inspect", str(gp_path)]) == 0
    assert "D2-A2-D3-G3-B3-E4" in capsys.readouterr().out

    nocache_out = tmp_path / "nocache.musicxml"
    assert (
        main(
            [
                "convert",
                str(tab_path),
                "-i",
                "guitar",
                "--guitar-staff",
                "both",
                "--recompute-tab",
                "--no-cache",
                "--cache-dir",
                str(cache_dir),
                "-o",
                str(nocache_out),
            ]
        )
        == 0
    )
    assert ET.parse(nocache_out).findtext(".//attributes/staves") == "2"

    assert main(["cache", "list", "--cache-dir", str(cache_dir)]) == 0
    assert "guitar" in capsys.readouterr().out
    assert len(CatalogStore(cache_dir).list_recent_jobs()) == 3

    assert main(["cache", "clear", "--cache-dir", str(cache_dir)]) == 0
    assert "Cleared 3 job records" in capsys.readouterr().out
