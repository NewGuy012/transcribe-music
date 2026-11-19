"""Tests for digital score ingestion, part selection, and OMR backend guards."""

from __future__ import annotations

import builtins
import zipfile
from pathlib import Path
from types import SimpleNamespace

import music21
import pytest
from conftest import (
    make_concerto_score,
    note_tab_tags,
    write_ascii_tab_file,
    write_gp5_file,
    write_gpif_file,
    write_pdf_file,
)

from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.digital import (
    ensure_converter21_registered,
    inspect_digital_score,
    load_digital_score,
)
from transcribe_music.models import Instrument, OmrBackend
from transcribe_music.transcribe.omr_pdf import run_pdf_omr
from transcribe_music.transcribe.part_selector import select_and_adapt_parts


def test_digital_formats_bridges_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src_score = make_concerto_score()
    xml_path = tmp_path / "concerto.musicxml"
    mxl_path = tmp_path / "concerto_compressed.mxl"
    src_score.write("musicxml", fp=str(xml_path))
    src_score.write("mxl", fp=str(mxl_path))

    for path in (xml_path, mxl_path):
        loaded = load_digital_score(path)
        assert loaded.backend == "musicxml"
        assert loaded.is_sounding_pitch_source is False
        assert len(loaded.score.parts) == 3

    insp = inspect_digital_score(mxl_path)
    assert insp.title == "Haydn D Major Sample"
    assert [pt.name for pt in insp.parts] == [
        "Orchestra Strings",
        "Solo Violoncello",
        "Violoncello Tutti",
    ]

    krn_path = tmp_path / "bach.krn"
    krn_path.write_text("**kern\n*clefF4\n*M4/4\n4D\n4E\n4F#\n4G\n==\n*-\n", encoding="utf-8")
    assert load_digital_score(krn_path).backend == "converter21_humdrum"

    abc_path = tmp_path / "tune.abc"
    abc_path.write_text("X:1\nT:Test\nM:4/4\nL:1/4\nK:D\nD E F G |]\n", encoding="utf-8")
    assert load_digital_score(abc_path).backend == "converter21_abc"

    v_score = music21.stream.Score()
    v_part = music21.stream.Part()
    v_part.insert(0, music21.instrument.Violin())
    for pitch in ("G3", "A3", "B3", "D4"):
        v_part.append(music21.note.Note(pitch, quarterLength=1.0))
    v_part.makeMeasures(inPlace=True)
    v_score.insert(0, v_part)
    mei_path = tmp_path / "violin.mei"
    v_score.write("mei", fp=str(mei_path))
    assert load_digital_score(mei_path).backend == "converter21_mei"

    capx_path = tmp_path / "study.capx"
    with zipfile.ZipFile(capx_path, "w") as zf:
        zf.writestr(
            "score.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<score xmlns="http://www.capella.de/schema/capx/1.0">
  <info><title>Capella Study</title></info>
  <systems><system><staves><staff layout="1"><voices><voice><noteObjects>
    <clefSign clef="G2"/>
    <chord><duration base="1/4"/><heads><head pitch="G4"/></heads></chord>
  </noteObjects></voice></voices></staff></staves></system></systems>
</score>""",
        )
    assert load_digital_score(capx_path).backend == "capella_xml"

    pdf_stub = tmp_path / "sheet.pdf"
    pdf_stub.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="got PDF"):
        load_digital_score(pdf_stub)

    bad_stub = tmp_path / "unknown.xyz"
    bad_stub.write_bytes(b"stub")
    with pytest.raises(ValueError, match="Unsupported digital score format"):
        load_digital_score(bad_stub)

    gpx_path = tmp_path / "modern.gpx"
    gpx_path.write_bytes(b"stub")
    monkeypatch.setattr("transcribe_music.ingest.digital._find_musescore_executable", lambda: None)
    with pytest.raises(RuntimeError, match="Guitar Pro 6/7/8"):
        load_digital_score(gpx_path)

    monkeypatch.setattr(
        "transcribe_music.ingest.digital._find_musescore_executable", lambda: "/usr/bin/mscore"
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **_kw: (
            v_score.write("musicxml", fp=cmd[-1]),
            SimpleNamespace(returncode=0, stderr="", stdout=""),
        )[1],
    )
    loaded_gpx = load_digital_score(gpx_path)
    assert loaded_gpx.backend == "musescore"
    assert loaded_gpx.format_name == "gpx"

    ly_path = tmp_path / "etude.ly"
    ly_path.write_text('\\version "2.24.0"\n{ c4 d4 e4 f4 }\n', encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="LilyPond"):
        load_digital_score(ly_path)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ly" if name == "ly" else None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=0, stdout=xml_path.read_text(encoding="utf-8"), stderr=""
        ),
    )
    assert load_digital_score(ly_path).backend == "python_ly"

    real_import = builtins.__import__

    def _block_extras(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("guitarpro") or name == "converter21":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    gp5_stub = tmp_path / "missing_extra.gp5"
    gp5_stub.write_bytes(b"stub")
    monkeypatch.setattr(builtins, "__import__", _block_extras)
    with pytest.raises(RuntimeError, match="uv sync --extra gp"):
        load_digital_score(gp5_stub)

    ensure_converter21_registered.cache_clear()
    with pytest.raises(RuntimeError, match="uv sync --extra formats"):
        load_digital_score(krn_path)
    ensure_converter21_registered.cache_clear()


def test_part_selection_piano_midi_split_and_guitar_adaptation(tmp_path: Path) -> None:
    concerto = make_concerto_score()
    tutti_sel = select_and_adapt_parts(concerto, Instrument.CELLO, part_selector="tutti")
    assert "Violoncello Tutti" in tutti_sel.selected_part
    with pytest.raises(ValueError, match="Available parts"):
        select_and_adapt_parts(concerto, Instrument.CELLO, part_selector="nonexistent")

    raw_midi = music21.stream.Score()
    p_flat = music21.stream.Part()
    p_flat.insert(0, music21.instrument.Piano())
    for pitch in ("C3", "E4", "G3", "C5"):
        p_flat.append(music21.note.Note(pitch, quarterLength=1.0))
    raw_midi.insert(0, p_flat)
    mid_path = tmp_path / "piano_single.mid"
    raw_midi.write("midi", fp=str(mid_path))

    pno_sel = select_and_adapt_parts(
        load_digital_score(mid_path).score, Instrument.PIANO, is_sounding_pitch_source=True
    )
    assert [n.pitch.nameWithOctave for n in pno_sel.score.parts[0].recurse().notes] == ["E4", "C5"]
    assert [n.pitch.nameWithOctave for n in pno_sel.score.parts[1].recurse().notes] == ["C3", "G3"]

    gtr_score = music21.stream.Score()
    rhythm = music21.stream.Part(id="Rhythm")
    rhythm.partName = "Rhythm Guitar"
    rhythm.insert(0, music21.instrument.AcousticGuitar())
    for p in ("E2", "B2", "E3", "E2"):
        rhythm.append(music21.note.Note(p, quarterLength=1.0))
    rhythm.makeMeasures(inPlace=True)
    lead = music21.stream.Part(id="Lead")
    lead.partName = "Lead Guitar"
    lead.insert(0, music21.instrument.ElectricGuitar())
    for p in ("E4", "G4", "A4", "B4"):
        lead.append(music21.note.Note(p, quarterLength=1.0))
    lead.makeMeasures(inPlace=True)
    gtr_score.insert(0, rhythm)
    gtr_score.insert(0, lead)

    assert "Lead Guitar" in select_and_adapt_parts(gtr_score, Instrument.GUITAR, is_sounding_pitch_source=True).selected_part
    assert "Rhythm Guitar" in select_and_adapt_parts(gtr_score, Instrument.GUITAR, part_selector="rhythm", is_sounding_pitch_source=True).selected_part

    # Cello MIDI auto-clef switching: Bass -> Tenor -> Bass hysteresis on sounding pitch; MusicXML untouched
    cel_s = music21.stream.Score()
    cel_p = music21.stream.Part()
    cel_p.insert(0, music21.instrument.Violoncello())
    for pitch in (
        "C2", "G2", "C3", "E3",  # M1: low bass
        "G3", "B3", "D4", "A4",  # M2: high tenor (min >= 50, max >= 65)
        "D3", "E3", "F#3", "G3",  # M3: drops below G3 and <= D4 -> back to Bass
    ):
        cel_p.append(music21.note.Note(pitch, quarterLength=1.0))
    cel_p.makeMeasures(inPlace=True)
    cel_s.insert(0, cel_p)

    cel_mid_path = tmp_path / "cello_clefs.mid"
    cel_s.write("midi", fp=str(cel_mid_path))
    cel_loaded = load_digital_score(cel_mid_path)
    cel_sel = select_and_adapt_parts(
        cel_loaded.score,
        Instrument.CELLO,
        is_sounding_pitch_source=cel_loaded.is_sounding_pitch_source,
    )
    cel_clefs = [
        (m.number, type(c).__name__)
        for m in cel_sel.score.parts[0].getElementsByClass(music21.stream.Measure)
        for c in m.getElementsByClass(music21.clef.Clef)
    ]
    assert cel_clefs == [(1, "BassClef"), (2, "TenorClef"), (3, "BassClef")]

    # Authored MusicXML clefs remain untouched (is_sounding_pitch_source=False)
    authored_s = music21.stream.Score()
    authored_p = music21.stream.Part()
    am1 = music21.stream.Measure(number=1)
    am1.insert(0, music21.clef.TrebleClef())
    for pitch in ("A4", "B4", "C5", "D5"):
        am1.append(music21.note.Note(pitch, quarterLength=1.0))
    authored_p.append(am1)
    authored_s.insert(0, authored_p)
    authored_sel = select_and_adapt_parts(authored_s, Instrument.CELLO, is_sounding_pitch_source=False)
    assert [
        type(c).__name__
        for c in authored_sel.score.parts[0].recurse().getElementsByClass(music21.clef.Clef)
    ] == ["TrebleClef"]


def test_gp5_gpif_and_ascii_tab_parsers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gp5_chord_path = write_gp5_file(
        tmp_path / "chord_order.gp5",
        chord_beats=(((1, 0), (2, 1), (3, 0), (4, 2), (5, 3)), ((6, 0),), ((6, 2),), ((6, 3),)),
        with_drums=True,
        tuning=(64, 59, 55, 50, 45, 38),
        capo=2,
        with_effects=True,
    )
    insp_gp5 = inspect_digital_score(gp5_chord_path)
    assert insp_gp5.parts[0].tuning == "D2-A2-D3-G3-B3-E4"
    assert insp_gp5.parts[0].capo == 2

    loaded_gp5 = load_digital_score(gp5_chord_path)
    assert len(loaded_gp5.score.parts) == 1
    assert any("Skipped 1 percussion track" in w for w in loaded_gp5.warnings)
    first_chord = next(loaded_gp5.score.parts[0].recurse().getElementsByClass(music21.chord.Chord))
    assert note_tab_tags(first_chord) == ([5, 4, 3, 2, 1], [3, 2, 0, 1, 0])

    gp5_xml = tmp_path / "gp5_effects.musicxml"
    export_musicxml(loaded_gp5.score, gp5_xml, Instrument.GUITAR)
    xml_txt = gp5_xml.read_text(encoding="utf-8")
    for tag in ("<pull-off", "<glissando", "<bend-alter>2</bend-alter>", "<harmonic", "P.M.", "<repeat"):
        assert tag in xml_txt

    monkeypatch.setattr("transcribe_music.ingest.digital._find_musescore_executable", lambda: None)
    gp_path = write_gpif_file(tmp_path / "modern.gp")
    loaded_gp = load_digital_score(gp_path)
    assert loaded_gp.backend == "gpif"
    assert len(loaded_gp.score.parts) == 1
    assert any("Skipped 1 percussion track" in w for w in loaded_gp.warnings)
    gp_xml = tmp_path / "gpif_out.musicxml"
    export_musicxml(loaded_gp.score, gp_xml, Instrument.GUITAR)
    gp_txt = gp_xml.read_text(encoding="utf-8")
    assert "<hammer-on" in gp_txt and "<bend-alter>2</bend-alter>" in gp_txt

    evil_gp = write_gpif_file(tmp_path / "evil.gp", doctype="<!DOCTYPE foo>")
    with pytest.raises(ValueError, match="Unsafe XML DTD"):
        load_digital_score(evil_gp)

    bad_gp = tmp_path / "legacy_container.gp"
    bad_gp.write_bytes(b"not a zip")
    with pytest.raises(RuntimeError, match="MuseScore"):
        load_digital_score(bad_gp)

    tab_path = write_ascii_tab_file(tmp_path / "riff.tab")
    loaded_tab = load_digital_score(tab_path)
    assert loaded_tab.backend == "ascii_tab"
    tab_xml = tmp_path / "tab_out.musicxml"
    tab_metrics, tab_warns = export_musicxml(loaded_tab.score, tab_xml, Instrument.GUITAR)
    assert not tab_warns and tab_metrics.note_count == 10
    tab_txt = tab_xml.read_text(encoding="utf-8")
    assert "<hammer-on" in tab_txt and "<glissando" in tab_txt and "<bend-alter>2</bend-alter>" in tab_txt

    not_tab = tmp_path / "readme.txt"
    not_tab.write_text("Just lyrics.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid ASCII guitar tab"):
        load_digital_score(not_tab)


def test_pdf_omr_audiveris_and_guitar_tab_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = write_pdf_file(tmp_path / "two_page.pdf", labels=("Page 1",))

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    with pytest.raises(RuntimeError, match="Audiveris"):
        run_pdf_omr(pdf_path, tmp_path / "w_err", Instrument.CELLO, omr_backend=OmrBackend.AUDIVERIS)

    def _fake_audiveris_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        out_dir = Path(cmd[cmd.index("-output") + 1]) / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        s = music21.stream.Score()
        p = music21.stream.Part()
        m1 = music21.stream.Measure(number=1)
        m1.insert(0, music21.clef.BassClef())
        for pch in ("D3", "F#3", "A3", "D4"):
            m1.append(music21.note.Note(pch, quarterLength=1.0))
        m2 = music21.stream.Measure(number=2)
        m2.insert(0, music21.clef.TenorClef())
        for pch in ("F#4", "G4", "A4", "D4"):
            m2.append(music21.note.Note(pch, quarterLength=1.0))
        p.append(m1)
        p.append(m2)
        s.insert(0, p)
        s.write("mxl", fp=str(out_dir / f"{pdf_path.stem}.mxl"))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "transcribe_music.transcribe.omr_pdf._find_audiveris_executable",
        lambda: "/usr/local/bin/audiveris",
    )
    monkeypatch.setattr("subprocess.run", _fake_audiveris_run)
    aud_score, aud_backend, aud_warnings = run_pdf_omr(pdf_path, tmp_path / "w_aud", Instrument.CELLO)
    assert aud_backend == "audiveris" and not aud_warnings
    assert any(
        isinstance(c, music21.clef.TenorClef)
        for c in aud_score.parts[0].recurse().getElementsByClass(music21.clef.Clef)
    )

    def _fake_dual_homr(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        img = next(Path(arg) for arg in cmd if arg.endswith(".png"))
        s = music21.stream.Score()
        p_not = music21.stream.Part(id="Notation")
        p_not.insert(0, music21.clef.Treble8vbClef())
        p_not.append(music21.note.Note("E3", quarterLength=4.0))
        p_not.makeMeasures(inPlace=True)
        p_tab = music21.stream.Part(id="Tab")
        p_tab.insert(0, music21.clef.TabClef())
        p_tab.append(music21.note.Note("C4", quarterLength=4.0))
        p_tab.makeMeasures(inPlace=True)
        s.insert(0, p_not)
        s.insert(0, p_tab)
        s.write("musicxml", fp=str(img.with_suffix(".musicxml")))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/homr" if name == "homr" else None)
    monkeypatch.setattr("subprocess.run", _fake_dual_homr)

    omr_score, _, omr_warns = run_pdf_omr(pdf_path, tmp_path / "w_dual", Instrument.GUITAR, dpi=150)
    assert len(omr_score.parts) == 1
    assert any("Filtered out OMR-recognized TAB staff" in w for w in omr_warns)

    def _fake_tab_only_homr(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        img = next(Path(arg) for arg in cmd if arg.endswith(".png"))
        p_tab = music21.stream.Part(id="TabOnly")
        p_tab.insert(0, music21.clef.TabClef())
        p_tab.append(music21.note.Note("E4", quarterLength=4.0))
        p_tab.makeMeasures(inPlace=True)
        music21.stream.Score([p_tab]).write("musicxml", fp=str(img.with_suffix(".musicxml")))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("subprocess.run", _fake_tab_only_homr)
    _, _, tab_only_warns = run_pdf_omr(pdf_path, tmp_path / "w_tabonly", Instrument.GUITAR, dpi=150)
    assert any("guitar TAB or unrecognized staves" in w for w in tab_only_warns)
