"""Tests for digital score ingestion, multi-part selection, and batched PDF OMR fallback."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import guitarpro
import guitarpro.models as gm
import music21
import pymupdf
import pytest

from transcribe_music.export.musicxml import export_musicxml
from transcribe_music.ingest.digital import inspect_digital_score, load_digital_score
from transcribe_music.models import Instrument, OmrBackend
from transcribe_music.transcribe.omr_pdf import (
    run_pdf_omr,
    stitch_musicxml_pages,
)
from transcribe_music.transcribe.part_selector import select_and_adapt_parts


def _build_orchestral_cello_concerto_score() -> music21.stream.Score:
    """Create a multi-part score with Flute, Solo Cello (Bass->Tenor clef), and Cello Tutti."""
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title="Concerto Excerpt")

    flute = music21.stream.Part(id="Flute")
    flute.partName = "Flute"
    flute.insert(0, music21.instrument.Flute())
    for p in ("D5", "F#5", "A5", "D6"):
        flute.append(music21.note.Note(p, quarterLength=1.0))
    flute.makeMeasures(inPlace=True)

    solo_vc = music21.stream.Part(id="SoloCello")
    solo_vc.partName = "Solo Violoncello"
    solo_vc.insert(0, music21.instrument.Violoncello())
    m1 = music21.stream.Measure(number=1)
    m1.insert(0, music21.clef.BassClef())
    m1.insert(0, music21.meter.TimeSignature("4/4"))
    m1.insert(0, music21.key.KeySignature(2))
    for p in ("D3", "F#3", "A3", "D4"):
        m1.append(music21.note.Note(p, quarterLength=1.0))
    m2 = music21.stream.Measure(number=2)
    m2.insert(0, music21.clef.TenorClef())
    for p in ("F#4", "G4", "A4", "D4"):
        m2.append(music21.note.Note(p, quarterLength=1.0))
    solo_vc.append(m1)
    solo_vc.append(m2)

    tutti_vc = music21.stream.Part(id="TuttiCello")
    tutti_vc.partName = "Violoncello Tutti"
    tutti_vc.insert(0, music21.instrument.Violoncello())
    for p in ("D2", "D2", "A2", "D2", "D2", "D2", "A2", "D2"):
        tutti_vc.append(music21.note.Note(p, quarterLength=1.0))
    tutti_vc.makeMeasures(inPlace=True)

    score.insert(0, flute)
    score.insert(0, solo_vc)
    score.insert(0, tutti_vc)
    return score


def test_digital_formats_musicxml_mxl_krn_mei_abc_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. MusicXML & compressed .mxl
    src_score = _build_orchestral_cello_concerto_score()
    xml_path = tmp_path / "concerto.musicxml"
    mxl_path = tmp_path / "concerto_compressed.mxl"
    src_score.write("musicxml", fp=str(xml_path))
    src_score.write("mxl", fp=str(mxl_path))

    for p in (xml_path, mxl_path):
        loaded = load_digital_score(p)
        assert loaded.backend == "musicxml"
        assert loaded.is_sounding_pitch_source is False
        assert len(loaded.score.parts) == 3

    insp = inspect_digital_score(mxl_path)
    assert insp.title == "Concerto Excerpt"
    assert [pt.name for pt in insp.parts] == ["Flute", "Solo Violoncello", "Violoncello Tutti"]

    # 2. Humdrum .krn via converter21
    krn_path = tmp_path / "bach.krn"
    krn_path.write_text(
        "**kern\n*clefF4\n*k[f#c#]\n*M4/4\n4D\n4E\n4F#\n4G\n=2\n2A\n2d\n==\n*-\n",
        encoding="utf-8",
    )
    loaded_krn = load_digital_score(krn_path)
    assert loaded_krn.backend == "converter21_humdrum"
    assert len(list(loaded_krn.score.recurse().notes)) == 6

    # 3. ABC via converter21
    abc_path = tmp_path / "tune.abc"
    abc_path.write_text(
        "X:1\nT:Test Tune\nM:4/4\nL:1/4\nK:D\nD E F G | A2 d2 |]\n",
        encoding="utf-8",
    )
    loaded_abc = load_digital_score(abc_path)
    assert loaded_abc.backend == "converter21_abc"
    assert len(list(loaded_abc.score.recurse().notes)) == 6

    # 4. MEI via converter21
    mei_path = tmp_path / "violin.mei"
    v_score = music21.stream.Score()
    v_part = music21.stream.Part()
    v_part.insert(0, music21.instrument.Violin())
    for pitch in ("G3", "A3", "B3", "D4"):
        v_part.append(music21.note.Note(pitch, quarterLength=1.0))
    v_part.makeMeasures(inPlace=True)
    v_score.insert(0, v_part)
    v_score.write("mei", fp=str(mei_path))
    loaded_mei = load_digital_score(mei_path)
    assert loaded_mei.backend == "converter21_mei"
    assert [n.pitch.nameWithOctave for n in loaded_mei.score.recurse().notes] == [
        "G3",
        "A3",
        "B3",
        "D4",
    ]

    # 5. Native .capx, MuseScore bridges (.gp/.gpx/.mscz/.cap), LilyPond (.ly), and clear error paths
    import zipfile

    capx_path = tmp_path / "study.capx"
    with zipfile.ZipFile(capx_path, "w") as zf:
        zf.writestr(
            "score.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<score xmlns="http://www.capella.de/schema/capx/1.0">
  <info><title>Capella Study</title></info>
  <systems>
    <system>
      <staves>
        <staff layout="1">
          <voices>
            <voice>
              <noteObjects>
                <clefSign clef="G2"/>
                <chord><duration base="1/4"/><heads><head pitch="G4"/></heads></chord>
                <chord><duration base="1/4"/><heads><head pitch="A4"/></heads></chord>
                <chord><duration base="1/4"/><heads><head pitch="B4"/></heads></chord>
                <chord><duration base="1/4"/><heads><head pitch="D5"/></heads></chord>
              </noteObjects>
            </voice>
          </voices>
        </staff>
      </staves>
    </system>
  </systems>
</score>""",
        )
    loaded_capx = load_digital_score(capx_path)
    assert loaded_capx.backend == "capella_xml"
    assert len(list(loaded_capx.score.recurse().notes)) == 4

    pdf_path = tmp_path / "sheet.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="PDF fallback"):
        load_digital_score(pdf_path)

    unknown_path = tmp_path / "unknown.xyz"
    unknown_path.write_bytes(b"stub")
    with pytest.raises(ValueError, match="Unsupported digital score format"):
        load_digital_score(unknown_path)

    gpx_path = tmp_path / "modern.gpx"
    gpx_path.write_bytes(b"stub")
    mscz_path = tmp_path / "piece.mscz"
    mscz_path.write_bytes(b"stub")
    cap_path = tmp_path / "legacy.cap"
    cap_path.write_bytes(b"stub")

    monkeypatch.setattr("transcribe_music.ingest.digital._find_musescore_executable", lambda: None)
    with pytest.raises(RuntimeError, match="Guitar Pro 6/7/8"):
        load_digital_score(gpx_path)
    with pytest.raises(RuntimeError, match="MuseScore"):
        load_digital_score(mscz_path)
    with pytest.raises(RuntimeError, match="Capella"):
        load_digital_score(cap_path)

    # Mocked MuseScore bridge converts .gpx -> MusicXML seamlessly
    def _fake_mscore_run(cmd, **_kwargs):
        out_xml = Path(cmd[-1])
        v_score.write("musicxml", fp=str(out_xml))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("transcribe_music.ingest.digital._find_musescore_executable", lambda: "/usr/bin/mscore")
    monkeypatch.setattr("subprocess.run", _fake_mscore_run)
    loaded_gpx = load_digital_score(gpx_path)
    assert loaded_gpx.backend == "musescore"
    assert loaded_gpx.format_name == "gpx"

    # LilyPond (.ly) via python-ly and clean error when neither ly nor lilypond is installed
    ly_path = tmp_path / "etude.ly"
    ly_path.write_text('\\version "2.24.0"\n{ c4 d4 e4 f4 }\n', encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="LilyPond"):
        load_digital_score(ly_path)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ly" if name == "ly" else None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=xml_path.read_text(encoding="utf-8"), stderr=""),
    )
    loaded_ly = load_digital_score(ly_path)
    assert loaded_ly.backend == "python_ly"
    assert len(list(loaded_ly.score.recurse().notes)) >= 4


def test_guitarpro_gp5_ingestion_skips_drums_and_adapts_guitar_tab(tmp_path: Path) -> None:
    song = gm.Song(title="GP5 Study", artist="Test Composer", tempo=108)
    song.measureHeaders[0].keySignature = gm.KeySignature.DMajor

    gtr_track = song.tracks[0]
    gtr_track.name = "Nylon Guitar"
    gtr_track.channel.instrument = 24  # Acoustic Guitar (nylon)

    drum_track = gm.Track(song, number=2, name="Drums", isPercussionTrack=True)
    drum_track.channel.channel = 9
    drum_track.measures.append(gm.Measure(drum_track, song.measureHeaders[0]))
    song.tracks.append(drum_track)

    v0 = gtr_track.measures[0].voices[0]
    b1 = gm.Beat(v0, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
    b1.notes.append(gm.Note(b1, value=0, string=6, type=gm.NoteType.normal))  # E2 (40)
    b2 = gm.Beat(v0, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
    b2.notes.append(gm.Note(b2, value=0, string=5, type=gm.NoteType.normal))  # A2 (45)
    b3 = gm.Beat(v0, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
    b3.notes.append(gm.Note(b3, value=0, string=4, type=gm.NoteType.normal))  # D3 (50)
    b4 = gm.Beat(v0, duration=gm.Duration(value=4), status=gm.BeatStatus.normal)
    b4.notes.append(gm.Note(b4, value=0, string=1, type=gm.NoteType.normal))  # E4 (64)
    v0.beats.extend([b1, b2, b3, b4])

    gp5_path = tmp_path / "study.gp5"
    guitarpro.write(song, str(gp5_path))

    loaded = load_digital_score(gp5_path)
    assert loaded.backend == "pyguitarpro"
    assert loaded.is_sounding_pitch_source is True
    assert len(loaded.score.parts) == 1
    assert any("Skipped 1 percussion track" in w for w in loaded.warnings)

    # Adapting sounding GP5 pitches to Guitar written 8vb raises written pitch +12 (E2 -> E3)
    selected = select_and_adapt_parts(
        loaded.score,
        Instrument.GUITAR,
        is_sounding_pitch_source=loaded.is_sounding_pitch_source,
    )
    notes = list(selected.score.parts[0].recurse().notes)
    assert [n.pitch.nameWithOctave for n in notes] == ["E3", "A3", "D4", "E5"]
    first_clef = selected.score.parts[0].recurse().getElementsByClass(music21.clef.Clef).first()
    assert isinstance(first_clef, music21.clef.Treble8vbClef)


def test_part_selector_isolates_soloist_preserves_clefs_and_splits_piano_midi(
    tmp_path: Path,
) -> None:
    # 1. Multi-part concerto auto-selects Solo Violoncello over Cello Tutti & keeps TenorClef
    concerto = _build_orchestral_cello_concerto_score()
    selected = select_and_adapt_parts(concerto, Instrument.CELLO)
    assert "Solo Violoncello" in selected.selected_part
    assert len(selected.score.parts) == 1
    clefs = list(selected.score.parts[0].recurse().getElementsByClass(music21.clef.Clef))
    assert any(isinstance(c, music21.clef.BassClef) for c in clefs)
    assert any(isinstance(c, music21.clef.TenorClef) for c in clefs)

    # Explicit override selects Tutti or Flute by substring/index
    tutti_sel = select_and_adapt_parts(concerto, Instrument.CELLO, part_selector="tutti")
    assert "Violoncello Tutti" in tutti_sel.selected_part

    with pytest.raises(ValueError, match="Available parts"):
        select_and_adapt_parts(concerto, Instrument.CELLO, part_selector="nonexistent")

    # 2. Single-track Piano MIDI splits cleanly into Grand Staff RH (Treble) + LH (Bass)
    raw_midi_score = music21.stream.Score()
    p_flat = music21.stream.Part()
    p_flat.insert(0, music21.instrument.Piano())
    for pitch in ("C3", "E4", "G3", "C5"):
        p_flat.append(music21.note.Note(pitch, quarterLength=1.0))
    raw_midi_score.insert(0, p_flat)
    mid_path = tmp_path / "piano_single.mid"
    raw_midi_score.write("midi", fp=str(mid_path))

    loaded_mid = load_digital_score(mid_path)
    pno_sel = select_and_adapt_parts(
        loaded_mid.score,
        Instrument.PIANO,
        is_sounding_pitch_source=loaded_mid.is_sounding_pitch_source,
    )
    assert len(pno_sel.score.parts) == 2
    rh_notes = [n.pitch.nameWithOctave for n in pno_sel.score.parts[0].recurse().notes]
    lh_notes = [n.pitch.nameWithOctave for n in pno_sel.score.parts[1].recurse().notes]
    assert rh_notes == ["E4", "C5"]
    assert lh_notes == ["C3", "G3"]

    # 3. Guitar MIDI sounding E2=40 is adapted once to written E3=52 with Treble8vbClef
    gtr_raw = music21.stream.Score()
    gp = music21.stream.Part()
    gp.insert(0, music21.instrument.AcousticGuitar())
    for pitch in ("E2", "A2", "D3", "G3"):
        gp.append(music21.note.Note(pitch, quarterLength=1.0))
    gtr_raw.insert(0, gp)
    gtr_mid = tmp_path / "gtr.mid"
    gtr_raw.write("midi", fp=str(gtr_mid))

    loaded_gtr = load_digital_score(gtr_mid)
    gtr_sel = select_and_adapt_parts(
        loaded_gtr.score,
        Instrument.GUITAR,
        is_sounding_pitch_source=loaded_gtr.is_sounding_pitch_source,
    )
    assert [n.pitch.nameWithOctave for n in gtr_sel.score.parts[0].recurse().notes] == [
        "E3",
        "A3",
        "D4",
        "G4",
    ]
    xml_out = tmp_path / "gtr_out.musicxml"
    export_musicxml(gtr_sel.score, xml_out, Instrument.GUITAR)
    root = ET.parse(xml_out).getroot()
    assert root.findtext(".//attributes/clef/clef-octave-change") == "-1"


def test_batched_pdf_omr_invokes_homr_once_and_stitches_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "two_page.pdf"
    doc = pymupdf.open()
    for label in ("Page 1", "Page 2"):
        doc.new_page().insert_text((72, 72), label)
    doc.save(str(pdf_path))
    doc.close()

    subprocess_calls: list[list[str]] = []

    def _fake_subprocess_run(cmd, **_kwargs):
        subprocess_calls.append(list(cmd))
        img_names = [Path(arg) for arg in cmd if arg.endswith(".png")]
        assert len(img_names) == 2

        for idx, img in enumerate(img_names):
            s = music21.stream.Score()
            p = music21.stream.Part()
            pitches = ["C4", "D4", "E4", "F4"] if idx == 0 else ["G4", "A4", "B4", "C5"]
            for pch in pitches:
                p.append(music21.note.Note(pch, quarterLength=1.0))
            p.makeMeasures(inPlace=True)
            s.insert(0, p)
            s.write("musicxml", fp=str(img.with_suffix(".musicxml")))

        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/homr" if name == "homr" else None)
    monkeypatch.setattr("subprocess.run", _fake_subprocess_run)

    score, backend, warnings = run_pdf_omr(pdf_path, tmp_path / "work", Instrument.VIOLIN, dpi=150)
    assert backend == "homr"
    assert not warnings
    assert len(subprocess_calls) == 1
    out_xml = tmp_path / "omr_out.musicxml"
    metrics, _ = export_musicxml(score, out_xml, Instrument.VIOLIN)
    assert metrics.measure_count == 2
    assert metrics.note_count == 8

    # Also test direct stitch_musicxml_pages
    p1 = tmp_path / "page1.musicxml"
    p2 = tmp_path / "page2.musicxml"
    for path, pitches in [(p1, ["C4", "D4", "E4", "F4"]), (p2, ["G4", "A4", "B4", "C5"])]:
        s = music21.stream.Score()
        part = music21.stream.Part()
        for p in pitches:
            part.append(music21.note.Note(p, quarterLength=1.0))
        part.makeMeasures(inPlace=True)
        s.insert(0, part)
        s.write("musicxml", fp=str(path))

    merged = stitch_musicxml_pages([p1, p2], Instrument.VIOLIN, title="Merged")
    merged_xml = tmp_path / "merged.musicxml"
    m_metrics, _ = export_musicxml(merged, merged_xml, Instrument.VIOLIN)
    assert m_metrics.measure_count == 2
    assert m_metrics.note_count == 8
    root = ET.parse(merged_xml).getroot()
    assert [m.get("number") for m in root.findall(".//measure")] == ["1", "2"]


def test_audiveris_omr_preferred_over_homr_and_explicit_backend_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "cello_solo.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Cello Solo")
    doc.save(str(pdf_path))
    doc.close()

    # Explicit audiveris without binary errors cleanly
    monkeypatch.setattr("transcribe_music.transcribe.omr_pdf._find_audiveris_executable", lambda: None)
    with pytest.raises(RuntimeError, match="Audiveris"):
        run_pdf_omr(pdf_path, tmp_path / "w_err", Instrument.CELLO, omr_backend=OmrBackend.AUDIVERIS)

    # When Audiveris CLI is available, AUTO prefers Audiveris and passes PDF directly
    aud_calls: list[list[str]] = []

    def _fake_audiveris_run(cmd, **_kwargs):
        aud_calls.append(list(cmd))
        out_dir = Path(cmd[cmd.index("-output") + 1])
        sub_dir = out_dir / pdf_path.stem
        sub_dir.mkdir(parents=True, exist_ok=True)
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
        s.write("mxl", fp=str(sub_dir / f"{pdf_path.stem}.mxl"))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "transcribe_music.transcribe.omr_pdf._find_audiveris_executable",
        lambda: "/usr/local/bin/audiveris",
    )
    monkeypatch.setattr("subprocess.run", _fake_audiveris_run)

    score, backend, warnings = run_pdf_omr(pdf_path, tmp_path / "w_aud", Instrument.CELLO)
    assert backend == "audiveris"
    assert not warnings
    assert len(aud_calls) == 1
    assert "-batch" in aud_calls[0]
    assert str(pdf_path.resolve()) in aud_calls[0]
    clefs = list(score.parts[0].recurse().getElementsByClass(music21.clef.Clef))
    assert any(isinstance(c, music21.clef.TenorClef) for c in clefs)
